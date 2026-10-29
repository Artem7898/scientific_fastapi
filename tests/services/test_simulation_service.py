import hashlib
import json
import math
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest

from scientific_api.domain.simulation import (
    MODEL_VERSION,
    SimulationCancelled,
    SimulationParameters,
)
from scientific_api.models.physics_engine import IdealGasSimulator
from scientific_api.services.simulation_service import run_simulation


def test_run_persists_auditable_state_and_reproduces_seed(tmp_path: Path) -> None:
    parameters = SimulationParameters(300.0, 250, time_step=0.1, num_steps=20, seed=2026)
    first = run_simulation(parameters, task_id="first", artifacts_dir=tmp_path)
    second = run_simulation(parameters, task_id="second", artifacts_dir=tmp_path)
    for metric in ("final_temperature", "total_energy", "temperature_standard_error"):
        assert first[metric] == second[metric]
        assert math.isfinite(first[metric])
    assert first["task_id"] == "first"
    assert first["temperature_standard_error"] == 300.0 / math.sqrt(250)
    manifest = json.loads((tmp_path / first["artifacts"]["manifest"]).read_text())
    assert manifest["parameters"]["seed"] == 2026
    assert manifest["outputs"] == first
    assert manifest["model"]["version"] == MODEL_VERSION
    assert manifest["rng"]["bit_generator"] == "PCG64"
    assert manifest["environment"]["numpy"] == np.__version__
    assert len(manifest["environment"]["source_sha256"]) == 64
    state_path = tmp_path / first["artifacts"]["final_state"]
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(state_path.read_bytes()).hexdigest()
    with (
        np.load(state_path, allow_pickle=False) as initial,
        np.load(tmp_path / second["artifacts"]["final_state"], allow_pickle=False) as repeated,
    ):
        velocities = initial["velocities"]
        np.testing.assert_array_equal(velocities, repeated["velocities"])
        energy_from_artifact = float(0.5 * np.einsum("ij,ij->", velocities, velocities))
        assert first["total_energy"] == energy_from_artifact
    assert not list(tmp_path.rglob(".state-*"))
    assert not list(tmp_path.rglob(".manifest-*"))


def test_stale_attempt_cannot_overwrite_committed_attempt_artifacts(tmp_path: Path) -> None:
    accepted = run_simulation(
        SimulationParameters(300.0, 20, num_steps=2, seed=42),
        task_id="shared-task",
        attempt_id="winning-owner",
        artifacts_dir=tmp_path,
    )
    accepted_artifacts = {
        name: (tmp_path / relative).read_bytes() for name, relative in accepted["artifacts"].items()
    }
    # A lease-expired worker can finish after the accepted worker. Its unique
    # owner isolates its writes even though both attempts use the same task ID.
    stale = run_simulation(
        SimulationParameters(300.0, 20, num_steps=2, seed=43),
        task_id="shared-task",
        attempt_id="stale-owner",
        artifacts_dir=tmp_path,
    )
    assert accepted["task_id"] == stale["task_id"] == "shared-task"
    assert accepted["attempt_id"] == "winning-owner"
    assert stale["attempt_id"] == "stale-owner"
    for name, relative in accepted["artifacts"].items():
        assert relative != stale["artifacts"][name]
        assert (tmp_path / relative).read_bytes() == accepted_artifacts[name]
    for result in (accepted, stale):
        manifest = json.loads((tmp_path / result["artifacts"]["manifest"]).read_text())
        assert manifest["attempt_id"] == result["attempt_id"]
        assert manifest["outputs"]["attempt_id"] == result["attempt_id"]


@pytest.mark.parametrize("attempt_id", ["../escape", "", "x/y", "а", "x" * 129])
def test_attempt_identifier_cannot_escape_artifact_root(tmp_path: Path, attempt_id: str) -> None:
    with pytest.raises(ValueError, match="attempt_id"):
        run_simulation(
            SimulationParameters(300.0, 10),
            task_id="task",
            attempt_id=attempt_id,
            artifacts_dir=tmp_path,
        )


def test_attempt_symlink_cannot_escape_task_directory(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (task_dir / "attempt").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_simulation(
            SimulationParameters(300.0, 10),
            task_id="task",
            attempt_id="attempt",
            artifacts_dir=tmp_path,
        )


class RecordingTracker:
    def __init__(self, fail_metrics: bool = False) -> None:
        self.events: list[str] = []
        self.fail_metrics = fail_metrics

    def start_run(self, run_name: str) -> None:
        self.events.append("start")

    def log_params(self, params: dict[str, Any]) -> None:
        self.events.append("params")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.events.append("metrics")
        if self.fail_metrics:
            raise RuntimeError("tracker unavailable")

    def end_run(self) -> None:
        self.events.append("end")


@pytest.mark.parametrize("fail_metrics", [False, True])
def test_tracker_always_ends_even_when_logging_fails(tmp_path: Path, fail_metrics: bool) -> None:
    tracker = RecordingTracker(fail_metrics=fail_metrics)
    parameters = SimulationParameters(300.0, 10, num_steps=2)
    if fail_metrics:
        with pytest.raises(RuntimeError, match="tracker unavailable"):
            run_simulation(parameters, task_id="tracked", artifacts_dir=tmp_path, tracker=tracker)
        assert not (tmp_path / "tracked" / "manifest.json").exists()
    else:
        run_simulation(parameters, task_id="tracked", artifacts_dir=tmp_path, tracker=tracker)
    assert tracker.events == ["start", "params", "metrics", "end"]


def test_failed_run_marks_tracker_failed_and_preserves_primary_error(tmp_path: Path) -> None:
    class FailedCleanupTracker(RecordingTracker):
        def mark_failed(self) -> None:
            self.events.append("failed")
            raise RuntimeError("mark failed unavailable")

        def end_run(self) -> None:
            super().end_run()
            raise RuntimeError("cleanup unavailable")

    tracker = FailedCleanupTracker(fail_metrics=True)
    with pytest.raises(RuntimeError, match="tracker unavailable"):
        run_simulation(
            SimulationParameters(300.0, 10, num_steps=2),
            task_id="failed",
            artifacts_dir=tmp_path,
            tracker=tracker,
        )
    assert tracker.events == ["start", "params", "metrics", "failed", "end"]


def test_successful_computation_does_not_swallow_tracking_cleanup_failure(tmp_path: Path) -> None:
    class FailedCleanupTracker(RecordingTracker):
        def mark_failed(self) -> None:
            self.events.append("failed")

        def end_run(self) -> None:
            super().end_run()
            raise RuntimeError("cleanup unavailable")

    tracker = FailedCleanupTracker()
    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        run_simulation(
            SimulationParameters(300.0, 10, num_steps=2),
            task_id="failed",
            artifacts_dir=tmp_path,
            tracker=tracker,
        )
    assert tracker.events == ["start", "params", "metrics", "end", "failed"]


def test_cancelled_run_is_not_published(tmp_path: Path) -> None:
    event = Event()
    event.set()
    with pytest.raises(SimulationCancelled):
        run_simulation(
            SimulationParameters(300.0, 10),
            task_id="cancelled",
            artifacts_dir=tmp_path,
            cancel_event=event,
        )
    assert not (tmp_path / "cancelled" / "manifest.json").exists()


def test_cancellation_during_a_run_ends_tracking_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = Event()
    tracker = RecordingTracker()
    original_advance = IdealGasSimulator.advance

    def cancel_after_step(simulator: IdealGasSimulator) -> None:
        original_advance(simulator)
        event.set()

    monkeypatch.setattr(IdealGasSimulator, "advance", cancel_after_step)
    with pytest.raises(SimulationCancelled):
        run_simulation(
            SimulationParameters(300.0, 10),
            task_id="cancelled",
            artifacts_dir=tmp_path,
            tracker=tracker,
            cancel_event=event,
        )
    assert tracker.events == ["start", "params", "end"]
    assert not (tmp_path / "cancelled" / "manifest.json").exists()


def test_manifest_json_write_is_atomic_on_serialization_failure(tmp_path: Path) -> None:
    from scientific_api.services.simulation_service import _write_manifest

    destination = tmp_path / "manifest.json"
    destination.write_text('{"original": true}\n')
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_manifest(destination, {"invalid": float("nan")})
    assert destination.read_text() == '{"original": true}\n'
    assert not list(tmp_path.glob(".manifest-*"))


@pytest.mark.parametrize("task_id", ["../escape", "", "x/y", "а", "x" * 129])
def test_task_identifier_cannot_escape_artifact_root(tmp_path: Path, task_id: str) -> None:
    with pytest.raises(ValueError, match="task_id"):
        run_simulation(SimulationParameters(300.0, 10), task_id=task_id, artifacts_dir=tmp_path)


def test_existing_symlink_cannot_escape_artifact_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "run").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_simulation(SimulationParameters(300.0, 10), task_id="run", artifacts_dir=tmp_path)
