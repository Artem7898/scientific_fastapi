"""Successful publication requires evidence for the exact committed attempt."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from scientific_api.core import queue_manager
from scientific_api.core.config import Settings
from scientific_api.core.experiment_tracker import NullTracker
from scientific_api.core.queue_manager import QueueManager
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.telemetry import Telemetry
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.services.analysis_service import create_velocity_analysis
from scientific_api.services.simulation_service import run_simulation

MODELS = ("ideal-gas-ou-v1", "langevin-free-v1", "lennard-jones-2d-v1")


def _parameters(model: str) -> SimulationParameters | LangevinParameters | LennardJonesParameters:
    if model == "langevin-free-v1":
        return LangevinParameters(
            initial_temperature=1.0,
            bath_temperature=1.0,
            num_particles=16,
            num_steps=3,
            sample_stride=1,
            seed=7,
        )
    if model == "lennard-jones-2d-v1":
        return LennardJonesParameters(
            num_particles=36,
            box_size=(8.0, 8.0),
            initial_temperature=0.1,
            num_steps=3,
            sample_stride=1,
            seed=7,
        )
    return SimulationParameters(initial_temperature=1.0, num_particles=16, num_steps=3, seed=7)


def _reject_publication(*args: Any, **kwargs: Any) -> None:
    raise OSError("envelope publication failed")


@pytest.mark.parametrize("model", MODELS)
def test_direct_service_never_returns_success_when_envelope_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    from scientific_api.services import result_envelope_service

    monkeypatch.setattr(result_envelope_service, "publish_result_envelope", _reject_publication)
    with pytest.raises(OSError, match="envelope publication failed"):
        run_simulation(
            _parameters(model), task_id="run", attempt_id="owner", artifacts_dir=tmp_path
        )
    assert not (tmp_path / "run" / "owner" / "envelope.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "fault",
    [
        "missing-envelope",
        "missing-marker-and-envelope",
        "missing-state",
        "corrupt-state",
        "wrong-owner",
    ],
)
async def test_queue_rejects_success_with_missing_corrupt_or_foreign_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, fault: str
) -> None:
    parameters = _parameters(model)
    backend = MemoryTaskStore()
    backend.submit("run", {"model": model, **asdict(parameters)}, {})
    job = backend.claim()
    assert job is not None
    settings_values: dict[str, Any] = {
        "_env_file": None,
        "artifacts_dir": tmp_path,
        "redis_uri": "",
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
        "worker_count": 1,
    }
    settings = Settings(**settings_values)
    manager = QueueManager(backend, settings, NullTracker(), Telemetry())
    completed_calls: list[dict[str, Any]] = []

    def damaged_service(configuration: Any, **kwargs: Any) -> dict[str, Any]:
        if fault == "wrong-owner":
            kwargs["attempt_id"] = "another-owner"
        result = run_simulation(configuration, **kwargs)
        completed_calls.append(result)
        if fault in {"missing-envelope", "missing-marker-and-envelope"}:
            (tmp_path / "run" / job.owner / "envelope.json").unlink()
            if fault == "missing-marker-and-envelope":
                path = tmp_path / result["artifacts"]["manifest"]
                manifest = json.loads(path.read_bytes())
                manifest.pop("result_envelope")
                path.write_text(json.dumps(manifest))
        elif fault == "missing-state":
            (tmp_path / result["artifacts"]["final_state"]).unlink()
        elif fault == "corrupt-state":
            state = tmp_path / result["artifacts"]["final_state"]
            original = state.read_bytes()
            state.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        return result

    monkeypatch.setattr(queue_manager, "run_simulation", damaged_service)
    try:
        await asyncio.wait_for(manager._execute(job), timeout=10)
        assert len(completed_calls) == 1, (
            "The scientific service must succeed before fault injection"
        )
        terminal = backend.get("run")
        assert terminal is not None
        assert terminal["status"] == "failed"
        assert "artifacts" not in terminal
        assert "envelope publication failed" not in terminal.get("error", "")
        page = backend.events("run")
        assert page is not None
        assert all(event["data"]["status"] != "completed" for event in page["events"])
    finally:
        await manager.stop()
        backend.close()


@pytest.mark.parametrize("model", MODELS)
def test_preexisting_cancellation_does_not_publish_success_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    from scientific_api.services import result_envelope_service

    def forbidden_publication(*args: Any, **kwargs: Any) -> None:
        pytest.fail("A cancelled simulation must not publish successful evidence")

    cancelled = Event()
    cancelled.set()
    monkeypatch.setattr(result_envelope_service, "publish_result_envelope", forbidden_publication)
    with pytest.raises(SimulationCancelled):
        run_simulation(
            _parameters(model),
            task_id="cancelled",
            attempt_id="owner",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
        )
    assert not list(tmp_path.rglob("envelope.json"))


def test_failed_analysis_evidence_publication_removes_only_new_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scientific_api.services import result_envelope_service

    source = run_simulation(
        _parameters("ideal-gas-ou-v1"),
        task_id="source",
        attempt_id="owner",
        artifacts_dir=tmp_path,
    )
    existing_analysis = create_velocity_analysis(source, artifacts_dir=tmp_path, bins=8, formats=())
    unrelated = tmp_path / "unrelated" / "keep.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("Keep this user-owned file unchanged.\n")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(result_envelope_service, "publish_result_envelope", _reject_publication)
    with pytest.raises(OSError, match="envelope publication failed"):
        create_velocity_analysis(source, artifacts_dir=tmp_path, bins=12, formats=())
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    analyses = tmp_path / "source" / "owner" / "analyses"
    assert sorted(path.name for path in analyses.iterdir()) == [existing_analysis["analysis_id"]]
    assert not list(tmp_path.rglob(".analysis-*"))
