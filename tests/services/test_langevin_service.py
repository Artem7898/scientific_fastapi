"""A Langevin run persists auditable coordinates and transport diagnostics."""

import hashlib
import json
import math
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from scientific_api.domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from scientific_api.domain.simulation import SimulationCancelled
from scientific_api.models.langevin_engine import LangevinSimulator, langevin_moments
from scientific_api.research.transport_analysis import analyze_transport, load_trajectory_state
from scientific_api.services import langevin_service
from scientific_api.services.langevin_service import run_langevin_simulation


def parameters(**changes) -> LangevinParameters:
    return LangevinParameters(
        **{
            "initial_temperature": 2.0,
            "bath_temperature": 1.0,
            "num_particles": 64,
            "time_step": 0.1,
            "num_steps": 30,
            "sample_stride": 7,
            "seed": 2026,
            **changes,
        }
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_run_repeats_coordinates_and_persists_verified_sampling(tmp_path, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    configuration = parameters(backend=backend, seed=2**64 - 1)
    first = run_langevin_simulation(
        configuration, task_id="first", attempt_id="owner", artifacts_dir=tmp_path
    )
    second = run_langevin_simulation(
        configuration, task_id="second", attempt_id="other-owner", artifacts_dir=tmp_path
    )
    for name in (
        "thermal_temperature",
        "kinetic_energy",
        "thermal_temperature_standard_error",
        "final_msd",
        "final_centered_msd",
        "mean_displacement",
        "mean_velocity",
        "diffusion",
    ):
        assert first[name] == second[name]
    manifest = json.loads((tmp_path / first["artifacts"]["manifest"]).read_text())
    assert manifest["outputs"] == first
    assert manifest["model"]["version"] == LANGEVIN_MODEL_VERSION
    assert manifest["execution"]["backend"] == backend
    assert manifest["rng"]["seed"] == 2**64 - 1
    assert manifest["sampling"]["sample_steps"] == [0, 7, 14, 21, 28, 30]
    assert manifest["sampling"]["coordinate_convention"] == "unwrapped"
    assert manifest["uncertainty"]["independent_units"].startswith("Particles")
    for entry in manifest["artifacts"]:
        artifact = tmp_path / entry["path"]
        assert entry["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert entry["bytes"] == artifact.stat().st_size
    record = next(
        entry for entry in manifest["artifacts"] if entry["path"].endswith("trajectory.npz")
    )
    trajectory = load_trajectory_state(
        tmp_path / record["path"],
        expected_sha256=record["sha256"],
        expected_particles=configuration.num_particles,
        expected_sample_steps=configuration.sampling_steps(),
        expected_time_step=configuration.time_step,
    )
    with (
        np.load(tmp_path / first["artifacts"]["final_state"], allow_pickle=False) as a,
        np.load(tmp_path / second["artifacts"]["final_state"], allow_pickle=False) as b,
    ):
        assert set(a.files) == {"positions", "unwrapped_positions", "velocities"}
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(trajectory["positions"][-1], a["unwrapped_positions"])
    assert not list(tmp_path.rglob(".arrays-*"))
    assert not list(tmp_path.rglob(".manifest-*"))


def test_periodic_presentation_does_not_saturate_msd(tmp_path):
    configuration = parameters(
        initial_temperature=1e-6,
        bath_temperature=1e-6,
        initial_position=(0.25, 0.5),
        initial_mean_velocity=(5.0, -4.0),
        acceleration=(2.0, 0.5),
        boundary="periodic",
        box_size=(0.2, 0.3),
    )
    result = run_langevin_simulation(configuration, task_id="periodic", artifacts_dir=tmp_path)
    with np.load(tmp_path / result["artifacts"]["final_state"], allow_pickle=False) as state:
        wrapped, unwrapped = state["positions"], state["unwrapped_positions"]
        assert (wrapped >= 0).all()
        assert (wrapped < np.asarray(configuration.box_size)).all()
        np.testing.assert_array_equal(wrapped, np.remainder(unwrapped, configuration.box_size))
        assert float(np.max(np.abs(unwrapped))) > 1.0
        assert result["final_msd"] > 10.0
        assert result["final_centered_msd"] < 1e-4


def test_transient_thermal_temperature_uses_peculiar_velocity_and_degrees_of_freedom(tmp_path):
    configuration = parameters(
        num_particles=512,
        initial_mean_velocity=(10.0, -7.0),
        acceleration=(25.0, 15.0),
    )
    result = run_langevin_simulation(configuration, task_id="thermal", artifacts_dir=tmp_path)
    with np.load(tmp_path / result["artifacts"]["final_state"], allow_pickle=False) as state:
        velocity = state["velocities"]
    peculiar = velocity - velocity.mean(axis=0)
    expected = (
        configuration.mass * float(np.sum(peculiar**2)) / (2 * (configuration.num_particles - 1))
    )
    assert math.isclose(result["thermal_temperature"], expected, rel_tol=1e-14)
    lab_temperature = result["kinetic_energy"] / configuration.num_particles
    assert lab_temperature > 100 * result["thermal_temperature"]
    theory = langevin_moments(configuration.num_steps * configuration.time_step, configuration)
    assert result["thermal_temperature_standard_error"] == (
        configuration.mass
        * theory["velocity_variance_per_component"]
        / math.sqrt(configuration.num_particles - 1)
    )


def test_saved_transport_is_independently_recomputed_without_advancing_physics(
    tmp_path, monkeypatch
):
    configuration = parameters(num_steps=120, time_step=0.1, sample_stride=10)
    result = run_langevin_simulation(configuration, task_id="transport", artifacts_dir=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("A saved trajectory analysis must not advance the simulation")

    monkeypatch.setattr(LangevinSimulator, "advance", forbidden)
    with np.load(tmp_path / result["artifacts"]["trajectory"], allow_pickle=False) as state:
        rebuilt = analyze_transport(
            state["positions"], state["sample_steps"], parameters=configuration
        )
    saved = json.loads((tmp_path / result["artifacts"]["transport"]).read_text())
    assert saved == rebuilt
    assert result["diffusion"] == rebuilt["diffusion"]
    assert rebuilt["diffusion"]["status"] == "available"
    assert rebuilt["diffusion"]["standard_error"] >= 0


def test_sampling_schedule_does_not_change_terminal_seeded_state(tmp_path):
    configuration = parameters()
    first = run_langevin_simulation(configuration, task_id="stride7", artifacts_dir=tmp_path)
    second = run_langevin_simulation(
        replace(configuration, sample_stride=11), task_id="stride11", artifacts_dir=tmp_path
    )
    with (
        np.load(tmp_path / first["artifacts"]["final_state"], allow_pickle=False) as a,
        np.load(tmp_path / second["artifacts"]["final_state"], allow_pickle=False) as b,
    ):
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])


class RecordingTracker:
    def __init__(self, *, fail_metrics=False, fail_end=False, fail_mark=False):
        self.events = []
        self.fail_metrics = fail_metrics
        self.fail_end = fail_end
        self.fail_mark = fail_mark

    def start_run(self, run_name):
        self.events.append("start")

    def log_params(self, parameters):
        self.events.append("params")
        assert parameters["model_version"] == LANGEVIN_MODEL_VERSION

    def log_metrics(self, metrics):
        self.events.append("metrics")
        assert "thermal_temperature" in metrics
        assert "final_temperature" not in metrics
        if self.fail_metrics:
            raise RuntimeError("metrics unavailable")

    def mark_failed(self):
        self.events.append("failed")
        if self.fail_mark:
            raise RuntimeError("mark unavailable")

    def end_run(self):
        self.events.append("end")
        if self.fail_end:
            raise RuntimeError("end unavailable")


@pytest.mark.parametrize("failure", [False, True])
def test_tracking_ends_and_metric_failure_is_not_a_success(tmp_path, failure):
    tracker = RecordingTracker(fail_metrics=failure)
    if failure:
        with pytest.raises(RuntimeError, match="metrics unavailable"):
            run_langevin_simulation(
                parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker
            )
        assert tracker.events == ["start", "params", "metrics", "failed", "end"]
        assert not (tmp_path / "tracked" / "manifest.json").exists()
    else:
        result = run_langevin_simulation(
            parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker
        )
        assert result["status"] == "completed"
        assert tracker.events == ["start", "params", "metrics", "end"]


def test_tracking_cleanup_failure_preserves_primary_exception(tmp_path):
    tracker = RecordingTracker(fail_metrics=True, fail_end=True, fail_mark=True)
    with pytest.raises(RuntimeError, match="metrics unavailable"):
        run_langevin_simulation(
            parameters(), task_id="failed", artifacts_dir=tmp_path, tracker=tracker
        )
    assert tracker.events == ["start", "params", "metrics", "failed", "end"]


def test_successful_calculation_does_not_swallow_tracking_end_failure(tmp_path):
    tracker = RecordingTracker(fail_end=True)
    with pytest.raises(RuntimeError, match="end unavailable"):
        run_langevin_simulation(
            parameters(), task_id="failed", artifacts_dir=tmp_path, tracker=tracker
        )
    assert tracker.events == ["start", "params", "metrics", "end", "failed"]


def test_cancelled_before_execution_creates_no_attempt_directory(tmp_path):
    cancelled = Event()
    cancelled.set()
    with pytest.raises(SimulationCancelled):
        run_langevin_simulation(
            parameters(), task_id="cancelled", artifacts_dir=tmp_path, cancel_event=cancelled
        )
    assert not (tmp_path / "cancelled").exists()


def test_cancelled_during_transition_never_publishes_manifest(tmp_path, monkeypatch):
    cancelled = Event()
    original = LangevinSimulator.advance

    def interrupt(simulator):
        original(simulator)
        cancelled.set()

    monkeypatch.setattr(LangevinSimulator, "advance", interrupt)
    tracker = RecordingTracker()
    with pytest.raises(SimulationCancelled):
        run_langevin_simulation(
            parameters(),
            task_id="cancelled",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
            tracker=tracker,
        )
    assert tracker.events == ["start", "params", "failed", "end"]
    assert not list((tmp_path / "cancelled").glob("*.npz"))
    assert not (tmp_path / "cancelled" / "manifest.json").exists()


def test_progress_precedes_persistence_and_allows_cancellation(tmp_path):
    cancelled = Event()
    updates = []

    def progress(done, total, phase):
        updates.append((done, total, phase))
        if phase == "saving_artifacts":
            cancelled.set()

    with pytest.raises(SimulationCancelled):
        run_langevin_simulation(
            parameters(),
            task_id="cancelled",
            artifacts_dir=tmp_path,
            progress_callback=progress,
            cancel_event=cancelled,
        )
    assert updates[0] == (0, 30, "computing")
    assert updates[-2:] == [(30, 30, "computing"), (30, 30, "saving_artifacts")]
    assert not (tmp_path / "cancelled" / "manifest.json").exists()
    assert not list((tmp_path / "cancelled").glob("*.npz"))


def test_nonfinite_state_is_never_persisted(tmp_path, monkeypatch):
    original = langevin_service.create_langevin_simulator

    def invalid_state(configuration):
        simulator = original(configuration)
        simulator.velocities[0, 0] = np.nan
        return simulator

    monkeypatch.setattr(langevin_service, "create_langevin_simulator", invalid_state)
    with pytest.raises((FloatingPointError, ValueError)):
        run_langevin_simulation(parameters(), task_id="invalid", artifacts_dir=tmp_path)
    assert not (tmp_path / "invalid" / "manifest.json").exists()


def test_npz_serialization_failure_does_not_replace_complete_artifact(tmp_path, monkeypatch):
    path = tmp_path / "state.npz"
    original = b"an existing complete artifact"
    path.write_bytes(original)

    def fail(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="storage unavailable"):
        langevin_service._write_arrays(path, {"positions": np.zeros((3, 2))})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".arrays-*"))


def test_state_writer_rejects_object_arrays_before_serialization(tmp_path):
    path = tmp_path / "state.npz"
    with pytest.raises(ValueError, match="Python objects"):
        langevin_service._write_arrays(path, {"positions": np.array([object()], dtype=object)})
    assert not path.exists()
    assert not list(tmp_path.glob(".arrays-*"))


@pytest.mark.parametrize("identifier", ["../escape", "", "x/y", "а", "x" * 129])
@pytest.mark.parametrize("field", ["task_id", "attempt_id"])
def test_artifact_identifiers_are_bounded_and_cannot_escape(tmp_path, identifier, field):
    arguments = {"task_id": "run", "attempt_id": "attempt", field: identifier}
    with pytest.raises(ValueError, match=field):
        run_langevin_simulation(parameters(), artifacts_dir=tmp_path, **arguments)


@pytest.mark.parametrize("attempt", [False, True])
def test_artifact_directory_symlinks_are_rejected(tmp_path, attempt):
    target = tmp_path / "target"
    target.mkdir()
    task = tmp_path / "run"
    if attempt:
        task.mkdir()
        (task / "owner").symlink_to(target, target_is_directory=True)
    else:
        task.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_langevin_simulation(
            parameters(), task_id="run", attempt_id="owner", artifacts_dir=tmp_path
        )
