"""An LJ attempt preserves signed observables and auditable state artifacts."""

import hashlib
import json
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled
from scientific_api.models.lennard_jones_engine import LennardJonesSimulator
from scientific_api.research.lj_analysis import analyze_lj_observables
from scientific_api.research.transport_analysis import load_trajectory_state
from scientific_api.services import lennard_jones_service
from scientific_api.services.lennard_jones_service import run_lj_simulation


def parameters(**changes):
    return LennardJonesParameters(
        **{
            "num_particles": 64,
            "box_size": (10.0, 10.0),
            "initial_temperature": 0.1,
            "time_step": 0.001,
            "num_steps": 12,
            "sample_stride": 5,
            "seed": 2026,
            **changes,
        }
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_seed_repeats_and_every_artifact_is_identified_by_manifest(tmp_path, backend, ensemble):
    if backend == "numba":
        pytest.importorskip("numba")
    configuration = parameters(
        backend=backend,
        ensemble=ensemble,
        bath_temperature=0.1 if ensemble == "nvt" else None,
        seed=2**64 - 1,
    )
    first = run_lj_simulation(
        configuration, task_id="first", attempt_id="winner", artifacts_dir=tmp_path
    )
    second = run_lj_simulation(
        configuration, task_id="second", attempt_id="other", artifacts_dir=tmp_path
    )
    assert first["model_version"] == LJ_MODEL_VERSION
    assert first["ensemble"] == ensemble
    assert first["potential_energy"] < 0
    assert first["total_energy"] < 0
    for key in (
        "thermal_temperature",
        "kinetic_energy",
        "potential_energy",
        "total_energy",
        "pressure",
        "diagnostics",
    ):
        assert first[key] == second[key]
    manifest = json.loads((tmp_path / first["artifacts"]["manifest"]).read_text())
    assert manifest["outputs"] == first
    assert manifest["model"]["version"] == LJ_MODEL_VERSION
    assert manifest["model"]["potential"]["tail_correction"] is None
    assert manifest["model"]["initialization"]["canonical_sample"] is False
    assert manifest["execution"]["backend"] == backend
    assert manifest["execution"]["integrator"] == (
        "velocity-verlet" if ensemble == "nve" else "baoab"
    )
    assert manifest["rng"]["seed"] == 2**64 - 1
    assert manifest["sampling"]["sample_steps"] == [0, 5, 10, 12]
    assert manifest["diagnostics"]["analysis_pair_evaluations"] == 64 * 63 // 2
    assert manifest["analysis"]["rdf_snapshot"] == "terminal frame only"
    for record in manifest["artifacts"]:
        path = tmp_path / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["bytes"] == path.stat().st_size
    record = next(
        record for record in manifest["artifacts"] if record["path"].endswith("trajectory.npz")
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
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(trajectory["positions"][-1], a["unwrapped_positions"])
        assert ((a["positions"] >= 0) & (a["positions"] < 10)).all()
    assert not list(tmp_path.rglob(".arrays-*"))
    assert not list(tmp_path.rglob(".manifest-*"))


def test_saved_analysis_reproduces_without_advancing_physics(tmp_path, monkeypatch):
    configuration = parameters()
    result = run_lj_simulation(configuration, task_id="analysis", artifacts_dir=tmp_path)
    saved = json.loads((tmp_path / result["artifacts"]["observables"]).read_text())

    def forbidden(*args, **kwargs):
        raise AssertionError("Reanalysis must not run the interacting dynamics")

    monkeypatch.setattr(LennardJonesSimulator, "advance", forbidden)
    with np.load(tmp_path / result["artifacts"]["trajectory"], allow_pickle=False) as trajectory:
        restored = analyze_lj_observables(
            trajectory["positions"],
            trajectory["sample_steps"],
            {
                name: np.asarray(values, dtype=np.float64)
                for name, values in saved["series"].items()
            },
            parameters=configuration,
        )
    assert restored == saved
    assert restored["energy_drift"]["applicable"] is True
    assert restored["rdf"]["dimensions"] == 2


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_terminal_artifacts_independently_recover_kinetic_and_direct_pair_observables(
    tmp_path, backend
):
    if backend == "numba":
        pytest.importorskip("numba")
    from scientific_api.models.lj_forces import compute_forces

    configuration = parameters(backend=backend, interaction_backend="verlet", num_steps=50)
    result = run_lj_simulation(configuration, task_id="physics", artifacts_dir=tmp_path)
    with np.load(tmp_path / result["artifacts"]["final_state"], allow_pickle=False) as state:
        velocities = state["velocities"]
        positions = state["unwrapped_positions"]
        kinetic = 0.5 * configuration.mass * float(np.sum(velocities**2))
        peculiar = velocities - velocities.mean(axis=0)
        temperature = (
            configuration.mass
            * float(np.sum(peculiar**2))
            / (2 * (configuration.num_particles - 1))
        )
        # Reconstruct using all unordered pairs, independently of the stored
        # Verlet list and the backend that produced the committed artifact.
        direct = compute_forces(
            positions,
            configuration.box_size,
            configuration.epsilon,
            configuration.sigma,
            configuration.cutoff_radius,
            backend="numpy",
        )
    assert result["kinetic_energy"] == pytest.approx(kinetic, rel=1e-12)
    assert result["thermal_temperature"] == pytest.approx(temperature, rel=1e-12)
    assert result["potential_energy"] == pytest.approx(direct.potential_energy, rel=1e-11)
    assert result["total_energy"] == pytest.approx(kinetic + direct.potential_energy, rel=1e-11)
    pressure = (2 * kinetic + direct.virial) / (2 * np.prod(configuration.box_size))
    assert result["pressure"] == pytest.approx(pressure, rel=1e-11)
    analysis = json.loads((tmp_path / result["artifacts"]["observables"]).read_text())
    assert analysis["series"]["virial"][-1] == pytest.approx(direct.virial, rel=1e-11)


def test_sampling_does_not_change_seeded_dynamics_and_progress_is_observational(tmp_path):
    configuration = parameters()
    first = run_lj_simulation(configuration, task_id="stride5", artifacts_dir=tmp_path)
    updates = []
    second = run_lj_simulation(
        replace(configuration, sample_stride=4),
        task_id="stride4",
        artifacts_dir=tmp_path,
        progress_callback=lambda done, total, phase: updates.append((done, total, phase)),
    )
    with (
        np.load(tmp_path / first["artifacts"]["final_state"]) as a,
        np.load(tmp_path / second["artifacts"]["final_state"]) as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert updates[0] == (0, 12, "computing")
    assert updates[-2:] == [(12, 12, "computing"), (12, 12, "saving_artifacts")]
    assert all(phase != "completed" for _, _, phase in updates)


def test_direct_and_neighbor_attempts_agree_over_the_same_short_trajectory(tmp_path):
    configuration = parameters(num_steps=20)
    results = [
        run_lj_simulation(
            replace(configuration, interaction_backend=backend),
            task_id=backend,
            artifacts_dir=tmp_path,
        )
        for backend in ("direct", "verlet")
    ]
    for key in ("kinetic_energy", "potential_energy", "total_energy", "pressure"):
        assert results[0][key] == pytest.approx(results[1][key], rel=1e-11, abs=1e-11)
    with (
        np.load(tmp_path / results[0]["artifacts"]["final_state"]) as a,
        np.load(tmp_path / results[1]["artifacts"]["final_state"]) as b,
    ):
        for key in a.files:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-11, atol=1e-11)
    assert results[0]["diagnostics"]["neighbor_rebuilds"] == 0
    assert results[1]["diagnostics"]["neighbor_rebuilds"] >= 1


def test_analysis_pair_budget_is_reserved_before_rdf_work(tmp_path, monkeypatch):
    from scientific_api.models.lj_forces import PairBudgetExceeded

    def exhausted(self, count):
        assert count == 64 * 63 // 2
        raise PairBudgetExceeded("analysis budget exhausted")

    def forbidden(*args, **kwargs):
        raise AssertionError("RDF work started without pair-budget reservation")

    monkeypatch.setattr(LennardJonesSimulator, "reserve_analysis_pairs", exhausted)
    monkeypatch.setattr(lennard_jones_service, "analyze_lj_observables", forbidden)
    with pytest.raises(PairBudgetExceeded, match="analysis budget"):
        run_lj_simulation(parameters(), task_id="budget", artifacts_dir=tmp_path)
    assert not (tmp_path / "budget" / "manifest.json").exists()


class RecordingTracker:
    def __init__(self, *, fail_metrics=False, fail_end=False):
        self.events = []
        self.fail_metrics = fail_metrics
        self.fail_end = fail_end

    def start_run(self, run_name):
        self.events.append("start")

    def log_params(self, values):
        self.events.append("params")
        assert values["model_version"] == LJ_MODEL_VERSION

    def log_metrics(self, values):
        self.events.append("metrics")
        assert values["potential_energy"] < 0
        assert values["total_energy"] < 0
        assert "temperature_standard_error" not in values
        if self.fail_metrics:
            raise RuntimeError("metrics unavailable")

    def mark_failed(self):
        self.events.append("failed")

    def end_run(self):
        self.events.append("end")
        if self.fail_end:
            raise RuntimeError("tracking cleanup unavailable")


@pytest.mark.parametrize("failure", [False, True])
def test_signed_tracking_metrics_and_failure_cleanup(tmp_path, failure):
    tracker = RecordingTracker(fail_metrics=failure)
    if failure:
        with pytest.raises(RuntimeError, match="metrics unavailable"):
            run_lj_simulation(
                parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker
            )
        assert tracker.events == ["start", "params", "metrics", "failed", "end"]
        assert not (tmp_path / "tracked" / "manifest.json").exists()
    else:
        run_lj_simulation(parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker)
        assert tracker.events == ["start", "params", "metrics", "end"]


def test_failed_tracking_cleanup_preserves_the_primary_error(tmp_path):
    tracker = RecordingTracker(fail_metrics=True, fail_end=True)
    with pytest.raises(RuntimeError, match="metrics unavailable"):
        run_lj_simulation(parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker)
    assert tracker.events == ["start", "params", "metrics", "failed", "end"]


def test_tracking_cleanup_failure_is_not_reported_as_success(tmp_path):
    tracker = RecordingTracker(fail_end=True)
    with pytest.raises(RuntimeError, match="tracking cleanup unavailable"):
        run_lj_simulation(parameters(), task_id="tracked", artifacts_dir=tmp_path, tracker=tracker)
    assert tracker.events == ["start", "params", "metrics", "end", "failed"]


def test_cancellation_before_computation_creates_no_run_directory(tmp_path):
    cancelled = Event()
    cancelled.set()
    with pytest.raises(SimulationCancelled):
        run_lj_simulation(
            parameters(), task_id="cancelled", artifacts_dir=tmp_path, cancel_event=cancelled
        )
    assert not (tmp_path / "cancelled").exists()


def test_cancellation_after_one_step_preserves_step_boundaries(tmp_path, monkeypatch):
    cancelled = Event()
    original = LennardJonesSimulator.advance
    calls = []

    def advance(self):
        original(self)
        calls.append(True)
        cancelled.set()

    monkeypatch.setattr(LennardJonesSimulator, "advance", advance)
    with pytest.raises(SimulationCancelled):
        run_lj_simulation(
            parameters(), task_id="cancelled", artifacts_dir=tmp_path, cancel_event=cancelled
        )
    assert len(calls) == 1
    assert not (tmp_path / "cancelled" / "manifest.json").exists()


def test_cancellation_before_artifact_saving_publishes_no_artifacts(tmp_path):
    cancelled = Event()

    def progress(done, total, phase):
        if phase == "saving_artifacts":
            cancelled.set()

    with pytest.raises(SimulationCancelled):
        run_lj_simulation(
            parameters(),
            task_id="cancelled",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
            progress_callback=progress,
        )
    assert not list((tmp_path / "cancelled").glob("*.npz"))
    assert not (tmp_path / "cancelled" / "manifest.json").exists()


def test_atomic_array_write_failure_does_not_publish_a_success_manifest(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("artifact filesystem unavailable")

    monkeypatch.setattr(lennard_jones_service, "_write_arrays", unavailable)
    with pytest.raises(OSError, match="filesystem unavailable"):
        run_lj_simulation(parameters(), task_id="failed", artifacts_dir=tmp_path)
    assert not (tmp_path / "failed" / "manifest.json").exists()


@pytest.mark.parametrize("identifier", ["../escape", "", "a/b", "а", "a" * 129])
def test_task_identifier_stays_inside_artifact_root(tmp_path, identifier):
    with pytest.raises(ValueError, match="task_id"):
        run_lj_simulation(parameters(), task_id=identifier, artifacts_dir=tmp_path)


def test_attempt_symlink_is_not_followed(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "task" / "attempt").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_lj_simulation(
            parameters(), task_id="task", attempt_id="attempt", artifacts_dir=tmp_path
        )
