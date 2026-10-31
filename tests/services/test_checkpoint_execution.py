"""A restored scientific service continues state and the original sample schedule."""

import importlib.util
import json
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.models.lennard_jones_engine import LennardJonesSimulator
from scientific_api.research.lj_analysis import SERIES_NAMES
from scientific_api.services import langevin_service, lennard_jones_service, simulation_service
from scientific_api.services.simulation_service import run_simulation

BACKENDS = [
    "numpy",
    pytest.param(
        "numba",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
        ),
    ),
]
MODELS = ["ou", "langevin", "lj-nve-direct", "lj-nve-verlet", "lj-nvt-direct", "lj-nvt-verlet"]


class InjectedFailure(RuntimeError):
    pass


def parameters(model, backend="numpy"):
    if model == "ou":
        return SimulationParameters(
            initial_temperature=1.5,
            num_particles=17,
            time_step=0.1,
            num_steps=23,
            seed=2**64 - 1,
            backend=backend,
        )
    if model == "langevin":
        return LangevinParameters(
            initial_temperature=2.0,
            bath_temperature=1.0,
            num_particles=17,
            mass=1.2,
            friction=0.7,
            time_step=0.2,
            num_steps=23,
            sample_stride=5,
            acceleration=(0.3, -0.2),
            initial_mean_velocity=(0.8, -0.9),
            boundary="periodic",
            box_size=(0.3, 0.4),
            seed=2**64 - 1,
            backend=backend,
        )
    _, ensemble, interaction = model.split("-")
    return LennardJonesParameters(
        num_particles=36,
        box_size=(8.0, 8.0),
        initial_temperature=0.1,
        ensemble=ensemble,
        bath_temperature=0.1 if ensemble == "nvt" else None,
        friction=0.7 if ensemble == "nvt" else None,
        interaction_backend=interaction,
        time_step=0.001,
        num_steps=23,
        sample_stride=5,
        seed=2**64 - 1,
        backend=backend,
    )


def control(configuration, root, attempt, *, register=None, loaded=None, interval=7):
    from scientific_api.services.checkpoint_service import CheckpointControl

    return CheckpointControl(
        configuration,
        task_id="scientific-run",
        attempt_id=attempt,
        artifacts_dir=root,
        interval_steps=interval,
        loaded=loaded,
        register=register,
    )


def load(descriptor, configuration, root):
    from scientific_api.services.checkpoint_service import load_checkpoint

    return load_checkpoint(descriptor, parameters=configuration, artifacts_dir=root)


def forbid_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A resumed service must not run the fresh simulator factory")

    monkeypatch.setattr(simulation_service, "create_simulator", forbidden)
    monkeypatch.setattr(langevin_service, "create_langevin_simulator", forbidden)
    monkeypatch.setattr(lennard_jones_service, "create_lj_simulator", forbidden)


def assert_science_identical(root, baseline, restored):
    for key, value in baseline.items():
        if key not in {"task_id", "attempt_id", "artifacts"}:
            assert restored[key] == value, key
    assert set(baseline["artifacts"]) == set(restored["artifacts"])
    for name in ("final_state", "trajectory"):
        if name not in baseline["artifacts"]:
            continue
        with (
            np.load(root / baseline["artifacts"][name], allow_pickle=False) as first,
            np.load(root / restored["artifacts"][name], allow_pickle=False) as second,
        ):
            assert first.files == second.files
            for field in first.files:
                assert first[field].dtype == second[field].dtype
                assert first[field].shape == second[field].shape
                # Include signed zero: compare bytes, not just numeric equality.
                assert first[field].tobytes() == second[field].tobytes(), (name, field)
    for name in ("transport", "observables"):
        if name in baseline["artifacts"]:
            assert (root / baseline["artifacts"][name]).read_bytes() == (
                root / restored["artifacts"][name]
            ).read_bytes()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stop_step", [0, 7])
def test_checkpoint_resume_preserves_scientific_bytes_samples_and_diagnostics(
    tmp_path, monkeypatch, model, backend, stop_step
):
    configuration = parameters(model, backend)
    baseline = run_simulation(
        configuration, task_id="baseline", attempt_id="uninterrupted", artifacts_dir=tmp_path
    )
    recorded = []

    def interrupted(descriptor):
        recorded.append(descriptor)
        if descriptor["completed_steps"] == stop_step:
            raise InjectedFailure("Worker disappeared after the checkpoint was recorded")

    with pytest.raises(InjectedFailure):
        run_simulation(
            configuration,
            task_id="scientific-run",
            attempt_id="first",
            artifacts_dir=tmp_path,
            checkpoint_control=control(configuration, tmp_path, "first", register=interrupted),
        )
    checkpoint = load(recorded[-1], configuration, tmp_path)
    assert checkpoint.completed_steps == stop_step
    if model != "ou":
        expected_samples = 1 if stop_step == 0 else 2
        # Global frames 0 and 5 stay at their original scientific time origin.
        assert checkpoint.next_sample == expected_samples
        assert checkpoint.samples["trajectory"].shape == (
            expected_samples,
            configuration.num_particles,
            2,
        )
        if model.startswith("lj"):
            assert set(checkpoint.samples) == {"trajectory", *SERIES_NAMES}
    forbid_initialization(monkeypatch)
    progress = []
    continuation = control(configuration, tmp_path, "second", loaded=checkpoint, interval=0)
    restored = run_simulation(
        configuration,
        task_id="scientific-run",
        attempt_id="second",
        artifacts_dir=tmp_path,
        progress_callback=lambda done, total, phase: progress.append((done, total, phase)),
        checkpoint_control=continuation,
    )
    assert continuation.loaded is None
    assert progress[0] == (stop_step, configuration.num_steps, "computing")
    assert progress[-1] == (configuration.num_steps, configuration.num_steps, "saving_artifacts")
    assert_science_identical(tmp_path, baseline, restored)
    manifest = json.loads((tmp_path / restored["artifacts"]["manifest"]).read_text())
    assert manifest["outputs"] == restored
    assert manifest["continuation"]["mode"] == "resumed_run"


@pytest.mark.parametrize("backend", BACKENDS)
def test_final_lj_checkpoint_precedes_terminal_rdf_and_can_republish_without_advancing(
    tmp_path, monkeypatch, backend
):
    configuration = replace(parameters("lj-nvt-verlet", backend), num_steps=10, sample_stride=3)
    baseline = run_simulation(
        configuration, task_id="baseline", attempt_id="uninterrupted", artifacts_dir=tmp_path
    )
    recorded = []
    original_write = lennard_jones_service._write_arrays

    def failed_artifact(*args, **kwargs):
        raise OSError("Temporary artifact storage failure after terminal analysis")

    with monkeypatch.context() as patch:
        patch.setattr(lennard_jones_service, "_write_arrays", failed_artifact)
        with pytest.raises(OSError, match="Temporary artifact"):
            run_simulation(
                configuration,
                task_id="scientific-run",
                attempt_id="first",
                artifacts_dir=tmp_path,
                checkpoint_control=control(
                    configuration, tmp_path, "first", register=recorded.append, interval=4
                ),
            )
    assert lennard_jones_service._write_arrays is original_write
    checkpoint = load(recorded[-1], configuration, tmp_path)
    assert checkpoint.completed_steps == configuration.num_steps
    assert checkpoint.next_sample == len(configuration.sampling_steps())
    assert checkpoint.model_metadata["engine_state"]["analysis_pair_evaluations"] == 0
    forbid_initialization(monkeypatch)

    def forbidden_advance(*args, **kwargs):
        raise AssertionError("A terminal checkpoint needs no further physical transition")

    monkeypatch.setattr(LennardJonesSimulator, "advance", forbidden_advance)
    restored = run_simulation(
        configuration,
        task_id="scientific-run",
        attempt_id="second",
        artifacts_dir=tmp_path,
        checkpoint_control=control(
            configuration, tmp_path, "second", loaded=checkpoint, interval=0
        ),
    )
    assert_science_identical(tmp_path, baseline, restored)
    assert restored["diagnostics"]["analysis_pair_evaluations"] == (
        configuration.num_particles * (configuration.num_particles - 1) // 2
    )


@pytest.mark.parametrize("model", ["ou", "langevin", "lj-nvt-verlet"])
def test_cancellation_after_completed_checkpoint_keeps_resumable_boundary(tmp_path, model):
    configuration = parameters(model)
    baseline = run_simulation(
        configuration, task_id="baseline", attempt_id="uninterrupted", artifacts_dir=tmp_path
    )
    cancelled = Event()
    recorded = []

    def cancel_after_boundary(descriptor):
        recorded.append(descriptor)
        if descriptor["completed_steps"] == 7:
            cancelled.set()

    with pytest.raises(SimulationCancelled):
        run_simulation(
            configuration,
            task_id="scientific-run",
            attempt_id="first",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
            checkpoint_control=control(
                configuration, tmp_path, "first", register=cancel_after_boundary
            ),
        )
    assert recorded[-1]["completed_steps"] == 7
    assert not (tmp_path / "scientific-run" / "first" / "envelope.json").exists()
    checkpoint = load(recorded[-1], configuration, tmp_path)
    restored = run_simulation(
        configuration,
        task_id="scientific-run",
        attempt_id="second",
        artifacts_dir=tmp_path,
        checkpoint_control=control(
            configuration, tmp_path, "second", loaded=checkpoint, interval=0
        ),
    )
    assert_science_identical(tmp_path, baseline, restored)


@pytest.mark.parametrize("model", ["ou", "langevin", "lj-nvt-verlet"])
def test_cancelled_entry_never_initializes_or_publishes_checkpoint(tmp_path, monkeypatch, model):
    configuration = parameters(model)
    recorded = []
    cancelled = Event()
    cancelled.set()
    forbid_initialization(monkeypatch)
    with pytest.raises(SimulationCancelled):
        run_simulation(
            configuration,
            task_id="scientific-run",
            attempt_id="first",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
            checkpoint_control=control(configuration, tmp_path, "first", register=recorded.append),
        )
    assert recorded == []
    assert not list(tmp_path.rglob("checkpoint.json"))
