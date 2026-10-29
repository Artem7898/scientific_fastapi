"""Production execution preserves the reference trajectory and control ports."""

import json
from threading import Event

import numpy as np
import pytest

from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.models.computation import create_simulator
from scientific_api.models.physics_engine import IdealGasSimulator
from scientific_api.services.simulation_service import run_simulation


def test_terminal_observables_preserve_original_numpy_step_results(tmp_path, monkeypatch):
    parameters = SimulationParameters(3.0, 256, time_step=0.2, num_steps=20, seed=2026)
    reference = IdealGasSimulator(256, 3.0, 0.2, seed=2026)
    for _ in range(parameters.num_steps):
        expected_temperature, expected_energy = reference.step()

    original = IdealGasSimulator.observables
    reductions = []

    def count_reduction(simulator):
        reductions.append(True)
        return original(simulator)

    monkeypatch.setattr(IdealGasSimulator, "observables", count_reduction)
    result = run_simulation(parameters, task_id="terminal-only", artifacts_dir=tmp_path)
    assert result["final_temperature"] == expected_temperature
    assert result["total_energy"] == expected_energy
    assert len(reductions) == 1
    with np.load(tmp_path / result["artifacts"]["final_state"], allow_pickle=False) as state:
        np.testing.assert_array_equal(state["velocities"], reference.velocities)
    manifest = json.loads((tmp_path / result["artifacts"]["manifest"]).read_text())
    assert manifest["execution"]["backend"] == "numpy"
    assert manifest["execution"]["observables_schedule"] == "terminal"


def test_numba_service_cancellation_remains_at_step_boundaries(tmp_path, monkeypatch):
    pytest.importorskip("numba")
    parameters = SimulationParameters(3.0, 64, num_steps=20, backend="numba")
    simulator = create_simulator(parameters)
    cancellation = Event()
    original = type(simulator).advance
    steps = []

    def cancel_after_transition(state):
        original(state)
        steps.append(True)
        cancellation.set()

    monkeypatch.setattr(type(simulator), "advance", cancel_after_transition)
    updates = []
    with pytest.raises(SimulationCancelled):
        run_simulation(
            parameters,
            task_id="cancelled-numba",
            artifacts_dir=tmp_path,
            cancel_event=cancellation,
            progress_callback=lambda done, total, phase: updates.append((done, total, phase)),
        )
    assert len(steps) == 1
    assert updates[0] == (0, 20, "computing")
    assert all(done <= 1 and phase == "computing" for done, _, phase in updates)
    assert not list((tmp_path / "cancelled-numba").glob("*.npz"))
    assert not (tmp_path / "cancelled-numba" / "manifest.json").exists()
