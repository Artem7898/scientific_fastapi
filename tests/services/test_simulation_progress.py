"""Scientific progress is observational and precedes committed completion."""

from threading import Event

import numpy as np
import pytest

from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.services.simulation_service import run_simulation


def test_progress_preserves_seeded_values_and_reports_artifact_phase(tmp_path):
    parameters = SimulationParameters(1.0, 128, num_steps=20, seed=42)
    baseline = run_simulation(parameters, task_id="baseline", artifacts_dir=tmp_path)
    updates = []
    observed = run_simulation(
        parameters,
        task_id="observed",
        artifacts_dir=tmp_path,
        progress_callback=lambda done, total, phase: updates.append((done, total, phase)),
    )
    assert updates[0] == (0, 20, "computing")
    assert updates[-2:] == [(20, 20, "computing"), (20, 20, "saving_artifacts")]
    assert all(phase != "completed" for _, _, phase in updates)
    for metric in ("final_temperature", "total_energy", "temperature_standard_error"):
        assert observed[metric] == baseline[metric]
    with (
        np.load(tmp_path / baseline["artifacts"]["final_state"]) as first,
        np.load(tmp_path / observed["artifacts"]["final_state"]) as second,
    ):
        np.testing.assert_array_equal(first["velocities"], second["velocities"])


def test_cancel_during_progress_does_not_publish_state(tmp_path):
    cancelled = Event()

    def cancel(done, total, phase):
        if phase == "saving_artifacts":
            cancelled.set()

    with pytest.raises(SimulationCancelled):
        run_simulation(
            SimulationParameters(1.0, 32, num_steps=2),
            task_id="cancelled-progress",
            artifacts_dir=tmp_path,
            cancel_event=cancelled,
            progress_callback=cancel,
        )
    assert not (tmp_path / "cancelled-progress" / "final_state.npz").exists()
    assert not (tmp_path / "cancelled-progress" / "manifest.json").exists()
