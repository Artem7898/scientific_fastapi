"""Run with an isolated Python and a core-only installed wheel outside checkout."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    optional_packages = ("numba", "llvmlite", "jax", "jaxlib", "matplotlib")
    assert all(importlib.util.find_spec(name) is None for name in optional_packages)

    import numpy as np

    import scientific_api
    from scientific_api.domain.langevin import LangevinParameters
    from scientific_api.domain.simulation import SimulationParameters
    from scientific_api.main import app
    from scientific_api.models.computation import (
        BackendUnavailable,
        create_simulator,
        require_backend,
    )
    from scientific_api.models.langevin_engine import create_langevin_simulator
    from scientific_api.research.transport_analysis import analyze_transport, load_trajectory_state
    from scientific_api.services.simulation_service import run_simulation

    expected = os.environ["EXPECTED_PACKAGE_VERSION"]
    assert scientific_api.__version__ == expected
    assert importlib.metadata.version("scientific-api") == expected
    assert app is not None

    require_backend("numpy")
    simulator = create_simulator(SimulationParameters(initial_temperature=1.0, num_particles=4))
    simulator.advance()
    assert all(math.isfinite(value) for value in simulator.observables())
    assert simulator.execution_metadata()["backend"] == "numpy"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parameters = LangevinParameters(
            1.0, 1.0, 4, num_steps=4, time_step=0.1, sample_stride=2, seed=42
        )
        result = run_simulation(
            parameters, task_id="core_langevin", attempt_id="wheel", artifacts_dir=root
        )
        assert result["status"] == "completed"
        assert result["model_version"] == "langevin-free-v1"
        assert set(result["artifacts"]) == {"manifest", "final_state", "trajectory", "transport"}
        manifest = json.loads((root / result["artifacts"]["manifest"]).read_text())
        assert manifest["model"]["version"] == "langevin-free-v1"
        assert manifest["parameters"]["backend"] == manifest["execution"]["backend"] == "numpy"
        records = {entry["path"]: entry for entry in manifest["artifacts"]}
        for path, record in records.items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == record["sha256"]
        trajectory = load_trajectory_state(
            root / result["artifacts"]["trajectory"],
            expected_sha256=records[result["artifacts"]["trajectory"]]["sha256"],
            expected_particles=parameters.num_particles,
            expected_sample_steps=parameters.sampling_steps(),
            expected_time_step=parameters.time_step,
        )
        assert trajectory["positions"].shape == (3, 4, 2)
        assert trajectory["sample_steps"].dtype == np.int64
        assert trajectory["times"].dtype == np.float64
        with np.load(root / result["artifacts"]["final_state"], allow_pickle=False) as state:
            assert set(state.files) == {"positions", "unwrapped_positions", "velocities"}
            np.testing.assert_array_equal(trajectory["positions"][-1], state["unwrapped_positions"])
            np.testing.assert_array_equal(state["positions"], state["unwrapped_positions"])
        observed = analyze_transport(
            trajectory["positions"], trajectory["sample_steps"], parameters=parameters
        )
        saved = json.loads((root / result["artifacts"]["transport"]).read_text())
        assert observed == saved
        assert saved["coordinate_convention"] == "unwrapped"
        assert saved["msd"][-1] == result["final_msd"]
        assert saved["centered_msd"][-1] == result["final_centered_msd"]
        assert saved["diffusion"]["status"] == "unavailable"
        assert saved["diffusion"]["estimate"] is None

    for operation in (
        lambda: require_backend("numba"),
        lambda: create_simulator(
            SimulationParameters(initial_temperature=1.0, num_particles=4, backend="numba")
        ),
        lambda: create_langevin_simulator(LangevinParameters(1.0, 1.0, 4, backend="numba")),
    ):
        try:
            operation()
        except BackendUnavailable as error:
            assert "scientific-api[numba]" in str(error)
        else:
            raise AssertionError("Unavailable Numba backend silently executed or fell back")

    for unknown_backend in ("jax", "auto", "unknown"):
        try:
            require_backend(unknown_backend)
        except ValueError:
            pass
        else:
            raise AssertionError("An unknown backend was admitted")

    assert not any(name.split(".", 1)[0] in optional_packages for name in sys.modules), (
        "The default runtime imported an optional compiled backend"
    )
    assert "scientific_api.models.numba_engine" not in sys.modules
    assert "scientific_api.models.langevin_numba" not in sys.modules
    print(
        "Core-only wheel: OU/Langevin transport runs; optional compute/plot libraries stay absent"
    )


if __name__ == "__main__":
    main()
