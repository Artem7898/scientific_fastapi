"""Run with an isolated Python and a core-only installed wheel outside checkout."""

import importlib.metadata
import importlib.util
import math
import os
import sys


def main() -> None:
    optional_packages = ("numba", "llvmlite", "jax", "jaxlib")
    assert all(importlib.util.find_spec(name) is None for name in optional_packages)

    import scientific_api
    from scientific_api.domain.simulation import SimulationParameters
    from scientific_api.main import app
    from scientific_api.models.computation import (
        BackendUnavailable,
        create_simulator,
        require_backend,
    )

    expected = os.environ["EXPECTED_PACKAGE_VERSION"]
    assert scientific_api.__version__ == expected
    assert importlib.metadata.version("scientific-api") == expected
    assert app is not None

    require_backend("numpy")
    simulator = create_simulator(SimulationParameters(initial_temperature=1.0, num_particles=4))
    simulator.advance()
    assert all(math.isfinite(value) for value in simulator.observables())
    assert simulator.execution_metadata()["backend"] == "numpy"

    for operation in (
        lambda: require_backend("numba"),
        lambda: create_simulator(
            SimulationParameters(initial_temperature=1.0, num_particles=4, backend="numba")
        ),
    ):
        try:
            operation()
        except BackendUnavailable as error:
            assert "scientific-api[numba]" in str(error)
        else:
            raise AssertionError("Unavailable Numba backend silently executed or fell back")

    assert not any(name.split(".", 1)[0] in optional_packages for name in sys.modules), (
        "The default runtime imported an optional compiled backend"
    )
    assert "scientific_api.models.numba_engine" not in sys.modules
    print("Core-only wheel: NumPy runs; optional JIT backends stay absent and fail explicitly")


if __name__ == "__main__":
    main()
