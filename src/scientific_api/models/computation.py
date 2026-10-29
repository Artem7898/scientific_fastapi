"""Allowlisted scientific backend construction with lightweight admission checks."""

from importlib.util import find_spec
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..domain.simulation import ComputeBackend, SimulationParameters
from .physics_engine import IdealGasSimulator


class BackendUnavailable(RuntimeError):
    """The requested implementation cannot execute in this installation."""


class Simulator(Protocol):
    """A synchronous scientific state, independent of transport and storage."""

    velocities: NDArray[np.float64]

    def advance(self) -> None: ...

    def observables(self) -> tuple[float, float]: ...

    def execution_metadata(self) -> dict[str, Any]: ...


def require_backend(backend: ComputeBackend) -> None:
    """Check availability without importing optional native code or compiling.

    API-only installations can check whether the requested optional package is
    installed. Workers still import and validate their own implementation; an
    import/compilation failure is never converted into a NumPy fallback.
    """
    if backend == "numpy":
        return
    if backend != "numba":
        raise ValueError("backend must be numpy or numba")
    if find_spec("numba") is None:
        raise BackendUnavailable("Numba backend requires scientific-api[numba]")


def create_simulator(parameters: SimulationParameters) -> Simulator:
    """Construct only explicitly supported implementations, without fallback."""
    if not isinstance(parameters, SimulationParameters):
        raise TypeError("parameters must be SimulationParameters")
    require_backend(parameters.backend)
    implementation: type[IdealGasSimulator] = IdealGasSimulator
    if parameters.backend == "numba":
        try:
            from .numba_engine import NumbaIdealGasSimulator
        except ImportError as exc:
            raise BackendUnavailable("Numba backend could not be imported") from exc
        implementation = NumbaIdealGasSimulator
    return implementation(
        num_particles=parameters.num_particles,
        initial_temp=parameters.initial_temperature,
        dt=parameters.time_step,
        seed=parameters.seed,
    )
