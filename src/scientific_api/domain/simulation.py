"""Validated inputs and errors shared by scientific workflows."""

import math
from dataclasses import dataclass
from typing import Literal

MODEL_VERSION = "ideal-gas-ou-v1"
MAX_PARTICLE_STEPS = 100_000_000
ComputeBackend = Literal["numpy", "numba"]


class SimulationCancelled(RuntimeError):
    """The caller requested cancellation at a simulation step boundary."""


@dataclass(frozen=True, slots=True)
class SimulationParameters:
    """An equilibrium 2D ideal gas thermostat in reduced units m = k_B = 1.

    The thermostat relaxation rate is one per reduced time unit. The limits
    bound memory and CPU work for the reference implementation. A fixed seed
    reproduces a trajectory in the same NumPy/platform/model environment.
    """

    initial_temperature: float
    num_particles: int
    time_step: float = 0.001
    num_steps: int = 100
    seed: int = 0
    backend: ComputeBackend = "numpy"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {"numpy", "numba"}:
            raise ValueError("backend must be numpy or numba")
        for name, maximum in (("initial_temperature", 1e6), ("time_step", 1000.0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not 0 < value <= maximum
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and in (0, {maximum}]")
        for name, minimum, maximum in (
            ("num_particles", 1, 1_000_000),
            ("num_steps", 1, 10_000),
            ("seed", 0, 2**64 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.num_particles * self.num_steps > MAX_PARTICLE_STEPS:
            raise ValueError(f"num_particles * num_steps must not exceed {MAX_PARTICLE_STEPS}")
