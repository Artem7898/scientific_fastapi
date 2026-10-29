"""Validated inputs for independently forced, free Langevin particles in 2D."""

import math
from dataclasses import dataclass
from typing import Literal

from .simulation import MAX_PARTICLE_STEPS, ComputeBackend

LANGEVIN_MODEL_VERSION = "langevin-free-v1"
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
MAX_TRAJECTORY_FRAMES = 1001
Boundary = Literal["unbounded", "periodic"]


def _number(value: object, name: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")


def _pair(
    value: object, name: str, maximum: float, *, positive: bool = False
) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite numbers")
    minimum = 1e-6 if positive else -maximum
    for component in value:
        _number(component, name, minimum, maximum)
    return float(value[0]), float(value[1])


@dataclass(frozen=True, slots=True)
class LangevinParameters:
    """Independent 2D particles, with reduced k_B=1 and lab-frame friction.

    ``friction`` is the velocity relaxation rate, not the mass-weighted drag
    coefficient. All particles start at the same supplied position and their
    independent Gaussian velocities have the specified mean and temperature.
    Periodic coordinates are only a presentation of unwrapped trajectories;
    they do not produce walls or sedimentation equilibrium.
    """

    initial_temperature: float
    bath_temperature: float
    num_particles: int
    mass: float = 1.0
    friction: float = 1.0
    time_step: float = 0.01
    num_steps: int = 1000
    seed: int = 0
    backend: ComputeBackend = "numpy"
    initial_mean_velocity: tuple[float, float] = (0.0, 0.0)
    initial_position: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)
    boundary: Boundary = "unbounded"
    box_size: tuple[float, float] | None = None
    sample_stride: int = 10
    diffusion_fit_start_step: int | None = None
    diffusion_fit_end_step: int | None = None

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("initial_temperature", 1e-6, 1e6),
            ("bath_temperature", 1e-6, 1e6),
            ("mass", 1e-3, 1e3),
            ("friction", 1e-3, 1e3),
            ("time_step", 1e-6, 10.0),
        ):
            _number(getattr(self, name), name, minimum, maximum)
        for name, minimum, maximum in (
            ("num_particles", 3, 1_000_000),
            ("num_steps", 1, 10_000),
            ("seed", 0, 2**64 - 1),
            ("sample_stride", 1, 10_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.num_particles * self.num_steps > MAX_PARTICLE_STEPS:
            raise ValueError(f"num_particles * num_steps must not exceed {MAX_PARTICLE_STEPS}")
        if not isinstance(self.backend, str) or self.backend not in {"numpy", "numba"}:
            raise ValueError("backend must be numpy or numba")
        if not isinstance(self.boundary, str) or self.boundary not in {"unbounded", "periodic"}:
            raise ValueError("boundary must be unbounded or periodic")
        for name, maximum in (
            ("initial_mean_velocity", 1e3),
            ("initial_position", 1e6),
            ("acceleration", 1e3),
        ):
            object.__setattr__(self, name, _pair(getattr(self, name), name, maximum))
        if self.boundary == "periodic":
            object.__setattr__(
                self, "box_size", _pair(self.box_size, "box_size", 1e6, positive=True)
            )
        elif self.box_size is not None:
            raise ValueError("box_size is only valid for periodic boundaries")
        start, end = self.diffusion_fit_start_step, self.diffusion_fit_end_step
        for name, value, minimum, maximum in (
            ("diffusion_fit_start_step", start, 0, self.num_steps - 1),
            ("diffusion_fit_end_step", end, 1, self.num_steps),
        ):
            if value is not None and (type(value) is not int or not minimum <= value <= maximum):
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}] or None")
        if start is not None and end is not None and start >= end:
            raise ValueError("diffusion fit start must precede its end")
        samples = self.sampling_steps()
        if len(samples) > MAX_TRAJECTORY_FRAMES:
            raise ValueError(f"sampled trajectory must not exceed {MAX_TRAJECTORY_FRAMES} frames")
        if len(samples) * self.num_particles * 2 * 8 > MAX_TRAJECTORY_BYTES:
            raise ValueError("sampled position trajectory must not exceed 64 MiB")

    def sampling_steps(self) -> tuple[int, ...]:
        """Record the initial state, stride samples, and the final state once."""
        samples = tuple(range(0, self.num_steps + 1, self.sample_stride))
        return samples if samples[-1] == self.num_steps else (*samples, self.num_steps)
