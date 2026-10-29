"""Validated parameters for a two-dimensional shifted-force LJ fluid."""

import math
from dataclasses import dataclass
from typing import Literal

from .simulation import ComputeBackend

LJ_MODEL_VERSION = "lennard-jones-2d-v1"
MAX_LJ_PAIR_EVALUATIONS = 100_000_000
MAX_LJ_PARTICLE_STEPS = 20_000_000
MAX_LJ_TRAJECTORY_BYTES = 64 * 1024 * 1024
MAX_LJ_TRAJECTORY_FRAMES = 1001


def _number(value: object, name: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class LennardJonesParameters:
    """Periodic 2D LJ particles with a smooth force cutoff and reduced k_B=1.

    Initialization is a regular lattice with Gaussian velocity directions,
    zero center-of-mass velocity and an exact initial kinetic temperature.
    That constrained initial state is not a canonical equilibrium sample.
    """

    num_particles: int = 64
    box_size: tuple[float, float] = (12.0, 12.0)
    initial_temperature: float = 1.0
    epsilon: float = 1.0
    sigma: float = 1.0
    mass: float = 1.0
    time_step: float = 0.002
    num_steps: int = 1000
    ensemble: Literal["nve", "nvt"] = "nve"
    bath_temperature: float | None = None
    friction: float | None = None
    cutoff_radius: float = 2.5
    neighbor_skin: float = 0.3
    interaction_backend: Literal["direct", "verlet"] = "verlet"
    backend: ComputeBackend = "numpy"
    seed: int = 0
    sample_stride: int = 10
    equilibration_steps: int = 0
    rdf_bins: int = 64

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("epsilon", 1e-3, 1e3),
            ("sigma", 0.1, 10.0),
            ("mass", 0.1, 10.0),
        ):
            _number(getattr(self, name), name, lower, upper)
        _number(self.initial_temperature, "initial_temperature", 1e-4, 100 * self.epsilon)
        for name, lower, upper in (
            ("num_particles", 3, 4096),
            ("num_steps", 1, 10_000),
            ("seed", 0, 2**64 - 1),
            ("sample_stride", 1, 10_000),
            ("equilibration_steps", 0, self.num_steps - 1),
            ("rdf_bins", 8, 256),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not isinstance(self.backend, str) or self.backend not in {"numpy", "numba"}:
            raise ValueError("backend must be numpy or numba")
        if not isinstance(self.ensemble, str) or self.ensemble not in {"nve", "nvt"}:
            raise ValueError("ensemble must be nve or nvt")
        if not isinstance(self.interaction_backend, str) or self.interaction_backend not in {
            "direct",
            "verlet",
        }:
            raise ValueError("interaction_backend must be direct or verlet")
        if self.ensemble == "nve":
            if self.bath_temperature is not None or self.friction is not None:
                raise ValueError("NVE must not specify bath_temperature or friction")
        else:
            _number(self.bath_temperature, "bath_temperature", 1e-4, 100 * self.epsilon)
            if self.friction is None:
                object.__setattr__(self, "friction", 1.0)
            _number(self.friction, "friction", 1e-3, 1e3)
        if not isinstance(self.box_size, (tuple, list)) or len(self.box_size) != 2:
            raise ValueError("box_size must contain two finite lengths")
        for length in self.box_size:
            _number(length, "box_size", 1e-6 * self.sigma, 1e4 * self.sigma)
        object.__setattr__(self, "box_size", tuple(float(value) for value in self.box_size))
        _number(self.cutoff_radius, "cutoff_radius", 2 * self.sigma, 5 * self.sigma)
        _number(self.neighbor_skin, "neighbor_skin", 0.05 * self.sigma, self.sigma)
        if self.cutoff_radius + self.neighbor_skin >= min(self.box_size) / 2:
            raise ValueError(
                "cutoff_radius + neighbor_skin must be less than half the shortest box"
            )
        time_scale = self.sigma * math.sqrt(self.mass / self.epsilon)
        _number(self.time_step, "time_step", 1e-6 * time_scale, 0.01 * time_scale)
        columns, rows = self.lattice_shape()
        if min(self.box_size[0] / columns, self.box_size[1] / rows) < 0.95 * self.sigma:
            raise ValueError("Initial lattice spacing must be at least 0.95 sigma")
        if self.num_particles * self.num_steps > MAX_LJ_PARTICLE_STEPS:
            raise ValueError("LJ num_particles * num_steps must not exceed 20000000")
        # All force calls plus one terminal RDF pass, not only integrator steps.
        if self.interaction_backend == "direct" and self.direct_pair_evaluations() > (
            MAX_LJ_PAIR_EVALUATIONS
        ):
            raise ValueError("Direct LJ force and RDF work must not exceed 100000000 pair checks")
        samples = self.sampling_steps()
        if len(samples) > MAX_LJ_TRAJECTORY_FRAMES:
            raise ValueError("LJ trajectory must not exceed 1001 sampled frames")
        if len(samples) * self.num_particles * 2 * 8 > MAX_LJ_TRAJECTORY_BYTES:
            raise ValueError("LJ sampled position trajectory must not exceed 64 MiB")

    def lattice_shape(self) -> tuple[int, int]:
        columns = math.ceil(math.sqrt(self.num_particles * self.box_size[0] / self.box_size[1]))
        return columns, math.ceil(self.num_particles / columns)

    def sampling_steps(self) -> tuple[int, ...]:
        samples = tuple(range(0, self.num_steps + 1, self.sample_stride))
        return samples if samples[-1] == self.num_steps else (*samples, self.num_steps)

    def direct_pair_evaluations(self) -> int:
        return self.num_particles * (self.num_particles - 1) // 2 * (self.num_steps + 2)
