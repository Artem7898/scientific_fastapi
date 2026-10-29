"""Exact joint position/velocity transitions for free forced Langevin motion.

The position and velocity innovations in a step are correlated. Integrating
positions with ``x += dt*v`` would not have the transition law implemented here.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.langevin import LangevinParameters
from .computation import BackendUnavailable, require_backend


@dataclass(frozen=True, slots=True)
class TransitionCoefficients:
    decay: float
    velocity_response: float
    acceleration_response: float
    velocity_noise_variance: float
    position_noise_variance: float
    noise_covariance: float
    velocity_noise_scale: float
    position_velocity_noise_scale: float
    position_independent_noise_scale: float


def _position_variance_factor(q: float) -> float:
    """Evaluate 2q-3+4exp(-q)-exp(-2q) without small-q cancellation."""
    if q < 0.1:
        return math.fsum(
            ((-1.0) ** order) * (4.0 - 2.0**order) * q**order / math.factorial(order)
            for order in range(3, 22)
        )
    decay = math.exp(-q)
    return 2 * q - 3 + 4 * decay - decay * decay


def _acceleration_factor(q: float) -> float:
    """Evaluate (q-1+exp(-q))/q², whose limit is one half."""
    if q < 0.1:
        return math.fsum(
            (-1.0) ** order * q ** (order - 2) / math.factorial(order) for order in range(2, 22)
        )
    return (q + math.expm1(-q)) / (q * q)


def exact_transition_coefficients(
    time_step: float, friction: float, thermal_velocity_variance: float
) -> TransitionCoefficients:
    """Return exact Gaussian-transition coefficients for nonnegative time.

    ``thermal_velocity_variance`` is k_B*T_bath/m, and ``friction`` is the
    velocity relaxation rate. Time zero is supported for analytic references.
    """
    if (
        not math.isfinite(time_step)
        or time_step < 0
        or not math.isfinite(friction)
        or friction <= 0
        or not math.isfinite(thermal_velocity_variance)
        or thermal_velocity_variance < 0
    ):
        raise ValueError(
            "Transition requires finite nonnegative time/variance and positive friction"
        )
    q = friction * time_step
    if not math.isfinite(q):
        raise ValueError("The dimensionless time must be finite")
    if q == 0:
        return TransitionCoefficients(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    decay = math.exp(-q)
    one_minus_decay = -math.expm1(-q)
    response = one_minus_decay / friction
    acceleration = time_step * time_step * _acceleration_factor(q)
    velocity_variance = thermal_velocity_variance * -math.expm1(-2 * q)
    position_variance = (
        thermal_velocity_variance / (friction * friction) * _position_variance_factor(q)
    )
    covariance = thermal_velocity_variance / friction * one_minus_decay * one_minus_decay
    velocity_scale = math.sqrt(velocity_variance)
    shared_scale = covariance / velocity_scale if velocity_scale else 0.0
    independent_variance = position_variance - shared_scale * shared_scale
    if independent_variance < -1e-12 * position_variance:
        raise FloatingPointError("Joint transition covariance is not positive semidefinite")
    return TransitionCoefficients(
        decay=decay,
        velocity_response=response,
        acceleration_response=acceleration,
        velocity_noise_variance=velocity_variance,
        position_noise_variance=position_variance,
        noise_covariance=covariance,
        velocity_noise_scale=velocity_scale,
        position_velocity_noise_scale=shared_scale,
        position_independent_noise_scale=math.sqrt(max(independent_variance, 0.0)),
    )


def langevin_moments(time: float, parameters: LangevinParameters) -> dict[str, Any]:
    """Analytic marginal means and per-coordinate covariances from time zero."""
    coefficient = exact_transition_coefficients(
        time, parameters.friction, parameters.bath_temperature / parameters.mass
    )
    initial_variance = parameters.initial_temperature / parameters.mass
    mean_displacement = [
        coefficient.velocity_response * velocity + coefficient.acceleration_response * acceleration
        for velocity, acceleration in zip(
            parameters.initial_mean_velocity, parameters.acceleration, strict=True
        )
    ]
    mean_velocity = [
        coefficient.decay * velocity + coefficient.velocity_response * acceleration
        for velocity, acceleration in zip(
            parameters.initial_mean_velocity, parameters.acceleration, strict=True
        )
    ]
    return {
        "mean_displacement": mean_displacement,
        "mean_velocity": mean_velocity,
        "position_variance_per_component": (
            initial_variance * coefficient.velocity_response**2
            + coefficient.position_noise_variance
        ),
        "velocity_variance_per_component": (
            initial_variance * coefficient.decay**2 + coefficient.velocity_noise_variance
        ),
        "position_velocity_covariance": (
            initial_variance * coefficient.velocity_response * coefficient.decay
            + coefficient.noise_covariance
        ),
        "theoretical_diffusion": parameters.bath_temperature
        / (parameters.mass * parameters.friction),
    }


def update_state(
    unwrapped_positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    shocks: NDArray[np.float64],
    acceleration: NDArray[np.float64],
    coefficients: TransitionCoefficients,
) -> None:
    """Advance one supplied-noise joint transition, using the old velocity."""
    with np.errstate(over="raise", invalid="raise"):
        unwrapped_positions += (
            coefficients.velocity_response * velocities
            + coefficients.acceleration_response * acceleration
            + coefficients.position_velocity_noise_scale * shocks[:, :, 0]
            + coefficients.position_independent_noise_scale * shocks[:, :, 1]
        )
        velocities *= coefficients.decay
        velocities += (
            coefficients.velocity_response * acceleration
            + coefficients.velocity_noise_scale * shocks[:, :, 0]
        )


class LangevinSimulator:
    """A local, seeded scientific state without transport or persistence."""

    def __init__(self, parameters: LangevinParameters) -> None:
        if not isinstance(parameters, LangevinParameters):
            raise TypeError("parameters must be LangevinParameters")
        self.parameters = parameters
        self.rng = np.random.Generator(np.random.PCG64(parameters.seed))
        self.velocities: NDArray[np.float64] = self.rng.normal(
            loc=np.asarray(parameters.initial_mean_velocity, dtype=np.float64),
            scale=math.sqrt(parameters.initial_temperature / parameters.mass),
            size=(parameters.num_particles, 2),
        )
        self.unwrapped_positions: NDArray[np.float64] = np.broadcast_to(
            np.asarray(parameters.initial_position, dtype=np.float64), (parameters.num_particles, 2)
        ).copy()
        self._acceleration = np.asarray(parameters.acceleration, dtype=np.float64)
        self._coefficients = exact_transition_coefficients(
            parameters.time_step, parameters.friction, parameters.bath_temperature / parameters.mass
        )

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return presentation coordinates; dynamics always remain unwrapped."""
        if self.parameters.boundary == "periodic":
            assert self.parameters.box_size is not None  # Enforced by LangevinParameters.
            box_size = np.asarray(self.parameters.box_size, dtype=np.float64)
            wrapped = np.remainder(self.unwrapped_positions, box_size)
            # A tiny negative coordinate can round to exactly the box length
            # after remainder. Its periodic representative is the origin.
            wrapped[(wrapped >= box_size) | (wrapped < 0.0)] = 0.0
            return wrapped
        return self.unwrapped_positions.copy()

    def advance(self) -> None:
        # Each coordinate consumes one velocity innovation and an independent
        # conditional-position innovation. No full-trajectory noise is kept.
        shocks = self.rng.standard_normal((self.parameters.num_particles, 2, 2))
        update_state(
            self.unwrapped_positions,
            self.velocities,
            shocks,
            self._acceleration,
            self._coefficients,
        )

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "backend": "numpy",
            "backend_version": np.__version__,
            "kernel_version": "langevin-joint-numpy-v1",
            "device": "cpu",
            "dtype": "float64",
            "fastmath": False,
            "parallel": False,
            "rng_policy": "numpy-pcg64-langevin-joint-v1",
            "innovation_shape_per_step": [self.parameters.num_particles, 2, 2],
            "innovation_order": "particle,dimension,(velocity,conditional-position)",
            "observables_schedule": "sampled positions and terminal velocities",
        }


def create_langevin_simulator(parameters: LangevinParameters) -> LangevinSimulator:
    """Allowlisted optional backend selection, with no fallback on failures."""
    if not isinstance(parameters, LangevinParameters):
        raise TypeError("parameters must be LangevinParameters")
    require_backend(parameters.backend)
    if parameters.backend == "numba":
        try:
            from .langevin_numba import NumbaLangevinSimulator
        except ImportError as error:
            raise BackendUnavailable("Numba Langevin backend could not be imported") from error
        return NumbaLangevinSimulator(parameters)
    return LangevinSimulator(parameters)
