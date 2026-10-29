"""Reference equilibrium thermostat, with an exact stochastic transition."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.simulation import SimulationParameters


def update_velocities(
    velocities: NDArray[np.float64],
    shocks: NDArray[np.float64],
    decay: float,
    noise_scale: float,
) -> None:
    """Reference in-place transition on supplied noise, without RNG or reduction."""
    with np.errstate(over="raise", invalid="raise"):
        velocities *= decay
        velocities += noise_scale * shocks


class IdealGasSimulator:
    """Independent 2D velocities at target temperature in reduced units.

    This demonstrates the Ornstein-Uhlenbeck thermostat
    ``dv = -v dt + sqrt(2 T) dW`` with unit relaxation rate, mass and Boltzmann
    constant. The exact transition preserves Maxwellian equilibrium for any
    positive time step. There are no positions, collisions or interparticle
    forces; this is not a molecular dynamics solver.
    """

    def __init__(self, num_particles: int, initial_temp: float, dt: float, seed: int = 0) -> None:
        parameters = SimulationParameters(
            initial_temperature=initial_temp,
            num_particles=num_particles,
            time_step=dt,
            num_steps=1,
            seed=seed,
        )
        self.num_particles = parameters.num_particles
        self.dt = parameters.time_step
        self.target_temperature = parameters.initial_temperature
        self.rng = np.random.Generator(np.random.PCG64(parameters.seed))
        self.velocities: NDArray[np.float64] = self.rng.normal(
            loc=0.0, scale=math.sqrt(self.target_temperature), size=(num_particles, 2)
        )
        self._decay = math.exp(-self.dt)
        # expm1 keeps 1 - exp(-2 dt) accurate for small time steps.
        self._noise_scale = math.sqrt(self.target_temperature * -math.expm1(-2.0 * self.dt))

    def observables(self) -> tuple[float, float]:
        """Return temperature and energy, rejecting nonfinite computations."""
        with np.errstate(over="raise", invalid="raise"):
            kinetic_energy = float(0.5 * np.einsum("ij,ij->", self.velocities, self.velocities))
        effective_temperature = kinetic_energy / self.num_particles
        if not math.isfinite(kinetic_energy) or not math.isfinite(effective_temperature):
            raise FloatingPointError("Simulation produced nonfinite observables")
        return effective_temperature, kinetic_energy

    def step(self) -> tuple[float, float]:
        """Advance one exact OU transition and return terminal observables."""
        self.advance()
        return self.observables()

    def advance(self) -> None:
        """Advance once, retaining only current state and a single noise array.

        The execution service samples observables at the terminal state. The
        public ``step`` method keeps its previous per-step reduction contract.
        """
        update_velocities(
            self.velocities,
            self.rng.standard_normal(self.velocities.shape),
            self._decay,
            self._noise_scale,
        )

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the implementation independently of the physical model."""
        return {
            "backend": "numpy",
            "backend_version": np.__version__,
            "kernel_version": "ou-numpy-v1",
            "device": "cpu",
            "dtype": "float64",
            "fastmath": False,
            "parallel": False,
            "rng_policy": "numpy-pcg64-per-step-v1",
            "observables_schedule": "terminal",
        }
