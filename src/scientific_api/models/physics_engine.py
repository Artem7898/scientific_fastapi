import numpy as np


class IdealGasSimulator:
    """
    Simplified 2D ideal gas simulator for demonstration.
    In production, replace with Numba/JAX/C++ bindings.
    """

    def __init__(self, num_particles: int, initial_temp: float, dt: float):
        self.num_particles = num_particles
        self.dt = dt
        # Maxwell-Boltzmann distribution initialization
        self.velocities = np.random.normal(
            loc=0.0,
            scale=np.sqrt(initial_temp),
            size=(num_particles, 2)
        )

    def step(self) -> tuple[float, float]:
        """
        Performs one integration step.
        Returns:
            tuple[float, float]: Effective temperature and total kinetic energy.
        """
        # In a real system: apply forces, boundary conditions, etc.
        # Here we just add small noise to simulate stochastic dynamics
        self.velocities += np.random.normal(0, self.dt, size=self.velocities.shape)

        # Kinetic energy: E_k = 0.5 * m * v^2 (assuming m=1)
        kinetic_energy = 0.5 * np.sum(self.velocities ** 2)

        # Effective temperature: T = 2 * E_k / (N * k_B * dof), assuming k_B=1, dof=2
        effective_temp = kinetic_energy / self.num_particles

        return effective_temp, kinetic_energy