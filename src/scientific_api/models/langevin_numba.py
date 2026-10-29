"""Optional serial float64 Numba implementation of the exact joint kernel."""

from typing import Any

import llvmlite
import numba
from llvmlite import binding as llvm

from .langevin_engine import LangevinSimulator


@numba.njit(nogil=True, fastmath=False, cache=False)
def update_state(
    positions,
    velocities,
    shocks,
    acceleration,
    decay,
    response,
    acceleration_response,
    velocity_scale,
    shared_scale,
    independent_scale,
):
    for particle in range(velocities.shape[0]):
        for dimension in range(2):
            old_velocity = velocities[particle, dimension]
            velocity_noise = shocks[particle, dimension, 0]
            position_noise = shocks[particle, dimension, 1]
            positions[particle, dimension] += (
                response * old_velocity
                + acceleration_response * acceleration[dimension]
                + shared_scale * velocity_noise
                + independent_scale * position_noise
            )
            velocities[particle, dimension] = decay * old_velocity + (
                response * acceleration[dimension] + velocity_scale * velocity_noise
            )


class NumbaLangevinSimulator(LangevinSimulator):
    def advance(self) -> None:
        coefficient = self._coefficients
        update_state(
            self.unwrapped_positions,
            self.velocities,
            self.rng.standard_normal((self.parameters.num_particles, 2, 2)),
            self._acceleration,
            coefficient.decay,
            coefficient.velocity_response,
            coefficient.acceleration_response,
            coefficient.velocity_noise_scale,
            coefficient.position_velocity_noise_scale,
            coefficient.position_independent_noise_scale,
        )

    def execution_metadata(self) -> dict[str, Any]:
        return {
            **super().execution_metadata(),
            "backend": "numba",
            "backend_version": numba.__version__,
            "kernel_version": "langevin-joint-numba-v1",
            "nogil": True,
            "cache": False,
            "llvmlite_version": llvmlite.__version__,
            "llvm_version": ".".join(str(part) for part in llvm.llvm_version_info),
        }
