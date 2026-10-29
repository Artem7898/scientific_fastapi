"""Optional CPU transition kernel; imported only for an explicit Numba run.

Randomness stays in NumPy's per-simulator PCG64 generator. Numba only fuses
the arithmetic on one noise array, so trajectories can be compared using
identical innovations. There is no parallel thread pool or relaxed floating
point arithmetic. Compilation is per process and intentionally uncached, to
support installed wheels on read-only filesystems without cache side effects.
"""

from typing import Any

import llvmlite
import numba
from llvmlite import binding as llvm

from .physics_engine import IdealGasSimulator


@numba.njit(nogil=True, fastmath=False, cache=False)
def update_velocities(velocities, shocks, decay, noise_scale):
    """Update C-contiguous float64 state in place without an O(steps*N) buffer."""
    for particle in range(velocities.shape[0]):
        for dimension in range(velocities.shape[1]):
            velocities[particle, dimension] = (
                decay * velocities[particle, dimension] + noise_scale * shocks[particle, dimension]
            )


class NumbaIdealGasSimulator(IdealGasSimulator):
    """The reference OU model with a lazily compiled, serial CPU transition."""

    def advance(self) -> None:
        update_velocities(
            self.velocities,
            self.rng.standard_normal(self.velocities.shape),
            self._decay,
            self._noise_scale,
        )

    def execution_metadata(self) -> dict[str, Any]:
        return {
            **super().execution_metadata(),
            "backend": "numba",
            "backend_version": numba.__version__,
            "kernel_version": "ou-numba-v1",
            "nogil": True,
            "cache": False,
            "llvmlite_version": llvmlite.__version__,
            "llvm_version": ".".join(str(part) for part in llvm.llvm_version_info),
        }
