"""Periodic 2D shifted-force Lennard–Jones dynamics with bounded pair work."""

import importlib.metadata
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.lennard_jones import MAX_LJ_PAIR_EVALUATIONS, LennardJonesParameters
from .computation import require_backend
from .lj_forces import ForceResult, PairBudgetExceeded, VerletList, compute_forces


class LennardJonesSimulator:
    """Velocity Verlet (NVE) or BAOAB (NVT), without transport dependencies."""

    def __init__(self, parameters: LennardJonesParameters) -> None:
        if not isinstance(parameters, LennardJonesParameters):
            raise TypeError("parameters must be LennardJonesParameters")
        require_backend(parameters.backend)
        self.parameters = parameters
        self.rng = np.random.Generator(np.random.PCG64(parameters.seed))
        columns, rows = parameters.lattice_shape()
        index = np.arange(parameters.num_particles)
        self.unwrapped_positions: NDArray[np.float64] = np.column_stack(
            (
                (index % columns + 0.5) * parameters.box_size[0] / columns,
                (index // columns + 0.5) * parameters.box_size[1] / rows,
            )
        )
        self.velocities: NDArray[np.float64] = self.rng.normal(
            0.0,
            math.sqrt(parameters.initial_temperature / parameters.mass),
            size=(parameters.num_particles, 2),
        )
        self.velocities -= np.mean(self.velocities, axis=0)
        norm = float(np.einsum("ij,ij->", self.velocities, self.velocities))
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Cannot initialize finite nonzero thermal velocities")
        self.velocities *= math.sqrt(
            2
            * (parameters.num_particles - 1)
            * parameters.initial_temperature
            / (parameters.mass * norm)
        )
        self._neighbors = (
            VerletList(parameters.box_size, parameters.cutoff_radius, parameters.neighbor_skin)
            if parameters.interaction_backend == "verlet"
            else None
        )
        self._pair_evaluations = 0
        self._analysis_pair_evaluations = 0
        self._steps = 0
        self._force_result = self._evaluate_forces()
        if parameters.ensemble == "nvt":
            assert parameters.friction is not None
            assert parameters.bath_temperature is not None
            self._decay = math.exp(-parameters.friction * parameters.time_step)
            self._noise_scale = math.sqrt(
                parameters.bath_temperature
                / parameters.mass
                * -math.expm1(-2 * parameters.friction * parameters.time_step)
            )

    @property
    def positions(self) -> NDArray[np.float64]:
        box = np.asarray(self.parameters.box_size, dtype=np.float64)
        wrapped = np.remainder(self.unwrapped_positions, box)
        # Floating-point remainder of a tiny negative number can round to L.
        wrapped[(wrapped >= box) | (wrapped < 0.0)] = 0.0
        return wrapped

    @property
    def forces(self) -> NDArray[np.float64]:
        """Current force state; mutating it is unsupported."""
        return self._force_result.forces

    def _used_pair_checks(self) -> int:
        return (
            self._pair_evaluations
            + self._analysis_pair_evaluations
            + (self._neighbors.candidate_checks if self._neighbors is not None else 0)
        )

    def _evaluate_forces(self) -> ForceResult:
        parameters = self.parameters
        remaining = MAX_LJ_PAIR_EVALUATIONS - self._used_pair_checks()
        pairs = None
        count = parameters.num_particles * (parameters.num_particles - 1) // 2
        if self._neighbors is not None:
            pairs = self._neighbors.pairs(self.unwrapped_positions, max_candidate_checks=remaining)
            count = len(pairs)
            remaining = MAX_LJ_PAIR_EVALUATIONS - self._used_pair_checks()
        if count > remaining:
            raise PairBudgetExceeded("LJ force and neighbor work exceeds the 100000000 pair budget")
        result = compute_forces(
            self.unwrapped_positions,
            parameters.box_size,
            parameters.epsilon,
            parameters.sigma,
            parameters.cutoff_radius,
            backend=parameters.backend,
            pairs=pairs,
        )
        self._pair_evaluations += result.pair_evaluations
        return result

    def reserve_analysis_pairs(self, count: int) -> None:
        """Reserve bounded analysis work before running a terminal RDF pass."""
        if type(count) is not int or count < 0:
            raise ValueError("Analysis pair count must be a nonnegative integer")
        if count > MAX_LJ_PAIR_EVALUATIONS - self._used_pair_checks():
            raise PairBudgetExceeded("LJ analysis exceeds the 100000000 total pair budget")
        self._analysis_pair_evaluations += count

    def advance(self) -> None:
        parameters = self.parameters
        dt = parameters.time_step
        half_kick = 0.5 * dt / parameters.mass
        with np.errstate(over="raise", invalid="raise"):
            self.velocities += half_kick * self.forces
            if parameters.ensemble == "nve":
                self.unwrapped_positions += dt * self.velocities
            else:
                self.unwrapped_positions += 0.5 * dt * self.velocities
                self.velocities *= self._decay
                self.velocities += self._noise_scale * self.rng.standard_normal(
                    self.velocities.shape
                )
                self.unwrapped_positions += 0.5 * dt * self.velocities
            self._force_result = self._evaluate_forces()
            self.velocities += half_kick * self.forces
        if (
            not np.isfinite(self.velocities).all()
            or not np.isfinite(self.unwrapped_positions).all()
        ):
            raise FloatingPointError("LJ integration produced nonfinite state")
        self._steps += 1

    def observables(self) -> dict[str, float]:
        parameters = self.parameters
        with np.errstate(over="raise", invalid="raise"):
            mean_velocity = np.mean(self.velocities, axis=0)
            fluctuations = self.velocities - mean_velocity
            kinetic = float(
                0.5 * parameters.mass * np.einsum("ij,ij->", self.velocities, self.velocities)
            )
            thermal_temperature = float(
                parameters.mass
                * np.einsum("ij,ij->", fluctuations, fluctuations)
                / (2 * (parameters.num_particles - 1))
            )
        potential = self._force_result.potential_energy
        virial = self._force_result.virial
        values = {
            "thermal_temperature": thermal_temperature,
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "total_energy": kinetic + potential,
            "pressure": (2 * kinetic + virial) / (2 * math.prod(parameters.box_size)),
            "virial": virial,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("LJ model produced nonfinite observables")
        return values

    def diagnostics(self) -> dict[str, Any]:
        return {
            "pair_evaluations": self._pair_evaluations,
            "neighbor_candidate_checks": (
                self._neighbors.candidate_checks if self._neighbors is not None else 0
            ),
            "analysis_pair_evaluations": self._analysis_pair_evaluations,
            "total_pair_checks": self._used_pair_checks(),
            "neighbor_rebuilds": (
                self._neighbors.rebuild_count if self._neighbors is not None else 0
            ),
            "interaction_backend": self.parameters.interaction_backend,
            "pair_budget": MAX_LJ_PAIR_EVALUATIONS,
        }

    def execution_metadata(self) -> dict[str, Any]:
        parameters = self.parameters
        metadata: dict[str, Any] = {
            "backend": parameters.backend,
            "backend_version": np.__version__,
            "kernel_version": "lj-shifted-force-numpy-v1",
            "interaction_backend": parameters.interaction_backend,
            "integrator": "velocity-verlet" if parameters.ensemble == "nve" else "baoab",
            "device": "cpu",
            "dtype": "float64",
            "fastmath": False,
            "parallel": False,
            "cutoff_policy": "shifted-force; potential and radial force vanish at cutoff",
            "rng_policy": "numpy-pcg64-lj-v1",
            "initialization": "rectangular lattice; Gaussian velocity directions, zero COM, fixed initial thermal temperature",
            "nvt_innovation_order": "particle,dimension; independent lab-frame OU thermostat",
        }
        if parameters.backend == "numba":
            # The optional module has already been loaded by compute_forces.
            from llvmlite import binding as llvm

            metadata.update(
                backend_version=importlib.metadata.version("numba"),
                kernel_version="lj-shifted-force-numba-v1",
                llvmlite_version=importlib.metadata.version("llvmlite"),
                llvm_version=".".join(str(value) for value in llvm.llvm_version_info),
                cache=False,
                nogil=True,
            )
        return metadata


def create_lj_simulator(parameters: LennardJonesParameters) -> LennardJonesSimulator:
    return LennardJonesSimulator(parameters)
