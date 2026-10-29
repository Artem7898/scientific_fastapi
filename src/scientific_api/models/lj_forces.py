"""Shifted-force Lennard–Jones reference forces and sparse periodic neighbours.

Pairs are counted once, with ``i < j``. The direct implementation retains only
one row of pair displacements; the Verlet implementation stores a bounded,
sorted neighbour list built using sparse cells. Neither creates an N-by-N
distance matrix. Computational backends share the same physical truncation.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .computation import BackendUnavailable, require_backend

MAX_PARTICLES = 4096
MAX_NEIGHBOR_PAIRS = 1_000_000


class ParticleOverlapError(FloatingPointError):
    """The unsoftened potential cannot safely advance a near-overlap."""


class NeighborCapacityError(ValueError):
    """The bounded neighbour-pair storage would be exceeded."""


class PairBudgetExceeded(RuntimeError):
    """The requested neighbour search exceeds its remaining work budget."""


@dataclass(frozen=True, slots=True)
class ForceResult:
    forces: NDArray[np.float64]
    potential_energy: float
    virial: float
    pair_evaluations: int


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _box(box_size: object) -> NDArray[np.float64]:
    try:
        values = np.asarray(box_size)
    except (ValueError, TypeError) as error:
        raise ValueError("box_size must contain two finite positive lengths") from error
    if (
        values.shape != (2,)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or not (values > 0).all()
    ):
        raise ValueError("box_size must contain two finite positive lengths")
    return values.astype(np.float64)


def _positions(positions: NDArray[np.float64]) -> None:
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.dtype(np.float64)
        or positions.ndim != 2
        or positions.shape[1] != 2
        or not 2 <= len(positions) <= MAX_PARTICLES
        or not np.isfinite(positions).all()
    ):
        raise ValueError("positions must be finite native float64 with shape (N,2), N=2..4096")


def _wrapped(positions: NDArray[np.float64], box: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.remainder(positions, box)
    # A tiny negative coordinate can round to L after remainder in float64.
    return np.where((result >= box) | (result < 0), 0.0, result)


def lennard_jones_pair(
    distance: float, epsilon: float = 1.0, sigma: float = 1.0
) -> tuple[float, float]:
    """Return bare potential and outward radial force, without truncation."""
    radius = _positive(distance, "distance")
    epsilon, sigma = _positive(epsilon, "epsilon"), _positive(sigma, "sigma")
    try:
        sixth = (sigma / radius) ** 6
        potential = 4.0 * epsilon * (sixth * sixth - sixth)
        force = 24.0 * epsilon / radius * (2.0 * sixth * sixth - sixth)
    except OverflowError as error:
        raise FloatingPointError("Lennard–Jones pair is not finite") from error
    if not math.isfinite(potential) or not math.isfinite(force):
        raise FloatingPointError("Lennard–Jones pair is not finite")
    return potential, force


def shifted_force_pair(
    distance: float, epsilon: float = 1.0, sigma: float = 1.0, cutoff_radius: float = 2.5
) -> tuple[float, float]:
    """Return U_sf and radial F_sf, both zero at and beyond the cutoff.

    U_sf(r) = U(r) - U(rc) + (r - rc) F(rc), so F_sf = F(r) - F(rc).
    Consequently the shifted potential's minimum differs from the bare one.
    """
    radius = _positive(distance, "distance")
    cutoff = _positive(cutoff_radius, "cutoff_radius")
    epsilon, sigma = _positive(epsilon, "epsilon"), _positive(sigma, "sigma")
    if radius >= cutoff:
        return 0.0, 0.0
    potential, force = lennard_jones_pair(radius, epsilon, sigma)
    cutoff_potential, cutoff_force = lennard_jones_pair(cutoff, epsilon, sigma)
    return potential - cutoff_potential + (radius - cutoff) * cutoff_force, force - cutoff_force


def _validate_pairs(pairs: NDArray[np.int64], particles: int) -> None:
    if (
        not isinstance(pairs, np.ndarray)
        or pairs.dtype != np.dtype(np.int64)
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or len(pairs) > MAX_NEIGHBOR_PAIRS
    ):
        raise ValueError("pairs must be bounded native int64 with shape (P,2)")
    if not len(pairs):
        return
    if (
        (pairs[:, 0] < 0).any()
        or (pairs[:, 1] >= particles).any()
        or not (pairs[:, 0] < pairs[:, 1]).all()
    ):
        raise ValueError("pairs must have valid particle indices i < j")
    keys = pairs[:, 0] * particles + pairs[:, 1]
    if not (np.diff(keys) > 0).all():
        raise ValueError("pairs must be unique and sorted in (i,j) order")


def _numpy_forces(
    positions: NDArray[np.float64],
    box: NDArray[np.float64],
    epsilon: float,
    sigma: float,
    cutoff: float,
    pairs: NDArray[np.int64] | None,
) -> tuple[NDArray[np.float64], float, float]:
    forces = np.zeros_like(positions)
    potential_energy = virial = 0.0
    cutoff_potential, cutoff_force = lennard_jones_pair(cutoff, epsilon, sigma)
    cutoff_squared, minimum_squared = cutoff * cutoff, (0.5 * sigma) ** 2

    def evaluate(displacements: NDArray[np.float64]):
        displacements -= box * np.rint(displacements / box)
        radius_squared = np.einsum("ij,ij->i", displacements, displacements)
        if (radius_squared < minimum_squared).any():
            raise ParticleOverlapError(
                "Particles are closer than 0.5 sigma; reduce timestep/density"
            )
        active = radius_squared < cutoff_squared
        radius = np.sqrt(radius_squared[active])
        sixth = (sigma * sigma / radius_squared[active]) ** 3
        bare_potential = 4.0 * epsilon * (sixth * sixth - sixth)
        radial = 24.0 * epsilon / radius * (2.0 * sixth * sixth - sixth) - cutoff_force
        pair_forces = (radial / radius)[:, None] * displacements[active]
        energy = bare_potential - cutoff_potential + (radius - cutoff) * cutoff_force
        pair_virial = np.einsum("ij,ij->i", displacements[active], pair_forces)
        return active, pair_forces, float(np.sum(energy)), float(np.sum(pair_virial))

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if pairs is None:
            for index in range(len(positions) - 1):
                active, values, energy, pair_virial = evaluate(
                    positions[index] - positions[index + 1 :]
                )
                forces[index] += np.sum(values, axis=0)
                indices = np.flatnonzero(active) + index + 1
                forces[indices] -= values
                potential_energy += energy
                virial += pair_virial
        else:
            active, values, potential_energy, virial = evaluate(
                positions[pairs[:, 0]] - positions[pairs[:, 1]]
            )
            selected = pairs[active]
            np.add.at(forces, selected[:, 0], values)
            np.add.at(forces, selected[:, 1], -values)
    return forces, potential_energy, virial


def compute_forces(
    positions: NDArray[np.float64],
    box_size: object,
    epsilon: float,
    sigma: float,
    cutoff_radius: float,
    backend: Literal["numpy", "numba"] = "numpy",
    pairs: NDArray[np.int64] | None = None,
) -> ForceResult:
    """Evaluate the same shifted-force model directly or on candidate pairs.

    A supplied list is responsible for containing every interacting pair. The
    VerletList class provides that guarantee using a skin/displacement fence.
    Candidate evaluations include pairs later excluded by the physical cutoff.
    """
    _positions(positions)
    box = _box(box_size)
    epsilon, sigma = _positive(epsilon, "epsilon"), _positive(sigma, "sigma")
    cutoff = _positive(cutoff_radius, "cutoff_radius")
    if cutoff >= 0.5 * float(np.min(box)):
        raise ValueError("cutoff_radius must be less than half the shortest box length")
    if pairs is not None:
        _validate_pairs(pairs, len(positions))
    require_backend(backend)
    wrapped = _wrapped(positions, box)
    if backend == "numba":
        try:
            from .lj_numba import forces_direct, forces_pairs
        except ImportError as error:
            raise BackendUnavailable("Numba Lennard–Jones backend could not be imported") from error
        arguments = (wrapped, box, epsilon, sigma, cutoff)
        values = forces_direct(*arguments) if pairs is None else forces_pairs(*arguments, pairs)
        forces, energy, virial, overlap = values
        if overlap:
            raise ParticleOverlapError(
                "Particles are closer than 0.5 sigma; reduce timestep/density"
            )
    else:
        forces, energy, virial = _numpy_forces(wrapped, box, epsilon, sigma, cutoff, pairs)
    if not np.isfinite(forces).all() or not math.isfinite(energy) or not math.isfinite(virial):
        raise FloatingPointError("Lennard–Jones forces or observables are not finite")
    return ForceResult(
        forces=forces,
        potential_energy=float(energy),
        virial=float(virial),
        pair_evaluations=len(pairs)
        if pairs is not None
        else len(positions) * (len(positions) - 1) // 2,
    )


class VerletList:
    """Sparse-cell neighbour list, rebuilt before any displacement reaches skin/2."""

    def __init__(
        self,
        box_size: object,
        cutoff_radius: float,
        skin: float,
        max_pairs: int = MAX_NEIGHBOR_PAIRS,
        max_candidate_checks: int = 100_000_000,
    ) -> None:
        self.box_size = _box(box_size)
        self.cutoff_radius = _positive(cutoff_radius, "cutoff_radius")
        self.skin = _positive(skin, "skin")
        self.radius = self.cutoff_radius + self.skin
        if self.radius >= 0.5 * float(np.min(self.box_size)):
            raise ValueError("cutoff_radius + skin must be less than half the shortest box length")
        if type(max_pairs) is not int or not 1 <= max_pairs <= MAX_NEIGHBOR_PAIRS:
            raise ValueError("max_pairs must be an integer in 1..1000000")
        if type(max_candidate_checks) is not int or max_candidate_checks < 0:
            raise ValueError("max_candidate_checks must be a nonnegative integer")
        self.max_pairs, self.max_candidate_checks = max_pairs, max_candidate_checks
        self.rebuild_count = self.candidate_checks = self.last_candidate_checks = 0
        self._reference: NDArray[np.float64] | None = None
        self._pairs: NDArray[np.int64] | None = None

    def pairs(
        self,
        unwrapped_positions: NDArray[np.float64],
        *,
        max_candidate_checks: int | None = None,
    ) -> NDArray[np.int64]:
        """Return immutable canonical pairs; an optional budget limits this build."""
        _positions(unwrapped_positions)
        if max_candidate_checks is not None and (
            type(max_candidate_checks) is not int or max_candidate_checks < 0
        ):
            raise ValueError("max_candidate_checks must be a nonnegative integer")
        self.last_candidate_checks = 0
        if self._reference is not None and self._reference.shape == unwrapped_positions.shape:
            with np.errstate(over="raise", invalid="raise"):
                displacement = unwrapped_positions - self._reference
                maximum = float(np.max(np.einsum("ij,ij->i", displacement, displacement)))
            if maximum < (0.5 * self.skin) ** 2:
                assert self._pairs is not None
                return self._pairs
        wrapped = _wrapped(unwrapped_positions, self.box_size)
        cell_counts = tuple(math.floor(float(length) / self.radius) for length in self.box_size)
        if any(count > 2**31 - 1 for count in cell_counts):
            raise ValueError("Box/cutoff ratio is outside the supported cell-index range")
        widths = self.box_size / np.asarray(cell_counts, dtype=np.float64)
        indices = np.minimum(
            np.floor(wrapped / widths).astype(np.int64), np.asarray(cell_counts) - 1
        )
        cells: dict[tuple[int, int], list[int]] = {}
        for index, cell in enumerate(indices):
            key = int(cell[0]), int(cell[1])
            cells.setdefault(key, []).append(index)
        pairs: list[tuple[int, int]] = []
        remaining = self.max_candidate_checks - self.candidate_checks
        if max_candidate_checks is not None:
            remaining = min(remaining, max_candidate_checks)
        radius_squared = self.radius * self.radius
        with np.errstate(over="raise", invalid="raise"):
            for first, cell in enumerate(indices):
                neighbours = sorted(
                    {
                        ((int(cell[0]) + dx) % cell_counts[0], (int(cell[1]) + dy) % cell_counts[1])
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                    }
                )
                candidates = np.fromiter(
                    (
                        second
                        for neighbour in neighbours
                        for second in cells.get(neighbour, ())
                        if second > first
                    ),
                    dtype=np.int64,
                )
                # Each row has at most N-1 entries. Batch its geometric tests
                # without creating a dense N-by-N distance array. If the work
                # budget ends inside this row, evaluate only its allowed prefix.
                available = remaining - self.last_candidate_checks
                exhausted = len(candidates) > available
                candidates = candidates[:available]
                if len(candidates):
                    self.candidate_checks += len(candidates)
                    self.last_candidate_checks += len(candidates)
                    displacement = wrapped[first] - wrapped[candidates]
                    displacement -= self.box_size * np.rint(displacement / self.box_size)
                    distances = np.einsum("ij,ij->i", displacement, displacement)
                    selected = candidates[distances < radius_squared]
                    if len(pairs) + len(selected) > self.max_pairs:
                        raise NeighborCapacityError("Neighbour pair capacity exhausted")
                    pairs.extend((first, int(second)) for second in selected)
                if exhausted:
                    raise PairBudgetExceeded("Neighbour candidate work budget exhausted")
        result = np.asarray(sorted(pairs), dtype=np.int64).reshape((-1, 2))
        result.setflags(write=False)
        self._pairs = result
        self._reference = unwrapped_positions.copy()
        self.rebuild_count += 1
        return result
