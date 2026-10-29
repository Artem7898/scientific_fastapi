"""Independent force, periodic geometry and neighbour-list correctness checks."""

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest

from scientific_api.models.lj_forces import (
    NeighborCapacityError,
    PairBudgetExceeded,
    ParticleOverlapError,
    VerletList,
    compute_forces,
    lennard_jones_pair,
    shifted_force_pair,
)

BACKENDS = [
    "numpy",
    pytest.param(
        "numba",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
        ),
    ),
]


def configuration(particles=64, length=12.0, seed=2026):
    side = math.ceil(math.sqrt(particles))
    positions = np.asarray(
        [
            ((i + 0.5) * length / side, (j + 0.5) * length / side)
            for i in range(side)
            for j in range(side)
        ],
        dtype=np.float64,
    )[:particles]
    rng = np.random.Generator(np.random.PCG64(seed))
    return positions + rng.uniform(-0.04, 0.04, positions.shape)


@pytest.mark.parametrize("radius", [0.9, 1.05, 1.3, 1.8, 2.499, 2.501])
def test_shifted_force_is_negative_derivative_of_its_own_potential(radius):
    step = 1e-6
    potential, force = shifted_force_pair(radius, epsilon=1.7, sigma=0.9, cutoff_radius=2.5)
    high = shifted_force_pair(radius + step, 1.7, 0.9, 2.5)[0]
    low = shifted_force_pair(radius - step, 1.7, 0.9, 2.5)[0]
    assert force == pytest.approx(-(high - low) / (2 * step), rel=2e-8, abs=2e-8)
    assert math.isfinite(potential)


def test_cutoff_is_continuous_in_both_potential_and_force():
    for radius in (2.5, 2.6, 3.0):
        assert shifted_force_pair(radius) == (0.0, 0.0)
    near_potential, near_force = shifted_force_pair(2.5 - 1e-4)
    closer_potential, closer_force = shifted_force_pair(2.5 - 5e-5)
    assert abs(near_potential) < 1e-8
    assert abs(near_force) < 2e-5
    assert abs(closer_potential / near_potential) == pytest.approx(0.25, rel=5e-4)
    assert abs(closer_force / near_force) == pytest.approx(0.5, rel=5e-4)


def test_bare_and_shifted_potential_minima_are_distinct_and_each_has_zero_force():
    bare_minimum = 2 ** (1 / 6)
    potential, force = lennard_jones_pair(bare_minimum)
    assert potential == pytest.approx(-1.0)
    assert force == pytest.approx(0.0, abs=1e-13)
    # A force shift changes the minimum; checking F_sf(r_bare)=0 would be wrong.
    assert shifted_force_pair(bare_minimum)[1] > 0.03
    low, high = bare_minimum, 1.2
    for _ in range(55):
        middle = (low + high) / 2
        if shifted_force_pair(middle)[1] > 0:
            low = middle
        else:
            high = middle
    shifted_minimum = (low + high) / 2
    assert shifted_minimum > bare_minimum
    value, force = shifted_force_pair(shifted_minimum)
    assert abs(force) < 1e-13
    assert shifted_force_pair(shifted_minimum - 1e-4)[0] > value
    assert shifted_force_pair(shifted_minimum + 1e-4)[0] > value


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_force_matches_independent_coordinate_energy_gradient(backend):
    positions = np.array([[1.0, 1.0], [2.2, 1.3], [4.0, 2.0]], dtype=np.float64)
    output = compute_forces(positions, (12.0, 12.0), 1.3, 0.9, 2.5, backend)
    gradient = np.zeros_like(positions)
    step = 1e-6
    for index in range(len(positions)):
        for dimension in range(2):
            high, low = positions.copy(), positions.copy()
            high[index, dimension] += step
            low[index, dimension] -= step
            gradient[index, dimension] = -(
                compute_forces(high, (12.0, 12.0), 1.3, 0.9, 2.5).potential_energy
                - compute_forces(low, (12.0, 12.0), 1.3, 0.9, 2.5).potential_energy
            ) / (2 * step)
    np.testing.assert_allclose(output.forces, gradient, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(output.forces.sum(axis=0), [0.0, 0.0], atol=1e-14)
    assert output.pair_evaluations == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_pair_action_reaction_virial_and_signed_attractive_energy(backend):
    positions = np.array([[1.0, 1.0], [2.2, 1.0]], dtype=np.float64)
    output = compute_forces(positions, (10.0, 10.0), 1.0, 1.0, 2.5, backend)
    energy, radial = shifted_force_pair(1.2)
    assert output.potential_energy == pytest.approx(energy)
    assert energy < 0
    assert output.forces[0, 0] > 0  # attraction towards the second particle
    np.testing.assert_array_equal(output.forces[0], -output.forces[1])
    assert output.forces[0, 1] == 0
    assert output.virial == pytest.approx(1.2 * radial)


@pytest.mark.parametrize("backend", BACKENDS)
def test_minimum_image_and_arbitrary_image_translations_preserve_forces(backend):
    positions = np.array([[0.1, 1.0], [8.9, 1.3], [4.2, 4.6]], dtype=np.float64)
    box = np.array([10.0, 12.0])
    first = compute_forces(positions, box, 1.0, 1.0, 2.5, backend)
    shifted = positions + np.array([[1, -2], [-3, 2], [5, 1]]) * box
    second = compute_forces(shifted, box, 1.0, 1.0, 2.5, backend)
    np.testing.assert_allclose(first.forces, second.forces, rtol=1e-12, atol=1e-12)
    assert first.potential_energy == pytest.approx(second.potential_energy, rel=1e-12)
    assert first.virial == pytest.approx(second.virial, rel=1e-12)
    displacement = positions[0] - positions[1]
    displacement -= box * np.rint(displacement / box)
    assert np.linalg.norm(displacement) == pytest.approx(math.hypot(1.2, 0.3))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length,particles", [(6.0, 9), (12.0, 64), (20.0, 100)])
def test_cell_verlet_matches_every_direct_force_energy_and_virial(backend, length, particles):
    positions = configuration(particles, length)
    neighbours = VerletList((length, length), 2.5, 0.3)
    pairs = neighbours.pairs(positions)
    direct = compute_forces(positions, (length, length), 1.0, 1.0, 2.5, backend)
    sparse = compute_forces(positions, (length, length), 1.0, 1.0, 2.5, backend, pairs)
    np.testing.assert_allclose(sparse.forces, direct.forces, rtol=1e-12, atol=1e-12)
    assert sparse.potential_energy == pytest.approx(direct.potential_energy, rel=1e-12, abs=1e-12)
    assert sparse.virial == pytest.approx(direct.virial, rel=1e-12, abs=1e-12)
    assert sparse.pair_evaluations == len(pairs)
    assert not pairs.flags.writeable
    assert neighbours.rebuild_count == 1
    assert neighbours.candidate_checks <= particles * (particles - 1) // 2
    if length == 6.0:
        # Two cells per direction: modulo maps several offsets to the same
        # neighbour cell, so deduplication must still count each pair once.
        assert neighbours.candidate_checks == particles * (particles - 1) // 2


def test_skin_fence_rebuilds_before_a_previously_excluded_pair_can_interact():
    positions = np.array([[1.0, 1.0], [3.81, 1.0]], dtype=np.float64)
    neighbours = VerletList((12.0, 12.0), 2.5, 0.3)
    first = neighbours.pairs(positions)
    assert first.shape == (0, 2)
    moved = positions + np.array([[0.14, 0.0], [-0.14, 0.0]])
    assert neighbours.pairs(moved) is first
    assert neighbours.rebuild_count == 1
    assert compute_forces(moved, (12.0, 12.0), 1.0, 1.0, 2.5, pairs=first).potential_energy == 0
    moved = positions + np.array([[0.16, 0.0], [-0.16, 0.0]])
    rebuilt = neighbours.pairs(moved)
    np.testing.assert_array_equal(rebuilt, [[0, 1]])
    assert neighbours.rebuild_count == 2
    direct = compute_forces(moved, (12.0, 12.0), 1.0, 1.0, 2.5)
    sparse = compute_forces(moved, (12.0, 12.0), 1.0, 1.0, 2.5, pairs=rebuilt)
    np.testing.assert_array_equal(direct.forces, sparse.forces)
    assert sparse.potential_energy < 0


def test_rebuild_uses_unwrapped_displacement_across_periodic_boundaries():
    positions = np.array([[11.95, 1.0], [1.1, 1.0]], dtype=np.float64)
    neighbours = VerletList((12.0, 12.0), 2.5, 0.3)
    first = neighbours.pairs(positions)
    positions[0, 0] += 0.09  # crossing L must not be interpreted as an 11.9 shift
    assert neighbours.pairs(positions) is first
    assert neighbours.rebuild_count == 1
    positions[0, 0] += 0.07
    pairs = neighbours.pairs(positions)
    assert neighbours.rebuild_count == 2
    direct = compute_forces(positions, (12.0, 12.0), 1.0, 1.0, 2.5)
    sparse = compute_forces(positions, (12.0, 12.0), 1.0, 1.0, 2.5, pairs=pairs)
    np.testing.assert_array_equal(direct.forces, sparse.forces)


def test_reused_neighbours_match_direct_for_moving_particles_in_a_rectangular_box():
    box = np.array([12.0, 9.0])
    positions = np.array(
        [[0.5 + 1.5 * i, 0.5 + 1.5 * j] for i in range(8) for j in range(6)],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(2026))
    positions += rng.integers(-2, 3, size=positions.shape) * box
    neighbours = VerletList(box, 2.5, 0.3)
    for _ in range(40):
        pairs = neighbours.pairs(positions)
        direct = compute_forces(positions, box, 1.0, 1.0, 2.5)
        cached = compute_forces(positions, box, 1.0, 1.0, 2.5, pairs=pairs)
        np.testing.assert_allclose(cached.forces, direct.forces, rtol=1e-11, atol=1e-11)
        assert cached.potential_energy == pytest.approx(direct.potential_energy, rel=1e-12)
        assert cached.virial == pytest.approx(direct.virial, rel=1e-12)
        positions += rng.normal(0.0, 0.01, size=positions.shape)
    assert neighbours.rebuild_count > 1


def test_sparse_cells_do_not_allocate_the_enormous_empty_grid():
    positions = configuration(16, 100.0)
    neighbours = VerletList((1e6, 1e6), 2.5, 0.3)
    pairs = neighbours.pairs(positions)
    assert pairs.shape == (0, 2)
    assert neighbours.candidate_checks == 0


def test_cell_rounding_just_below_box_edge_does_not_hide_a_pair():
    length = 14.001999199919993
    below_edge = np.nextafter(length, 0.0)
    # Dividing a valid coordinate just below L by a cell width can round to
    # exactly n_cells. It must remain in the last cell, not an unreachable key.
    positions = np.array([[length - 1.2, 1.0], [below_edge, 1.0]], dtype=np.float64)
    pairs = VerletList((length, length), 2.5, 0.3).pairs(positions)
    np.testing.assert_array_equal(pairs, [[0, 1]])
    direct = compute_forces(positions, (length, length), 1.0, 1.0, 2.5)
    sparse = compute_forces(positions, (length, length), 1.0, 1.0, 2.5, pairs=pairs)
    np.testing.assert_array_equal(direct.forces, sparse.forces)
    assert direct.potential_energy == sparse.potential_energy


def test_candidate_budgets_count_actual_search_and_allow_zero_work_cache_hits():
    positions = np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0]], dtype=np.float64)
    neighbours = VerletList((12.0, 12.0), 2.5, 0.3, max_candidate_checks=3)
    pairs = neighbours.pairs(positions, max_candidate_checks=3)
    assert neighbours.candidate_checks == neighbours.last_candidate_checks == 3
    assert neighbours.pairs(positions, max_candidate_checks=0) is pairs
    assert neighbours.last_candidate_checks == 0
    with pytest.raises(PairBudgetExceeded):
        neighbours.pairs(positions + 0.2, max_candidate_checks=100)
    assert neighbours.candidate_checks == 3
    partial = VerletList((12.0, 12.0), 2.5, 0.3)
    with pytest.raises(PairBudgetExceeded):
        partial.pairs(positions, max_candidate_checks=2)
    assert partial.candidate_checks == partial.last_candidate_checks == 2
    assert partial.rebuild_count == 0


def test_pair_storage_capacity_is_checked_before_appending_an_excess_pair():
    positions = np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0]], dtype=np.float64)
    with pytest.raises(NeighborCapacityError):
        VerletList((12.0, 12.0), 2.5, 0.3, max_pairs=1).pairs(positions)


def test_partial_candidate_row_never_evaluates_beyond_its_remaining_work_budget(monkeypatch):
    positions = np.array(
        [[1.0, 1.0], [2.0, 1.0], [1.5, 2.0], [2.0, 2.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    original = np.einsum
    evaluated_rows = []

    def record_geometry(subscripts, left, right):
        if subscripts == "ij,ij->i":
            evaluated_rows.append(len(left))
        return original(subscripts, left, right)

    monkeypatch.setattr(np, "einsum", record_geometry)
    neighbours = VerletList((12.0, 12.0), 2.5, 0.3)
    with pytest.raises(PairBudgetExceeded):
        neighbours.pairs(positions, max_candidate_checks=2)
    assert evaluated_rows == [2]
    assert neighbours.candidate_checks == neighbours.last_candidate_checks == 2
    assert neighbours.rebuild_count == 0
    # The row prefix is bounded independently of its neighbours being active;
    # a zero remaining budget cannot perform any geometric pair tests.
    evaluated_rows.clear()
    with pytest.raises(PairBudgetExceeded):
        neighbours.pairs(positions, max_candidate_checks=0)
    assert evaluated_rows == []
    assert neighbours.candidate_checks == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsoftened_near_overlap_is_an_explicit_error(backend):
    positions = np.array([[1.0, 1.0], [1.49, 1.0]], dtype=np.float64)
    with pytest.raises(ParticleOverlapError, match="0.5 sigma"):
        compute_forces(positions, (12.0, 12.0), 1.0, 1.0, 2.5, backend)
    with pytest.raises(ParticleOverlapError):
        compute_forces(
            positions, (12.0, 12.0), 1.0, 1.0, 2.5, backend, np.array([[0, 1]], dtype=np.int64)
        )


@pytest.mark.parametrize(
    "pairs",
    [
        np.array([[0, 0]], dtype=np.int64),
        np.array([[1, 0]], dtype=np.int64),
        np.array([[0, 3]], dtype=np.int64),
        np.array([[0, 1], [0, 1]], dtype=np.int64),
        np.array([[1, 2], [0, 1]], dtype=np.int64),
        np.array([[0.0, 1.0]]),
    ],
)
def test_invalid_pair_lists_cannot_double_count_or_select_invalid_particles(pairs):
    positions = np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="pairs"):
        compute_forces(positions, (12.0, 12.0), 1.0, 1.0, 2.5, pairs=pairs)


def test_nonfinite_and_invalid_physical_geometry_is_rejected():
    positions = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="positions"):
        compute_forces(positions * np.nan, (12.0, 12.0), 1.0, 1.0, 2.5)
    with pytest.raises(ValueError, match="box_size"):
        compute_forces(positions, (12.0, np.inf), 1.0, 1.0, 2.5)
    with pytest.raises(ValueError, match="half"):
        compute_forces(positions, (5.0, 12.0), 1.0, 1.0, 2.5)
    with pytest.raises(ValueError, match="half"):
        VerletList((5.6, 12.0), 2.5, 0.3)
    with pytest.raises(ValueError, match="epsilon"):
        compute_forces(positions, (12.0, 12.0), True, 1.0, 2.5)


def test_numpy_force_import_and_execution_do_not_load_optional_numba():
    script = """
import sys
import numpy as np
from scientific_api.models.lj_forces import compute_forces, VerletList
positions=np.array([[1.,1.],[2.2,1.]],dtype=np.float64)
pairlist=VerletList((12.,12.),2.5,.3)
compute_forces(positions,(12.,12.),1.,1.,2.5,pairs=pairlist.pairs(positions))
assert not set(sys.modules) & {'numba','llvmlite','jax','jaxlib'}
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
)
def test_numba_matches_numpy_for_the_same_snapshot_and_pair_order():
    positions = configuration(100, 15.0)
    neighbours = VerletList((15.0, 15.0), 2.5, 0.3)
    for pairs in (None, neighbours.pairs(positions)):
        numpy = compute_forces(positions, (15.0, 15.0), 1.0, 1.0, 2.5, "numpy", pairs)
        numba = compute_forces(positions, (15.0, 15.0), 1.0, 1.0, 2.5, "numba", pairs)
        np.testing.assert_allclose(numba.forces, numpy.forces, rtol=1e-12, atol=1e-12)
        assert numba.potential_energy == pytest.approx(numpy.potential_energy, rel=1e-12)
        assert numba.virial == pytest.approx(numpy.virial, rel=1e-12)
        assert numba.pair_evaluations == numpy.pair_evaluations
