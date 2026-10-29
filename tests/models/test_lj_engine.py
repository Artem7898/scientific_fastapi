"""Integrator tests use conservation laws, limiting distributions and convergence."""

import importlib.util
import math

import numpy as np
import pytest

import scientific_api.models.lennard_jones_engine as engine_module
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.models.lennard_jones_engine import create_lj_simulator
from scientific_api.models.lj_forces import PairBudgetExceeded

BACKENDS = [
    "numpy",
    pytest.param(
        "numba",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
        ),
    ),
]


def parameters(**changes):
    return LennardJonesParameters(
        **{
            "num_particles": 16,
            "box_size": (6.0, 6.0),
            "initial_temperature": 0.5,
            "time_step": 0.002,
            "num_steps": 100,
            "seed": 2026,
            **changes,
        }
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("interaction", ["direct", "verlet"])
def test_lattice_initialization_has_target_thermal_temperature_and_zero_momentum(
    backend, interaction
):
    configuration = parameters(mass=2.0, backend=backend, interaction_backend=interaction)
    simulator = create_lj_simulator(configuration)
    observed = simulator.observables()
    assert observed["thermal_temperature"] == pytest.approx(0.5, rel=1e-14)
    np.testing.assert_allclose(simulator.velocities.sum(axis=0), 0.0, atol=2e-15)
    expected_kinetic = (configuration.num_particles - 1) * configuration.initial_temperature
    assert observed["kinetic_energy"] == pytest.approx(expected_kinetic, rel=1e-14)
    assert observed["total_energy"] == observed["kinetic_energy"] + observed["potential_energy"]
    assert observed["pressure"] == pytest.approx(
        (2 * observed["kinetic_energy"] + observed["virial"]) / (2 * 36)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_velocity_verlet_energy_error_decreases_quadratically_with_time_step(backend):
    errors = []
    for time_step in (0.008, 0.004, 0.002):
        simulator = create_lj_simulator(
            parameters(time_step=time_step, num_steps=round(2 / time_step), backend=backend)
        )
        initial = simulator.observables()["total_energy"]
        maximum = 0.0
        for _ in range(round(2 / time_step)):
            simulator.advance()
            maximum = max(maximum, abs(simulator.observables()["total_energy"] - initial))
        errors.append(maximum)
    assert errors[0] > 0
    assert errors[1] < 0.4 * errors[0]
    assert errors[2] < 0.4 * errors[1]
    assert errors[-1] < 0.01


@pytest.mark.parametrize("backend", BACKENDS)
def test_nve_conserves_momentum_and_is_time_reversible(backend):
    simulator = create_lj_simulator(parameters(backend=backend))
    initial_positions = simulator.unwrapped_positions.copy()
    initial_velocities = simulator.velocities.copy()
    initial_momentum = simulator.velocities.sum(axis=0)
    for _ in range(100):
        simulator.advance()
    np.testing.assert_allclose(simulator.velocities.sum(axis=0), initial_momentum, atol=3e-13)
    simulator.velocities *= -1
    for _ in range(100):
        simulator.advance()
    np.testing.assert_allclose(simulator.unwrapped_positions, initial_positions, atol=1e-12)
    np.testing.assert_allclose(simulator.velocities, -initial_velocities, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_direct_and_neighbor_trajectories_match_over_a_short_common_initial_state(
    backend, ensemble
):
    options = {"ensemble": ensemble, "backend": backend}
    if ensemble == "nvt":
        options.update(bath_temperature=0.75, friction=2.0)
    direct = create_lj_simulator(parameters(interaction_backend="direct", **options))
    neighbor = create_lj_simulator(parameters(interaction_backend="verlet", **options))
    for _ in range(50):
        direct.advance()
        neighbor.advance()
        np.testing.assert_allclose(direct.forces, neighbor.forces, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            direct.unwrapped_positions, neighbor.unwrapped_positions, atol=1e-12
        )
        np.testing.assert_allclose(direct.velocities, neighbor.velocities, atol=1e-12)
    for name, value in direct.observables().items():
        assert neighbor.observables()[name] == pytest.approx(value, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_baoab_free_gas_limit_recovers_stationary_gaussian_velocity_statistics(backend):
    # Large interparticle separation makes the force exactly zero. The BAOAB O
    # substep is then an independent exact OU process, with known Maxwell law.
    configuration = LennardJonesParameters(
        num_particles=1024,
        box_size=(1000.0, 1000.0),
        initial_temperature=0.1,
        ensemble="nvt",
        bath_temperature=2.0,
        friction=30.0,
        neighbor_skin=1.0,
        time_step=0.002,
        num_steps=1000,
        seed=42,
        backend=backend,
    )
    simulator = create_lj_simulator(configuration)
    for _ in range(configuration.num_steps):
        simulator.advance()
    assert simulator.observables()["potential_energy"] == 0.0
    variance = configuration.bath_temperature / configuration.mass
    particles = configuration.num_particles
    assert np.max(np.abs(simulator.velocities.mean(axis=0))) < 6 * math.sqrt(variance / particles)
    assert np.max(np.abs(simulator.velocities.var(axis=0, ddof=1) - variance)) < (
        6 * variance * math.sqrt(2 / (particles - 1))
    )
    temperature = simulator.observables()["thermal_temperature"]
    assert abs(temperature - configuration.bath_temperature) < (
        6 * configuration.bath_temperature / math.sqrt(particles - 1)
    )
    squared_speeds = np.sum(simulator.velocities**2, axis=1)
    below_rayleigh_median = np.mean(squared_speeds < 2 * variance * math.log(2))
    assert abs(below_rayleigh_median - 0.5) < 6 * math.sqrt(0.25 / particles)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_seed_repeats_nvt_and_sampling_does_not_consume_randomness(backend):
    configuration = parameters(
        ensemble="nvt", bath_temperature=1.0, friction=2.0, backend=backend, seed=2**64 - 1
    )
    first = create_lj_simulator(configuration)
    second = create_lj_simulator(
        parameters(
            ensemble="nvt",
            bath_temperature=1.0,
            friction=2.0,
            backend=backend,
            seed=2**64 - 1,
            sample_stride=1,
        )
    )
    global_before = np.random.get_state(legacy=True)
    for _ in range(100):
        first.advance()
        first.observables()
        second.advance()
    np.testing.assert_array_equal(first.unwrapped_positions, second.unwrapped_positions)
    np.testing.assert_array_equal(first.velocities, second.velocities)
    global_after = np.random.get_state(legacy=True)
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


def test_signed_energy_and_pressure_are_not_clipped():
    simulator = create_lj_simulator(parameters(initial_temperature=0.01))
    values = simulator.observables()
    assert values["potential_energy"] < 0
    assert values["total_energy"] < 0
    assert values["pressure"] < 0


def test_periodic_presentation_preserves_unwrapped_state_and_half_open_box():
    simulator = create_lj_simulator(parameters())
    simulator.unwrapped_positions[0] = [-1e-30, 31.0]
    original = simulator.unwrapped_positions.copy()
    assert np.all((simulator.positions >= 0) & (simulator.positions < 6.0))
    assert simulator.positions[0, 0] == 0.0
    np.testing.assert_array_equal(simulator.unwrapped_positions, original)


def test_work_budget_checks_force_and_analysis_before_work(monkeypatch):
    monkeypatch.setattr(engine_module, "MAX_LJ_PAIR_EVALUATIONS", 240)
    simulator = create_lj_simulator(parameters(interaction_backend="direct"))
    assert simulator.diagnostics()["pair_evaluations"] == 120
    simulator.reserve_analysis_pairs(120)
    assert simulator.diagnostics()["total_pair_checks"] == 240
    with pytest.raises(PairBudgetExceeded, match="analysis"):
        simulator.reserve_analysis_pairs(1)
    with pytest.raises(PairBudgetExceeded, match="force"):
        simulator.advance()
    assert simulator.diagnostics()["pair_evaluations"] == 120
    for value in (-1, True):
        with pytest.raises(ValueError, match="pair count"):
            simulator.reserve_analysis_pairs(value)


def test_neighbor_candidate_search_obeys_same_total_pair_budget(monkeypatch):
    monkeypatch.setattr(engine_module, "MAX_LJ_PAIR_EVALUATIONS", 1)
    with pytest.raises(PairBudgetExceeded):
        create_lj_simulator(parameters())


@pytest.mark.parametrize("backend", BACKENDS)
def test_execution_metadata_identifies_integrator_force_policy_and_backend(backend):
    simulator = create_lj_simulator(parameters(backend=backend))
    metadata = simulator.execution_metadata()
    assert metadata["backend"] == backend
    assert metadata["integrator"] == "velocity-verlet"
    assert metadata["interaction_backend"] == "verlet"
    assert metadata["dtype"] == "float64" and metadata["fastmath"] is False
    assert "shifted-force" in metadata["cutoff_policy"]
    assert metadata["rng_policy"] == "numpy-pcg64-lj-v1"
    if backend == "numba":
        assert metadata["llvm_version"] and metadata["llvmlite_version"]
        assert metadata["cache"] is False and metadata["nogil"] is True
