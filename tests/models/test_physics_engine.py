import math

import numpy as np
import pytest

from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models.physics_engine import IdealGasSimulator


def test_seed_is_local_and_repeats_the_entire_trajectory() -> None:
    first = IdealGasSimulator(80, 300.0, 0.2, seed=42)
    second = IdealGasSimulator(80, 300.0, 0.2, seed=42)
    different = IdealGasSimulator(80, 300.0, 0.2, seed=43)
    state_before = np.random.get_state(legacy=True)
    for _ in range(15):
        assert first.step() == second.step()
        np.testing.assert_array_equal(first.velocities, second.velocities)
    state_after = np.random.get_state(legacy=True)
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    assert not np.array_equal(first.velocities, different.velocities)


@pytest.mark.parametrize("dt", [0.001, 0.2, 2.0, 1000.0])
def test_thermostat_preserves_equilibrium_without_systematic_heating(dt: float) -> None:
    particles, target_temperature = 40_000, 300.0
    simulator = IdealGasSimulator(particles, target_temperature, dt, seed=2026)
    for _ in range(20):
        temperature, energy = simulator.step()
    # In 2D, T_hat is Gamma(N, T/N) with exact standard error T/sqrt(N).
    assert abs(temperature - target_temperature) < 6 * target_temperature / math.sqrt(particles)
    assert energy == temperature * particles


def test_exact_ou_conditional_mean_and_variance() -> None:
    particles, target_temperature, dt = 100_000, 3.0, 0.2
    simulator = IdealGasSimulator(particles, target_temperature, dt, seed=123)
    simulator.velocities.fill(2.0)
    simulator.step()
    expected_mean = 2.0 * math.exp(-dt)
    expected_variance = target_temperature * -math.expm1(-2.0 * dt)
    observations = 2 * particles
    assert abs(float(simulator.velocities.mean()) - expected_mean) < 6 * math.sqrt(
        expected_variance / observations
    )
    assert abs(
        float(simulator.velocities.var()) - expected_variance
    ) < 6 * expected_variance * math.sqrt(2 / (observations - 1))


def test_temperature_uncertainty_matches_independent_equilibrium_ensembles() -> None:
    ensembles, particles_per_ensemble, target = 1000, 100, 300.0
    simulator = IdealGasSimulator(ensembles * particles_per_ensemble, target, 2.0, seed=2026)
    simulator.step()
    velocities = simulator.velocities.reshape(ensembles, particles_per_ensemble, 2)
    temperatures = 0.5 * np.sum(velocities**2, axis=(1, 2)) / particles_per_ensemble
    predicted_standard_error = target / math.sqrt(particles_per_ensemble)
    # For independent Gamma(N, T/N) temperature estimates, the sample standard
    # deviation converges to T/sqrt(N). Allow six asymptotic standard errors.
    observed_standard_error = float(temperatures.std(ddof=1))
    tolerance = 6 * predicted_standard_error / math.sqrt(2 * (ensembles - 1))
    assert abs(observed_standard_error - predicted_standard_error) < tolerance


def test_nonfinite_numerical_state_cannot_be_reported_as_success() -> None:
    simulator = IdealGasSimulator(10, 300.0, 0.1)
    simulator.velocities[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        simulator.step()


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_temperature": float("nan")},
        {"initial_temperature": float("inf")},
        {"initial_temperature": 10**1000},
        {"initial_temperature": True},
        {"time_step": float("inf")},
        {"time_step": 0.0},
        {"num_particles": True},
        {"num_particles": 1.5},
        {"num_steps": 0},
        {"num_steps": 10_001},
        {"num_particles": 1_000_000, "num_steps": 101},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
    ],
)
def test_domain_defensively_rejects_invalid_inputs(changes: dict[str, object]) -> None:
    inputs = {"initial_temperature": 300.0, "num_particles": 10, **changes}
    with pytest.raises(ValueError):
        SimulationParameters(**inputs)  # type: ignore[arg-type]
