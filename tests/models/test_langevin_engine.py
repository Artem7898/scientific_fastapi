"""Independent Gaussian-transition, transport and reproducibility references."""

import importlib.util
import json
import math
import subprocess
import sys
import weakref
from dataclasses import replace

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.models import computation
from scientific_api.models.computation import BackendUnavailable
from scientific_api.models.langevin_engine import (
    create_langevin_simulator,
    exact_transition_coefficients,
    langevin_moments,
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


def parameters(**changes):
    return LangevinParameters(
        **{
            "initial_temperature": 2.0,
            "bath_temperature": 3.0,
            "num_particles": 128,
            "num_steps": 20,
            "seed": 2026,
            **changes,
        }
    )


@pytest.mark.parametrize("q", [1e-9, 1e-6, 0.001, 0.05, 0.1, 0.2, 2.0, 10.0])
def test_exact_covariance_matches_independent_noise_kernel_quadrature(q):
    # Integrate the two impulse-response kernels, rather than repeating the
    # closed-form coefficient formulas from the implementation.
    gamma, temperature_per_mass = 0.7, 2.3
    dt = q / gamma
    nodes, weights = np.polynomial.legendre.leggauss(96)
    times = 0.5 * dt * (nodes + 1)
    weights = 0.5 * dt * weights
    velocity_kernel = math.sqrt(2 * gamma * temperature_per_mass) * np.exp(-gamma * times)
    position_kernel = (
        math.sqrt(2 * gamma * temperature_per_mass) * -np.expm1(-gamma * times) / gamma
    )
    coefficient = exact_transition_coefficients(dt, gamma, temperature_per_mass)
    np.testing.assert_allclose(
        [
            coefficient.velocity_noise_variance,
            coefficient.position_noise_variance,
            coefficient.noise_covariance,
        ],
        [
            np.sum(weights * velocity_kernel**2),
            np.sum(weights * position_kernel**2),
            np.sum(weights * position_kernel * velocity_kernel),
        ],
        rtol=5e-12,
        atol=0,
    )
    assert coefficient.position_independent_noise_scale > 0


@pytest.mark.parametrize("gamma,dt", [(1e-3, 1e-6), (1e3, 10), (1.0, 100_000.0)])
def test_coefficients_remain_finite_and_positive_at_supported_extremes(gamma, dt):
    coefficient = exact_transition_coefficients(dt, gamma, 1e9)
    assert math.isfinite(coefficient.acceleration_response)
    assert coefficient.position_noise_variance > 0
    assert coefficient.velocity_noise_variance > 0
    assert coefficient.position_independent_noise_scale > 0
    assert coefficient.noise_covariance**2 < (
        coefficient.position_noise_variance * coefficient.velocity_noise_variance
    )


def test_zero_time_has_identity_transition_and_initial_moments():
    inputs = parameters(initial_mean_velocity=(2.0, -1.0), mass=2.0)
    coefficient = exact_transition_coefficients(0, 1.0, 3.0)
    assert coefficient.decay == 1
    assert coefficient.velocity_response == 0
    assert coefficient.acceleration_response == 0
    moments = langevin_moments(0, inputs)
    assert moments["mean_displacement"] == [0.0, 0.0]
    assert moments["mean_velocity"] == [2.0, -1.0]
    assert moments["position_variance_per_component"] == 0
    assert moments["position_velocity_covariance"] == 0
    assert moments["velocity_variance_per_component"] == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_empirical_joint_conditional_noise_covariance_and_drift(backend):
    inputs = parameters(
        backend=backend,
        num_particles=100_000,
        num_steps=1,
        time_step=0.4,
        friction=0.7,
        mass=1.3,
        initial_position=(1.0, -2.0),
        acceleration=(0.3, -0.2),
    )
    simulator = create_langevin_simulator(inputs)
    simulator.velocities[:] = (2.0, -1.0)
    simulator.advance()
    coefficient = exact_transition_coefficients(
        inputs.time_step, inputs.friction, inputs.bath_temperature / inputs.mass
    )
    mean_position = (
        np.asarray(inputs.initial_position)
        + coefficient.velocity_response * np.array([2.0, -1.0])
        + coefficient.acceleration_response * np.asarray(inputs.acceleration)
    )
    mean_velocity = coefficient.decay * np.array(
        [2.0, -1.0]
    ) + coefficient.velocity_response * np.asarray(inputs.acceleration)
    positions = simulator.unwrapped_positions - mean_position
    velocities = simulator.velocities - mean_velocity
    observations = 2 * inputs.num_particles
    assert abs(float(positions.mean())) < 6 * math.sqrt(
        coefficient.position_noise_variance / observations
    )
    assert abs(float(velocities.mean())) < 6 * math.sqrt(
        coefficient.velocity_noise_variance / observations
    )
    assert abs(float(np.mean(positions**2)) - coefficient.position_noise_variance) < (
        6 * coefficient.position_noise_variance * math.sqrt(2 / observations)
    )
    assert abs(float(np.mean(velocities**2)) - coefficient.velocity_noise_variance) < (
        6 * coefficient.velocity_noise_variance * math.sqrt(2 / observations)
    )
    assert abs(float(np.mean(positions * velocities)) - coefficient.noise_covariance) < (
        6
        * math.sqrt(
            (
                coefficient.position_noise_variance * coefficient.velocity_noise_variance
                + coefficient.noise_covariance**2
            )
            / observations
        )
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_multistep_nonequilibrium_marginals_match_analytic_solution(backend):
    inputs = parameters(
        backend=backend,
        num_particles=40_000,
        num_steps=15,
        time_step=0.2,
        initial_temperature=7.0,
        bath_temperature=2.0,
        mass=1.7,
        friction=0.8,
        acceleration=(0.2, -0.6),
        initial_mean_velocity=(0.5, -0.3),
        initial_position=(1.0, -1.0),
    )
    simulator = create_langevin_simulator(inputs)
    for _ in range(inputs.num_steps):
        simulator.advance()
    theory = langevin_moments(inputs.num_steps * inputs.time_step, inputs)
    positions = simulator.unwrapped_positions - np.asarray(inputs.initial_position)
    for values, mean, variance in (
        (positions, theory["mean_displacement"], theory["position_variance_per_component"]),
        (simulator.velocities, theory["mean_velocity"], theory["velocity_variance_per_component"]),
    ):
        np.testing.assert_allclose(
            values.mean(axis=0), mean, rtol=0, atol=6 * math.sqrt(variance / inputs.num_particles)
        )
        np.testing.assert_allclose(
            values.var(axis=0),
            variance,
            rtol=0,
            atol=6 * variance * math.sqrt(2 / inputs.num_particles),
        )


def test_equilibrium_msd_has_ballistic_and_diffusive_limits():
    inputs = parameters(initial_temperature=3.0, bath_temperature=3.0, mass=2.0, friction=0.7)
    short = 1e-8
    short_msd = 2 * langevin_moments(short, inputs)["position_variance_per_component"]
    assert short_msd == pytest.approx(2 * (3.0 / 2.0) * short**2, rel=1e-8)
    long = 100_000.0
    long_msd = 2 * langevin_moments(long, inputs)["position_variance_per_component"]
    diffusion = 3.0 / (2.0 * 0.7)
    assert long_msd / (4 * long) == pytest.approx(diffusion, rel=2e-5)
    for time in (0.2, 2.0, 10.0):
        exact_msd = 4 * diffusion * (time + math.expm1(-0.7 * time) / 0.7)
        observed = 2 * langevin_moments(time, inputs)["position_variance_per_component"]
        assert observed == pytest.approx(exact_msd, rel=5e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_periodic_wrapping_never_changes_the_unwrapped_trajectory(backend):
    inputs = parameters(
        backend=backend,
        num_particles=32,
        num_steps=15,
        time_step=0.5,
        initial_position=(-0.2, 0.7),
        initial_mean_velocity=(4.0, -3.0),
        acceleration=(3.0, -2.0),
    )
    unbounded = create_langevin_simulator(inputs)
    periodic = create_langevin_simulator(replace(inputs, boundary="periodic", box_size=(0.5, 0.8)))
    for _ in range(inputs.num_steps):
        unbounded.advance()
        periodic.advance()
        np.testing.assert_array_equal(periodic.unwrapped_positions, unbounded.unwrapped_positions)
        np.testing.assert_array_equal(periodic.velocities, unbounded.velocities)
        np.testing.assert_array_equal(
            periodic.positions, np.remainder(periodic.unwrapped_positions, [0.5, 0.8])
        )
        assert (periodic.positions >= 0).all()
        assert (periodic.positions < [0.5, 0.8]).all()
    assert np.max(np.abs(periodic.unwrapped_positions)) > 5


def test_periodic_presentation_handles_rounding_to_box_length():
    inputs = parameters(
        initial_temperature=1e-6,
        bath_temperature=1e-6,
        mass=1e3,
        time_step=1e-6,
        num_steps=1,
        num_particles=1024,
        boundary="periodic",
        box_size=(1e6, 1e6),
    )
    simulator = create_langevin_simulator(inputs)
    simulator.advance()
    original = simulator.unwrapped_positions.copy()
    # This valid low-temperature/short-step case contains negative displacements
    # so small that the raw floating-point remainder rounds up to the box size.
    raw = np.remainder(original, inputs.box_size)
    assert (raw >= inputs.box_size).any()
    wrapped = simulator.positions
    assert (wrapped >= 0).all()
    assert (wrapped < inputs.box_size).all()
    assert (wrapped[raw >= inputs.box_size] == 0).all()
    np.testing.assert_array_equal(simulator.unwrapped_positions, original)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 42, 2**64 - 1])
def test_seed_repeatability_global_rng_isolation_and_sampling_invariance(backend, seed):
    inputs = parameters(backend=backend, seed=seed, num_steps=40, sample_stride=1)
    first = create_langevin_simulator(inputs)
    second = create_langevin_simulator(replace(inputs, sample_stride=7))
    global_before = np.random.get_state(legacy=True)
    for _ in range(inputs.num_steps):
        first.advance()
        second.advance()
        np.testing.assert_array_equal(first.unwrapped_positions, second.unwrapped_positions)
        np.testing.assert_array_equal(first.velocities, second.velocities)
    global_after = np.random.get_state(legacy=True)
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
)
@pytest.mark.parametrize("friction,dt", [(1e-3, 1e-6), (0.7, 0.2), (1e3, 10.0)])
def test_numba_matches_numpy_with_common_joint_innovations(friction, dt):
    inputs = parameters(friction=friction, time_step=dt, acceleration=(0.5, -0.3))
    first = create_langevin_simulator(inputs)
    second = create_langevin_simulator(replace(inputs, backend="numba"))
    for _ in range(inputs.num_steps):
        first.advance()
        second.advance()
        np.testing.assert_allclose(
            first.unwrapped_positions, second.unwrapped_positions, rtol=1e-11
        )
        np.testing.assert_allclose(first.velocities, second.velocities, rtol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_noise_allocation_is_one_step_and_engine_never_allocates_trajectory(backend):
    simulator = create_langevin_simulator(parameters(backend=backend, num_steps=10))
    original = simulator.rng
    previous = []
    shapes = []

    class ObservedGenerator:
        def standard_normal(self, shape):
            assert not any(reference() is not None for reference in previous)
            noise = original.standard_normal(shape)
            previous.append(weakref.ref(noise))
            shapes.append(shape)
            return noise

    simulator.rng = ObservedGenerator()
    position_identity = id(simulator.unwrapped_positions)
    velocity_identity = id(simulator.velocities)
    for _ in range(10):
        simulator.advance()
    assert shapes == [(128, 2, 2)] * 10
    assert id(simulator.unwrapped_positions) == position_identity
    assert id(simulator.velocities) == velocity_identity
    assert not any(reference() is not None for reference in previous)


def test_default_path_keeps_optional_native_libraries_unimported():
    script = """
import json, sys
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.models.langevin_engine import create_langevin_simulator
engine=create_langevin_simulator(LangevinParameters(1.0,1.0,3,num_steps=1))
engine.advance()
print(json.dumps({'backend':engine.execution_metadata()['backend'],
 'imports':sorted(set(sys.modules)&{'numba','llvmlite','jax','jaxlib'})}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert json.loads(result.stdout) == {"backend": "numpy", "imports": []}


def test_missing_optional_backend_is_explicit(monkeypatch):
    monkeypatch.setattr(computation, "find_spec", lambda name: None)
    with pytest.raises(BackendUnavailable, match=r"scientific-api\[numba\]"):
        create_langevin_simulator(parameters(backend="numba"))
