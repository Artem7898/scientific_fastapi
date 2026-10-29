"""Runtime backends preserve the OU model, RNG contract and bounded state."""

import importlib.util
import json
import math
import subprocess
import sys
import weakref

import numpy as np
import pytest

from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models import computation
from scientific_api.models.computation import BackendUnavailable, create_simulator, require_backend
from scientific_api.models.physics_engine import IdealGasSimulator

BACKENDS = [
    "numpy",
    pytest.param(
        "numba",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
        ),
    ),
]


def test_numpy_path_does_not_import_optional_native_backends() -> None:
    script = """
import json
import sys
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models.computation import create_simulator, require_backend
require_backend('numpy')
simulator = create_simulator(SimulationParameters(1.0, 16, num_steps=3))
for _ in range(3):
    simulator.advance()
print(json.dumps({'imports': sorted(set(sys.modules) & {'numba', 'llvmlite', 'jax', 'jaxlib'}),
                  'backend': simulator.execution_metadata()['backend']}))
"""
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert json.loads(process.stdout) == {"imports": [], "backend": "numpy"}


def test_missing_numba_is_explicit_and_numpy_remains_available(monkeypatch) -> None:
    monkeypatch.setattr(computation, "find_spec", lambda name: None)
    require_backend("numpy")
    with pytest.raises(BackendUnavailable, match=r"scientific-api\[numba\]"):
        require_backend("numba")
    with pytest.raises(BackendUnavailable):
        create_simulator(SimulationParameters(1.0, 4, backend="numba"))
    simulator = create_simulator(SimulationParameters(1.0, 4))
    assert simulator.execution_metadata()["backend"] == "numpy"


def test_availability_check_never_imports_numba(monkeypatch) -> None:
    # find_spec only checks availability. A missing LLVM runtime is discovered
    # when the worker loads the engine, not by compiling on the HTTP event loop.
    checked = []
    monkeypatch.setattr(computation, "find_spec", lambda name: checked.append(name) or object())
    require_backend("numba")
    assert checked == ["numba"]


def test_worker_import_failure_is_explicit_without_reference_fallback(monkeypatch) -> None:
    monkeypatch.setattr(computation, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "scientific_api.models.numba_engine", None)

    def forbidden_reference(**kwargs):
        raise AssertionError("A requested Numba run must never fall back to NumPy")

    monkeypatch.setattr(computation, "IdealGasSimulator", forbidden_reference)
    with pytest.raises(BackendUnavailable, match="could not be imported"):
        create_simulator(SimulationParameters(1.0, 4, backend="numba"))


@pytest.mark.parametrize("backend", [None, True, 1, [], "", "jax", "os.system"])
def test_domain_rejects_unregistered_backends(backend) -> None:
    with pytest.raises(ValueError, match="backend"):
        SimulationParameters(1.0, 4, backend=backend)


def test_factory_rejects_unvalidated_inputs() -> None:
    with pytest.raises(TypeError, match="SimulationParameters"):
        create_simulator({"num_particles": 4})
    with pytest.raises(ValueError, match="backend"):
        require_backend("jax")


def test_numpy_advance_preserves_pre_backend_reference_exactly() -> None:
    parameters = SimulationParameters(3.0, 137, time_step=0.2, num_steps=25, seed=2**64 - 1)
    simulator = create_simulator(parameters)
    rng = np.random.Generator(np.random.PCG64(parameters.seed))
    reference = rng.normal(0.0, math.sqrt(parameters.initial_temperature), size=(137, 2))
    decay = math.exp(-parameters.time_step)
    amplitude = math.sqrt(parameters.initial_temperature * -math.expm1(-2 * parameters.time_step))
    for _ in range(parameters.num_steps):
        reference *= decay
        reference += amplitude * rng.standard_normal(reference.shape)
        simulator.advance()
        np.testing.assert_array_equal(simulator.velocities, reference)
    energy = float(0.5 * np.einsum("ij,ij->", reference, reference))
    assert simulator.observables() == (energy / parameters.num_particles, energy)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 42, 2**64 - 1])
def test_backend_repeats_locally_without_modifying_global_rng(backend, seed) -> None:
    parameters = SimulationParameters(
        3.0, 137, time_step=0.2, num_steps=25, seed=seed, backend=backend
    )
    first, second = create_simulator(parameters), create_simulator(parameters)
    global_before = np.random.get_state(legacy=True)
    for _ in range(parameters.num_steps):
        first.advance()
        second.advance()
        np.testing.assert_array_equal(first.velocities, second.velocities)
    assert first.observables() == second.observables()
    global_after = np.random.get_state(legacy=True)
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


@pytest.mark.parametrize("dt", [1e-12, 0.001, 0.2, 2.0, 1000.0])
@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
)
def test_numba_matches_numpy_entire_trajectory_with_identical_pcg64_innovations(dt) -> None:
    numpy_simulator = create_simulator(
        SimulationParameters(3.0, 257, time_step=dt, num_steps=40, seed=42)
    )
    numba_simulator = create_simulator(
        SimulationParameters(3.0, 257, time_step=dt, num_steps=40, seed=42, backend="numba")
    )
    for _ in range(40):
        numpy_simulator.advance()
        numba_simulator.advance()
        np.testing.assert_allclose(
            numba_simulator.velocities, numpy_simulator.velocities, rtol=1e-11, atol=1e-11
        )
    np.testing.assert_allclose(
        numba_simulator.observables(), numpy_simulator.observables(), rtol=1e-11, atol=1e-11
    )
    assert numba_simulator.rng.bit_generator.state == numpy_simulator.rng.bit_generator.state


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
)
def test_numba_arithmetic_matches_reference_for_fixed_noise_without_using_an_rng() -> None:
    from scientific_api.models.numba_engine import update_velocities as numba_update
    from scientific_api.models.physics_engine import update_velocities as numpy_update

    numpy_state = np.array([[1.0, -2.0], [-0.5, 4.0]], dtype=np.float64)
    numba_state = numpy_state.copy()
    shocks = np.array([[0.75, -1.0], [0.0, 2.0]], dtype=np.float64)
    for _ in range(8):
        numpy_update(numpy_state, shocks, 0.91, 0.17)
        numba_update(numba_state, shocks, 0.91, 0.17)
    np.testing.assert_allclose(numba_state, numpy_state, rtol=1e-11, atol=1e-11)
    assert numba_update.targetoptions["nopython"] is True
    assert numba_update.targetoptions["nogil"] is True
    assert numba_update.targetoptions["fastmath"] is False
    assert numba_update.targetoptions.get("parallel", False) is False


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dt", [0.001, 0.2, 2.0, 1000.0])
def test_backends_preserve_equilibrium_and_exact_conditional_moments(backend, dt) -> None:
    particles, temperature = 40_000, 3.0
    simulator = create_simulator(
        SimulationParameters(temperature, particles, time_step=dt, seed=2026, backend=backend)
    )
    for _ in range(10):
        simulator.advance()
    observed_temperature, energy = simulator.observables()
    assert abs(observed_temperature - temperature) < 6 * temperature / math.sqrt(particles)
    assert energy == observed_temperature * particles
    simulator.velocities.fill(2.0)
    simulator.advance()
    expected_mean = 2.0 * math.exp(-dt)
    expected_variance = temperature * -math.expm1(-2.0 * dt)
    observations = 2 * particles
    assert abs(float(simulator.velocities.mean()) - expected_mean) < 6 * math.sqrt(
        expected_variance / observations
    )
    assert abs(float(simulator.velocities.var()) - expected_variance) < (
        6 * expected_variance * math.sqrt(2 / (observations - 1))
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_retains_only_current_state_and_one_step_of_noise(backend) -> None:
    simulator = create_simulator(SimulationParameters(1.0, 64, num_steps=40, backend=backend))
    noise_refs = []

    class CheckedRNG:
        def standard_normal(self, shape):
            assert shape == (64, 2)
            assert all(reference() is None for reference in noise_refs)
            noise = np.full(shape, 0.5, dtype=np.float64)
            noise_refs.append(weakref.ref(noise))
            return noise

    simulator.rng = CheckedRNG()
    state = simulator.velocities
    for _ in range(40):
        simulator.advance()
        assert simulator.velocities is state
    assert len(noise_refs) == 40
    assert all(reference() is None for reference in noise_refs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_state_is_never_reported_as_a_successful_observation(backend, nonfinite) -> None:
    simulator = create_simulator(SimulationParameters(1.0, 4, backend=backend))
    simulator.velocities[0, 0] = nonfinite
    with pytest.raises(FloatingPointError):
        simulator.step()


@pytest.mark.parametrize("backend", BACKENDS)
def test_execution_metadata_identifies_implementation_without_changing_model(backend) -> None:
    simulator = create_simulator(SimulationParameters(1.0, 4, backend=backend))
    metadata = simulator.execution_metadata()
    assert metadata["backend"] == backend
    assert metadata["backend_version"]
    assert metadata["kernel_version"] == f"ou-{backend}-v1"
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float64"
    assert metadata["fastmath"] is False
    assert metadata["parallel"] is False
    assert metadata["rng_policy"] == "numpy-pcg64-per-step-v1"
    assert metadata["observables_schedule"] == "terminal"
    if backend == "numba":
        assert metadata["nogil"] is True
        assert metadata["cache"] is False
        assert metadata["llvmlite_version"]
        assert metadata["llvm_version"]


def test_reference_step_keeps_the_existing_observables_contract() -> None:
    simulator = IdealGasSimulator(4, 1.0, 0.2, seed=42)
    assert simulator.step() == simulator.observables()
