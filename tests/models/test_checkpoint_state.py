"""Continuation reproduces numerical state rather than rerunning initialization."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import MAX_LJ_PAIR_EVALUATIONS, LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models.checkpoint_state import (
    SimulatorStateError,
    capture_simulator,
    restore_simulator,
)
from scientific_api.models.computation import create_simulator
from scientific_api.models.langevin_engine import create_langevin_simulator
from scientific_api.models.lennard_jones_engine import LennardJonesSimulator, create_lj_simulator
from scientific_api.models.lj_forces import PairBudgetExceeded, VerletList

BACKENDS = [
    "numpy",
    pytest.param(
        "numba",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("numba") is None, reason="Numba extra is not installed"
        ),
    ),
]


def parameters(model, backend="numpy", *, ensemble="nve", interaction="verlet"):
    if model == "ou":
        return SimulationParameters(
            1.2, 17, time_step=0.03, num_steps=80, seed=2**64 - 1, backend=backend
        )
    if model == "langevin":
        return LangevinParameters(
            0.5,
            1.4,
            17,
            time_step=0.1,
            num_steps=80,
            seed=2**64 - 1,
            backend=backend,
            acceleration=(0.2, -0.1),
            initial_mean_velocity=(0.2, -0.3),
            initial_position=(5.9, 0.01),
            boundary="periodic",
            box_size=(6.0, 8.0),
        )
    return LennardJonesParameters(
        num_particles=16,
        box_size=(6.0, 6.0),
        initial_temperature=0.5,
        time_step=0.008,
        num_steps=80,
        seed=2**64 - 1,
        backend=backend,
        ensemble=ensemble,
        interaction_backend=interaction,
        bath_temperature=0.8 if ensemble == "nvt" else None,
    )


def create(configuration):
    if isinstance(configuration, LennardJonesParameters):
        return create_lj_simulator(configuration)
    if isinstance(configuration, LangevinParameters):
        return create_langevin_simulator(configuration)
    return create_simulator(configuration)


def assert_state_equal(first, second):
    for field in ("velocities", "unwrapped_positions", "positions", "forces"):
        if hasattr(first, field):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    assert first.execution_metadata() == second.execution_metadata()
    if isinstance(first, LennardJonesSimulator):
        assert first.observables() == second.observables()
        assert first.diagnostics() == second.diagnostics()
        assert first._steps == second._steps
        if first._neighbors is not None:
            assert not second._neighbors._pairs.flags.writeable
            np.testing.assert_array_equal(first._neighbors._pairs, second._neighbors._pairs)
            np.testing.assert_array_equal(first._neighbors._reference, second._neighbors._reference)
            assert first._neighbors.last_candidate_checks == second._neighbors.last_candidate_checks


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", ["ou", "langevin"])
@pytest.mark.parametrize("split", [0, 7, 29])
def test_seeded_cpu_continuation_is_bitwise_equal_including_future_rng(model, backend, split):
    configuration = parameters(model, backend)
    uninterrupted = create(configuration)
    for _ in range(split):
        uninterrupted.advance()
    metadata, arrays = capture_simulator(uninterrupted)
    # The wire representation of tuple parameters must remain restorable.
    restored = restore_simulator(configuration, json.loads(json.dumps(metadata)), arrays)
    assert_state_equal(uninterrupted, restored)
    for _ in range(split, configuration.num_steps):
        uninterrupted.advance()
        restored.advance()
    assert_state_equal(uninterrupted, restored)
    np.testing.assert_array_equal(
        uninterrupted.rng.standard_normal((100, 2)), restored.rng.standard_normal((100, 2))
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
@pytest.mark.parametrize("interaction", ["direct", "verlet"])
@pytest.mark.parametrize("split", [0, 7, 29])
def test_lj_continuation_preserves_force_neighbor_caches_and_work_counters(
    backend, ensemble, interaction, split
):
    configuration = parameters("lj", backend, ensemble=ensemble, interaction=interaction)
    uninterrupted = create(configuration)
    for _ in range(split):
        uninterrupted.advance()
    uninterrupted.reserve_analysis_pairs(17)
    metadata, arrays = capture_simulator(uninterrupted)
    restored = restore_simulator(configuration, json.loads(json.dumps(metadata)), arrays)
    assert_state_equal(uninterrupted, restored)
    for _ in range(split, configuration.num_steps):
        uninterrupted.advance()
        restored.advance()
        assert_state_equal(uninterrupted, restored)
    if interaction == "verlet":
        assert uninterrupted.diagnostics()["neighbor_rebuilds"] > 1


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
def test_capture_and_restore_do_not_alias_live_state(model):
    configuration = parameters(model)
    live = create(configuration)
    metadata, arrays = capture_simulator(live)
    original = live.velocities.copy()
    arrays["velocities"][0, 0] += 1.0
    np.testing.assert_array_equal(live.velocities, original)
    restored = restore_simulator(configuration, metadata, arrays)
    copied = restored.velocities.copy()
    arrays["velocities"][0, 0] += 1.0
    metadata["rng_state"]["state"]["state"] = 0
    np.testing.assert_array_equal(restored.velocities, copied)
    assert restored.rng.bit_generator.state != metadata["rng_state"]


def test_restore_does_not_initialize_positions_velocities_forces_or_neighbors(monkeypatch):
    configuration = parameters("lj", ensemble="nvt")
    live = create(configuration)
    for _ in range(7):
        live.advance()
    metadata, arrays = capture_simulator(live)

    def forbidden(*args, **kwargs):
        pytest.fail("Restore must not run initialization or rebuild force/neighbor work")

    monkeypatch.setattr(LennardJonesSimulator, "__init__", forbidden)
    monkeypatch.setattr(LennardJonesSimulator, "_evaluate_forces", forbidden)
    monkeypatch.setattr(VerletList, "pairs", forbidden)
    restored = restore_simulator(configuration, metadata, arrays)
    assert_state_equal(live, restored)


def test_pcg64_cached_uint32_and_128_bit_state_are_preserved():
    configuration = parameters("ou")
    live = create(configuration)
    live.rng.integers(0, 2**32, dtype=np.uint32)
    metadata, arrays = capture_simulator(live)
    assert metadata["rng_state"]["has_uint32"] == 1
    restored = restore_simulator(configuration, json.loads(json.dumps(metadata)), arrays)
    np.testing.assert_array_equal(
        live.rng.integers(0, 2**32, dtype=np.uint32, size=23),
        restored.rng.integers(0, 2**32, dtype=np.uint32, size=23),
    )


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
@pytest.mark.parametrize("fault", ["nan", "float32", "shape", "object", "missing", "extra"])
def test_corrupt_array_snapshots_are_rejected_without_mutating_inputs(model, fault):
    configuration = parameters(model)
    metadata, arrays = capture_simulator(create(configuration))
    if fault == "nan":
        arrays["velocities"][0, 0] = np.nan
    elif fault == "float32":
        arrays["velocities"] = arrays["velocities"].astype(np.float32)
    elif fault == "shape":
        arrays["velocities"] = arrays["velocities"][:-1]
    elif fault == "object":
        arrays["velocities"] = arrays["velocities"].astype(object)
    elif fault == "missing":
        arrays.pop("velocities")
    else:
        arrays["arbitrary_attribute"] = np.zeros(1)
    with pytest.raises(SimulatorStateError):
        restore_simulator(configuration, metadata, arrays)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("backend",), "other"),
        (("model_version",), "unknown-v1"),
        (("physics_parameters", "num_particles"), True),
        (("rng_state", "bit_generator"), "PCG64DXSM"),
        (("rng_state", "state", "inc"), 2),
        (("rng_state", "state", "state"), 2**128),
        (("rng_state", "has_uint32"), True),
        (("rng_state", "uinteger"), -1),
        (("execution", "kernel_version"), "future-kernel"),
    ],
)
def test_corrupt_metadata_and_execution_semantics_are_rejected(path, value):
    configuration = parameters("ou")
    metadata, arrays = capture_simulator(create(configuration))
    target = metadata
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(SimulatorStateError):
        restore_simulator(configuration, metadata, arrays)


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
def test_changed_physics_backend_or_integrator_is_not_a_continuation(model):
    configuration = parameters(model)
    metadata, arrays = capture_simulator(create(configuration))
    with pytest.raises(SimulatorStateError):
        restore_simulator(
            replace(configuration, time_step=configuration.time_step / 2), metadata, arrays
        )
    if model == "lj":
        with pytest.raises(SimulatorStateError):
            restore_simulator(
                replace(configuration, interaction_backend="direct"), metadata, arrays
            )


@pytest.mark.parametrize(
    "fault", ["negative", "budget", "force_count", "steps", "rebuilds", "reference", "pairs"]
)
def test_lj_cache_and_budget_corruption_cannot_reset_or_escape_work(fault):
    configuration = parameters("lj")
    live = create(configuration)
    live.advance()
    metadata, arrays = capture_simulator(live)
    state = metadata["engine_state"]
    if fault == "negative":
        state["analysis_pair_evaluations"] = -1
    elif fault == "budget":
        state["analysis_pair_evaluations"] = MAX_LJ_PAIR_EVALUATIONS
    elif fault == "force_count":
        state["force_result"]["pair_evaluations"] += 1
    elif fault == "steps":
        state["steps"] = True
    elif fault == "rebuilds":
        state["neighbors"]["rebuild_count"] = 0
    elif fault == "reference":
        arrays["neighbor_reference"][0, 0] += 1.0
    else:
        arrays["neighbor_pairs"][0] = arrays["neighbor_pairs"][1]
    with pytest.raises(SimulatorStateError):
        restore_simulator(configuration, metadata, arrays)


def test_restored_lj_budget_exhausts_at_same_boundary():
    configuration = parameters("lj", interaction="direct")
    live = create(configuration)
    for _ in range(3):
        live.advance()
    live.reserve_analysis_pairs(MAX_LJ_PAIR_EVALUATIONS - live.diagnostics()["total_pair_checks"])
    metadata, arrays = capture_simulator(live)
    restored = restore_simulator(configuration, metadata, arrays)
    assert live.diagnostics() == restored.diagnostics()
    with pytest.raises(PairBudgetExceeded):
        live.advance()
    with pytest.raises(PairBudgetExceeded):
        restored.advance()
    assert_state_equal(live, restored)


def test_numpy_snapshot_path_does_not_import_optional_backends():
    script = """
import sys
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models.computation import create_simulator
from scientific_api.models.checkpoint_state import capture_simulator, restore_simulator
p=SimulationParameters(1.,8,num_steps=3)
s=create_simulator(p); s.advance()
m,a=capture_simulator(s); r=restore_simulator(p,m,a); r.advance()
assert not set(sys.modules)&{'numba','llvmlite','jax','jaxlib'}
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_capture_rejects_nonfinite_live_arrays_and_unknown_classes():
    simulator = create(parameters("ou"))
    simulator.velocities[0, 0] = np.nan
    with pytest.raises(SimulatorStateError):
        capture_simulator(simulator)
    with pytest.raises(TypeError):
        capture_simulator(object())
