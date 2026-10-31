"""Allowlisted numerical state snapshots, without storage or transport concerns.

Restoration never advances a model, consumes random innovations or evaluates
initial forces. The checkpoint owner must separately bind the entire admitted
request, completed step and sampled buffers to the snapshot. In particular the
historical OU object stores physical coefficients, but not the original seed
or requested total number of steps.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Any, overload

import numpy as np

from ..domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from ..domain.lennard_jones import (
    LJ_MODEL_VERSION,
    MAX_LJ_PAIR_EVALUATIONS,
    LennardJonesParameters,
)
from ..domain.simulation import MODEL_VERSION, ComputeBackend, SimulationParameters
from .computation import BackendUnavailable, require_backend
from .langevin_engine import LangevinSimulator, exact_transition_coefficients
from .lennard_jones_engine import LennardJonesSimulator
from .lj_forces import MAX_NEIGHBOR_PAIRS, ForceResult, VerletList, _validate_pairs
from .physics_engine import IdealGasSimulator

STATE_SCHEMA_VERSION = 1
SimulatorState = IdealGasSimulator | LangevinSimulator | LennardJonesSimulator
Parameters = SimulationParameters | LangevinParameters | LennardJonesParameters


class SimulatorStateError(ValueError):
    """The supplied state is not a complete supported numerical snapshot."""


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SimulatorStateError(f"Invalid {label} fields")
    return value


def _integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SimulatorStateError(f"Invalid {label}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SimulatorStateError(f"Invalid {label}")
    return float(value)


def _same_parameters(recorded: Any, expected: Any) -> bool:
    """Accept JSON arrays for tuples, retaining strict integer/boolean types."""
    if isinstance(expected, dict):
        return (
            isinstance(recorded, dict)
            and set(recorded) == set(expected)
            and all(_same_parameters(recorded[name], value) for name, value in expected.items())
        )
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(recorded, (tuple, list))
            and len(recorded) == len(expected)
            and all(_same_parameters(left, right) for left, right in zip(recorded, expected))
        )
    if type(expected) is int:
        return type(recorded) is int and recorded == expected
    if isinstance(expected, float):
        return (
            not isinstance(recorded, bool)
            and isinstance(recorded, (int, float))
            and math.isfinite(recorded)
            and recorded == expected
        )
    return type(recorded) is type(expected) and recorded == expected


def _array(value: Any, shape: tuple[int, ...], dtype: np.dtype, label: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise SimulatorStateError(f"Invalid {label} array")
    return value.copy(order="C")


def _rng(state: Any) -> np.random.Generator:
    state = _keys(state, {"bit_generator", "state", "has_uint32", "uinteger"}, "RNG")
    if state["bit_generator"] != "PCG64":
        raise SimulatorStateError("Only PCG64 state is supported")
    core = _keys(state["state"], {"state", "inc"}, "PCG64")
    _integer(core["state"], 0, 2**128 - 1, "PCG64 state")
    increment = _integer(core["inc"], 1, 2**128 - 1, "PCG64 increment")
    if increment % 2 != 1:
        raise SimulatorStateError("PCG64 increment must be odd")
    _integer(state["has_uint32"], 0, 1, "PCG64 cached-value flag")
    _integer(state["uinteger"], 0, 2**32 - 1, "PCG64 cached value")
    generator = np.random.Generator(np.random.PCG64(0))
    generator.bit_generator.state = copy.deepcopy(state)
    return generator


def _ou_class(backend: ComputeBackend) -> type[IdealGasSimulator]:
    require_backend(backend)
    if backend == "numpy":
        return IdealGasSimulator
    try:
        from .numba_engine import NumbaIdealGasSimulator
    except ImportError as error:
        raise BackendUnavailable("Numba backend could not be imported") from error
    return NumbaIdealGasSimulator


def _langevin_class(backend: ComputeBackend) -> type[LangevinSimulator]:
    require_backend(backend)
    if backend == "numpy":
        return LangevinSimulator
    try:
        from .langevin_numba import NumbaLangevinSimulator
    except ImportError as error:
        raise BackendUnavailable("Numba Langevin backend could not be imported") from error
    return NumbaLangevinSimulator


def _physics(parameters: Parameters) -> tuple[str, dict[str, Any]]:
    if isinstance(parameters, LennardJonesParameters):
        return LJ_MODEL_VERSION, asdict(parameters)
    if isinstance(parameters, LangevinParameters):
        return LANGEVIN_MODEL_VERSION, asdict(parameters)
    if isinstance(parameters, SimulationParameters):
        return MODEL_VERSION, {
            "num_particles": parameters.num_particles,
            "initial_temperature": parameters.initial_temperature,
            "time_step": parameters.time_step,
        }
    raise TypeError("Unsupported checkpoint parameters")


def capture_simulator(simulator: object) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Copy only declared numerical fields; never serialize arbitrary objects."""
    if type(simulator) is LennardJonesSimulator:
        model, physics = _physics(simulator.parameters)
        execution = simulator.execution_metadata()
        count = simulator.parameters.num_particles
        arrays: dict[str, np.ndarray] = {
            "velocities": _array(
                simulator.velocities, (count, 2), np.dtype("float64"), "velocities"
            ),
            "unwrapped_positions": _array(
                simulator.unwrapped_positions, (count, 2), np.dtype("float64"), "positions"
            ),
            "forces": _array(simulator.forces, (count, 2), np.dtype("float64"), "forces"),
        }
        neighbor = simulator._neighbors
        engine_state = {
            "steps": simulator._steps,
            "pair_evaluations": simulator._pair_evaluations,
            "analysis_pair_evaluations": simulator._analysis_pair_evaluations,
            "force_result": {
                "potential_energy": simulator._force_result.potential_energy,
                "virial": simulator._force_result.virial,
                "pair_evaluations": simulator._force_result.pair_evaluations,
            },
            "neighbors": None,
        }
        if neighbor is not None:
            if neighbor._reference is None or neighbor._pairs is None:
                raise SimulatorStateError("LJ neighbor cache is incomplete")
            _validate_pairs(neighbor._pairs, count)
            arrays.update(
                neighbor_reference=_array(
                    neighbor._reference, (count, 2), np.dtype("float64"), "neighbor reference"
                ),
                neighbor_pairs=neighbor._pairs.copy(order="C"),
            )
            engine_state["neighbors"] = {
                "max_pairs": neighbor.max_pairs,
                "max_candidate_checks": neighbor.max_candidate_checks,
                "rebuild_count": neighbor.rebuild_count,
                "candidate_checks": neighbor.candidate_checks,
                "last_candidate_checks": neighbor.last_candidate_checks,
            }
    elif isinstance(simulator, LangevinSimulator):
        model, physics = _physics(simulator.parameters)
        execution = simulator.execution_metadata()
        if type(simulator) is not _langevin_class(execution["backend"]):
            raise TypeError("Unsupported Langevin simulator implementation")
        count = simulator.parameters.num_particles
        arrays = {
            "velocities": _array(
                simulator.velocities, (count, 2), np.dtype("float64"), "velocities"
            ),
            "unwrapped_positions": _array(
                simulator.unwrapped_positions, (count, 2), np.dtype("float64"), "positions"
            ),
        }
        engine_state = None
    elif isinstance(simulator, IdealGasSimulator):
        model = MODEL_VERSION
        execution = simulator.execution_metadata()
        if type(simulator) is not _ou_class(execution["backend"]):
            raise TypeError("Unsupported OU simulator implementation")
        physics = {
            "num_particles": simulator.num_particles,
            "initial_temperature": simulator.target_temperature,
            "time_step": simulator.dt,
        }
        arrays = {
            "velocities": _array(
                simulator.velocities,
                (simulator.num_particles, 2),
                np.dtype("float64"),
                "velocities",
            )
        }
        engine_state = None
    else:
        raise TypeError("Unsupported simulator implementation")
    metadata = {
        "schema_version": STATE_SCHEMA_VERSION,
        "model_version": model,
        "backend": execution["backend"],
        "physics_parameters": copy.deepcopy(physics),
        "rng_state": copy.deepcopy(simulator.rng.bit_generator.state),
        "execution": execution,
    }
    if engine_state is not None:
        metadata["engine_state"] = engine_state
    return metadata, arrays


def _restore_lj(
    parameters: LennardJonesParameters, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> LennardJonesSimulator:
    count = parameters.num_particles
    state = _keys(
        metadata["engine_state"],
        {"steps", "pair_evaluations", "analysis_pair_evaluations", "force_result", "neighbors"},
        "LJ engine",
    )
    steps = _integer(state["steps"], 0, parameters.num_steps, "LJ completed steps")
    force_checks = _integer(state["pair_evaluations"], 0, MAX_LJ_PAIR_EVALUATIONS, "force work")
    analysis_checks = _integer(
        state["analysis_pair_evaluations"], 0, MAX_LJ_PAIR_EVALUATIONS, "analysis work"
    )
    force = _keys(
        state["force_result"], {"potential_energy", "virial", "pair_evaluations"}, "force"
    )
    potential, virial = (
        _finite(force["potential_energy"], "potential"),
        _finite(force["virial"], "virial"),
    )
    last_force_checks = _integer(
        force["pair_evaluations"], 0, count * (count - 1) // 2, "cached force work"
    )
    simulator = LennardJonesSimulator.__new__(LennardJonesSimulator)
    simulator.parameters = parameters
    simulator.velocities = _array(
        arrays["velocities"], (count, 2), np.dtype("float64"), "velocities"
    )
    simulator.unwrapped_positions = _array(
        arrays["unwrapped_positions"], (count, 2), np.dtype("float64"), "positions"
    )
    forces = _array(arrays["forces"], (count, 2), np.dtype("float64"), "forces")
    neighbor_checks = 0
    simulator._neighbors = None
    if parameters.interaction_backend == "verlet":
        neighbor = _keys(
            state["neighbors"],
            {
                "max_pairs",
                "max_candidate_checks",
                "rebuild_count",
                "candidate_checks",
                "last_candidate_checks",
            },
            "neighbor cache",
        )
        maximum_pairs = _integer(neighbor["max_pairs"], 1, MAX_NEIGHBOR_PAIRS, "neighbor capacity")
        maximum_checks = _integer(
            neighbor["max_candidate_checks"], 0, MAX_LJ_PAIR_EVALUATIONS, "neighbor work limit"
        )
        neighbor_checks = _integer(neighbor["candidate_checks"], 0, maximum_checks, "neighbor work")
        rebuilds = _integer(neighbor["rebuild_count"], 1, steps + 1, "neighbor rebuild count")
        last_checks = _integer(
            neighbor["last_candidate_checks"], 0, neighbor_checks, "last neighbor work"
        )
        pairs = arrays["neighbor_pairs"]
        try:
            _validate_pairs(pairs, count)
        except ValueError as error:
            raise SimulatorStateError("Invalid neighbor pair array") from error
        if (
            not pairs.flags.c_contiguous
            or len(pairs) > maximum_pairs
            or len(pairs) != last_force_checks
        ):
            raise SimulatorStateError("Neighbor pair cache disagrees with cached force work")
        reference = _array(
            arrays["neighbor_reference"], (count, 2), np.dtype("float64"), "neighbor reference"
        )
        with np.errstate(over="raise", invalid="raise"):
            delta = simulator.unwrapped_positions - reference
            displacement = float(np.max(np.einsum("ij,ij->i", delta, delta)))
        if displacement >= (parameters.neighbor_skin / 2) ** 2:
            raise SimulatorStateError("Neighbor reference is not current at this step boundary")
        restored = VerletList(
            parameters.box_size,
            parameters.cutoff_radius,
            parameters.neighbor_skin,
            max_pairs=maximum_pairs,
            max_candidate_checks=maximum_checks,
        )
        restored._pairs = pairs.copy(order="C")
        restored._pairs.setflags(write=False)
        restored._reference = reference
        restored.rebuild_count = rebuilds
        restored.candidate_checks = neighbor_checks
        restored.last_candidate_checks = last_checks
        simulator._neighbors = restored
    elif state["neighbors"] is not None or last_force_checks != count * (count - 1) // 2:
        raise SimulatorStateError("Direct interactions cannot restore a neighbor cache")
    if (
        force_checks < last_force_checks
        or force_checks + analysis_checks + neighbor_checks > MAX_LJ_PAIR_EVALUATIONS
    ):
        raise SimulatorStateError("LJ work counters violate the shared budget")
    if (
        parameters.interaction_backend == "direct"
        and force_checks != (steps + 1) * last_force_checks
    ):
        raise SimulatorStateError("Direct force work does not match the completed step count")
    simulator._force_result = ForceResult(forces, potential, virial, last_force_checks)
    simulator._pair_evaluations = force_checks
    simulator._analysis_pair_evaluations = analysis_checks
    simulator._steps = steps
    simulator.rng = _rng(metadata["rng_state"])
    if parameters.ensemble == "nvt":
        assert parameters.friction is not None and parameters.bath_temperature is not None
        simulator._decay = math.exp(-parameters.friction * parameters.time_step)
        simulator._noise_scale = math.sqrt(
            parameters.bath_temperature
            / parameters.mass
            * -math.expm1(-2 * parameters.friction * parameters.time_step)
        )
    return simulator


@overload
def restore_simulator(
    parameters: SimulationParameters, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> IdealGasSimulator: ...
@overload
def restore_simulator(
    parameters: LangevinParameters, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> LangevinSimulator: ...
@overload
def restore_simulator(
    parameters: LennardJonesParameters, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> LennardJonesSimulator: ...


def restore_simulator(
    parameters: Parameters, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> SimulatorState:
    """Restore a declared backend without initialization draws or force work."""
    model, physics = _physics(parameters)
    expected = {
        "schema_version",
        "model_version",
        "backend",
        "physics_parameters",
        "rng_state",
        "execution",
    }
    if isinstance(parameters, LennardJonesParameters):
        expected.add("engine_state")
    _keys(metadata, expected, "simulator state")
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != STATE_SCHEMA_VERSION
        or metadata["model_version"] != model
        or metadata["backend"] != parameters.backend
        or not _same_parameters(metadata["physics_parameters"], physics)
    ):
        raise SimulatorStateError("Snapshot does not match the physical model and parameters")
    expected_arrays = {"velocities"}
    if not isinstance(parameters, SimulationParameters):
        expected_arrays.add("unwrapped_positions")
    if isinstance(parameters, LennardJonesParameters):
        expected_arrays.add("forces")
        if parameters.interaction_backend == "verlet":
            expected_arrays.update({"neighbor_reference", "neighbor_pairs"})
    _keys(arrays, expected_arrays, "simulator array")
    require_backend(parameters.backend)
    if isinstance(parameters, LennardJonesParameters):
        simulator: SimulatorState = _restore_lj(parameters, metadata, arrays)
    elif isinstance(parameters, LangevinParameters):
        implementation = _langevin_class(parameters.backend)
        simulator = implementation.__new__(implementation)
        simulator.parameters = parameters
        simulator.rng = _rng(metadata["rng_state"])
        simulator.velocities = _array(
            arrays["velocities"], (parameters.num_particles, 2), np.dtype("float64"), "velocities"
        )
        simulator.unwrapped_positions = _array(
            arrays["unwrapped_positions"],
            (parameters.num_particles, 2),
            np.dtype("float64"),
            "positions",
        )
        simulator._acceleration = np.asarray(parameters.acceleration, dtype=np.float64)
        simulator._coefficients = exact_transition_coefficients(
            parameters.time_step, parameters.friction, parameters.bath_temperature / parameters.mass
        )
    else:
        ou_implementation = _ou_class(parameters.backend)
        simulator = ou_implementation.__new__(ou_implementation)
        simulator.num_particles = parameters.num_particles
        simulator.dt = parameters.time_step
        simulator.target_temperature = parameters.initial_temperature
        simulator.rng = _rng(metadata["rng_state"])
        simulator.velocities = _array(
            arrays["velocities"], (parameters.num_particles, 2), np.dtype("float64"), "velocities"
        )
        simulator._decay = math.exp(-parameters.time_step)
        simulator._noise_scale = math.sqrt(
            parameters.initial_temperature * -math.expm1(-2.0 * parameters.time_step)
        )
    if not _same_parameters(metadata["execution"], simulator.execution_metadata()):
        raise SimulatorStateError("Snapshot execution semantics differ from this backend")
    return simulator
