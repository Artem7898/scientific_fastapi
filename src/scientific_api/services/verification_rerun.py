"""Bounded same-seed observations in verifier-owned temporary storage.

Equality is about canonical scientific values observed in this invocation. It
does not establish equilibrium, model validity, authorship, statistical coverage
or a general guarantee of reproducibility on a different numerical platform.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

import numpy as np

from ..domain.checkpoint import checkpoint_model_version, checkpoint_parameters_sha256
from ..domain.experiments import ExperimentParameters, parameters_from_payload
from ..domain.langevin import LangevinParameters
from ..domain.lennard_jones import LennardJonesParameters
from ..domain.simulation import SimulationCancelled
from ..models.computation import BackendUnavailable, require_backend
from ..schemas.result_envelope import ResultEnvelope
from .checkpoint_service import checkpoint_fingerprint
from .result_envelope_service import MAX_JSON_BYTES, _json, _root
from .simulation_service import run_simulation


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _check(name: str, status: str, reason: str, **values: Any) -> dict[str, Any]:
    return {"name": "rerun_" + name, "status": status, "reason": reason, **values}


def _cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SimulationCancelled("Result verification interrupted")


class _CancellationProbe(Event):
    """Use the existing per-transition cancellation boundary of every model."""

    def __init__(self, callback: Callable[[], bool] | None):
        super().__init__()
        self.callback = callback

    def is_set(self) -> bool:
        return super().is_set() or (self.callback is not None and self.callback())


def _contracts(
    parameters: ExperimentParameters,
) -> dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]]:
    n = parameters.num_particles
    state: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        "velocities": ((n, 2), np.dtype(np.float64))
    }
    result: dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]] = {"final_state": state}
    if isinstance(parameters, (LangevinParameters, LennardJonesParameters)):
        state.update(
            positions=((n, 2), np.dtype(np.float64)),
            unwrapped_positions=((n, 2), np.dtype(np.float64)),
        )
        frames = len(parameters.sampling_steps())
        result["trajectory"] = {
            "positions": ((frames, n, 2), np.dtype(np.float64)),
            "sample_steps": ((frames,), np.dtype(np.int64)),
            "times": ((frames,), np.dtype(np.float64)),
        }
    return result


def _array_digest(array: np.ndarray, cancel: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical({"shape": list(array.shape), "dtype": array.dtype.str}))
    raw = memoryview(array).cast("B")
    for offset in range(0, len(raw), 1024 * 1024):
        cancel()
        digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _current_environment(parameters: ExperimentParameters) -> dict[str, Any]:
    require_backend(parameters.backend)
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "backend": parameters.backend,
        "backend_version": np.__version__,
        "device": "cpu",
        "dtype": "float64",
        "fastmath": False,
        "parallel": False,
        "bit_generator": "PCG64",
    }
    if parameters.backend == "numba":
        binding = importlib.import_module("llvmlite.binding")
        result.update(
            backend_version=importlib.metadata.version("numba"),
            llvmlite_version=importlib.metadata.version("llvmlite"),
            llvm_version=".".join(str(value) for value in binding.llvm_version_info),
        )
    return result


def _environment_check(
    parameters: ExperimentParameters, envelope: ResultEnvelope, current: dict[str, Any]
) -> dict[str, Any]:
    recorded = {
        **{
            name: envelope.environment.get(name)
            for name in (
                "python",
                "python_implementation",
                "numpy",
                "system",
                "machine",
                "processor",
            )
        },
        **{
            name: envelope.execution.get(name)
            for name in (
                "backend",
                "backend_version",
                "device",
                "dtype",
                "fastmath",
                "parallel",
                "llvmlite_version",
                "llvm_version",
            )
        },
        "bit_generator": envelope.rng.get("bit_generator"),
    }
    # Missing historical CPU metadata is explicitly unknown, not evidence of a
    # mismatch. A matching architecture alone is not a matching processor.
    mismatches = [
        name
        for name, value in recorded.items()
        if value not in (None, "") and value != current.get(name)
    ]
    fingerprint = (envelope.continuation or {}).get("fingerprint_sha256")
    observed_fingerprint = None
    if fingerprint is not None:
        observed_fingerprint = hashlib.sha256(
            _canonical(checkpoint_fingerprint(parameters))
        ).hexdigest()
        if fingerprint != observed_fingerprint:
            mismatches.append("numerical_fingerprint")
    return _check(
        "environment",
        "inconclusive" if mismatches else "passed",
        "recorded_numerical_environment_differs"
        if mismatches
        else "recorded_numerical_metadata_matches_current_runtime",
        details={
            "recorded": recorded,
            "current": current,
            "mismatched_fields": mismatches,
            "recorded_numerical_fingerprint": fingerprint,
            "current_numerical_fingerprint": observed_fingerprint,
            "cpu_identity_recorded": bool(recorded["processor"]),
            "scope": "Known metadata comparison; missing metadata does not establish platform equivalence",
            "repeatability_guaranteed": False,
        },
    )


def _scientific_scalars(parameters: ExperimentParameters, result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parameters, LennardJonesParameters):
        names = (
            "thermal_temperature",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
            "pressure",
        )
    elif isinstance(parameters, LangevinParameters):
        names = (
            "thermal_temperature",
            "thermal_temperature_standard_error",
            "kinetic_energy",
            "mean_velocity",
            "mean_displacement",
            "final_msd",
            "final_centered_msd",
            "diffusion",
        )
    else:
        names = ("final_temperature", "total_energy", "temperature_standard_error")
    return {name: result[name] for name in names}


def verify_same_seed_rerun(
    parameters: ExperimentParameters,
    source_envelope: ResultEnvelope,
    artifacts: dict[str, bytes],
    *,
    temporary_root: Path,
    cancelled: Callable[[], bool] | None = None,
    max_particle_steps: int = 1_000_000,
    max_lj_pair_checks: int = 5_000_000,
) -> list[dict[str, Any]]:
    """Compare one same-backend seed realization without reopening source files.

    Source bytes must already be pinned to committed evidence by the caller.
    This implementation executes the same allowlisted model, not an independent
    physical implementation; independent reference checks are separate evidence.
    """
    from .verification_science import VerificationArtifactError, decode_verification_npz

    def cancel() -> None:
        _cancel(cancelled)

    cancel()
    if any(
        type(value) is not int or not 1 <= value <= 100_000_000
        for value in (max_particle_steps, max_lj_pair_checks)
    ):
        raise ValueError("Rerun workload limits must be positive bounded integers")
    if source_envelope.kind != "simulation":
        return [_check("source_contract", "unsupported", "rerun_requires_simulation_result")]
    expected_result = source_envelope.result.model_dump(mode="json")
    try:
        recorded_parameters = parameters_from_payload(
            {**source_envelope.parameters, "model": checkpoint_model_version(parameters)}
        )
        if (
            expected_result.get("model_version") != checkpoint_model_version(parameters)
            or checkpoint_parameters_sha256(recorded_parameters)
            != checkpoint_parameters_sha256(parameters)
            or expected_result.get("seed") != parameters.seed
            or expected_result.get("num_steps") != parameters.num_steps
        ):
            return [_check("source_contract", "failed", "source_parameters_do_not_match_rerun")]
    except (ValueError, TypeError, KeyError):
        return [
            _check("source_contract", "unsupported", "source_parameters_unavailable_or_invalid")
        ]

    particle_steps = parameters.num_particles * parameters.num_steps
    pair_bound = None
    if isinstance(parameters, LennardJonesParameters):
        pairs = parameters.num_particles * (parameters.num_particles - 1) // 2
        pair_bound = pairs * (
            parameters.num_steps + 2
            if parameters.interaction_backend == "direct"
            else 2 * (parameters.num_steps + 1) + 1
        )
    budget = {
        "particle_steps": particle_steps,
        "max_particle_steps": max_particle_steps,
        "lj_pair_checks_upper_bound": pair_bound,
        "max_lj_pair_checks": max_lj_pair_checks,
        "lj_bound_policy": "Initial/step force checks, worst-case neighbor rebuilds, and terminal RDF",
    }
    if particle_steps > max_particle_steps or (
        pair_bound is not None and pair_bound > max_lj_pair_checks
    ):
        return [_check("budget", "unsupported", "rerun_workload_exceeds_limit", details=budget)]
    checks = [_check("budget", "passed", "rerun_workload_within_limits", details=budget)]
    try:
        current = _current_environment(parameters)
        compatibility = _environment_check(parameters, source_envelope, current)
    except SimulationCancelled:
        raise
    except (BackendUnavailable, ImportError, ValueError, RuntimeError):
        return [
            *checks,
            _check("environment", "unsupported", "rerun_backend_or_fingerprint_unavailable"),
        ]
    checks.append(compatibility)
    if compatibility["status"] != "passed":
        return checks
    cancel()
    contracts = _contracts(parameters)
    sources = {}
    try:
        for name, contract in contracts.items():
            cancel()
            sources[name] = decode_verification_npz(
                artifacts[name], contract, check_cancelled=cancel
            )
    except SimulationCancelled:
        raise
    except (VerificationArtifactError, ValueError, TypeError, KeyError):
        return [*checks, _check("scientific_arrays", "failed", "source_array_snapshot_invalid")]

    parent = Path(temporary_root)
    parent.mkdir(parents=True, exist_ok=True)
    parent = _root(parent)
    try:
        with TemporaryDirectory(prefix="same-seed-rerun-", dir=parent) as directory:
            root = Path(directory)
            cancel()
            rerun = run_simulation(
                parameters,
                task_id="verification-rerun",
                attempt_id="current-runtime",
                artifacts_dir=root,
                tracker=None,
                cancel_event=_CancellationProbe(cancelled),
            )
            cancel()
            manifest = _json((root / rerun["artifacts"]["manifest"]).read_bytes())
            # Check actual execution metadata as well as preflight library versions.
            recorded_kernel = source_envelope.execution.get("kernel_version")
            observed_kernel = manifest["execution"].get("kernel_version")
            if recorded_kernel is not None and recorded_kernel != observed_kernel:
                checks[-1] = _check(
                    "environment",
                    "inconclusive",
                    "recorded_kernel_version_differs",
                    expected=recorded_kernel,
                    observed=observed_kernel,
                    details={"scope": "No numerical-equivalence assertion across kernel versions"},
                )
                return checks
            expected_hashes, observed_hashes = {}, {}
            for name, contract in contracts.items():
                observed = decode_verification_npz(
                    (root / rerun["artifacts"][name]).read_bytes(), contract, check_cancelled=cancel
                )
                for array_name in sorted(contract):
                    key = name + "/" + array_name
                    expected_hashes[key] = _array_digest(sources[name][array_name], cancel)
                    observed_hashes[key] = _array_digest(observed[array_name], cancel)
            equal = expected_hashes == observed_hashes
            checks.append(
                _check(
                    "scientific_arrays",
                    "passed" if equal else "failed",
                    "observed_canonical_arrays_equal_under_current_environment"
                    if equal
                    else "observed_canonical_arrays_differ",
                    expected=expected_hashes,
                    observed=observed_hashes,
                    details={
                        "comparison": "dtype, shape and C-order bytes, including signed zero",
                        "backend": parameters.backend,
                        "seed": parameters.seed,
                        "scope": "One same-seed realization; no cross-platform repeatability guarantee",
                    },
                )
            )
            expected = _scientific_scalars(parameters, expected_result)
            observed = _scientific_scalars(parameters, rerun)
            equal = _canonical(expected) == _canonical(observed)
            checks.append(
                _check(
                    "scientific_scalars",
                    "passed" if equal else "failed",
                    "observed_scientific_scalars_equal"
                    if equal
                    else "observed_scientific_scalars_differ",
                    expected=expected,
                    observed=observed,
                    details={
                        "comparison": "Canonical exact scientific values; identifiers and clocks excluded"
                    },
                )
            )
            for name in ("transport", "observables"):
                if name not in rerun["artifacts"]:
                    continue
                cancel()
                raw = artifacts.get(name)
                if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
                    checks.append(_check(name, "failed", "source_observables_snapshot_invalid"))
                    continue
                try:
                    expected_data = _json(raw)
                except (ValueError, TypeError):
                    checks.append(_check(name, "failed", "source_observables_snapshot_invalid"))
                    continue
                observed_data = _json((root / rerun["artifacts"][name]).read_bytes())
                expected_digest = hashlib.sha256(_canonical(expected_data)).hexdigest()
                observed_digest = hashlib.sha256(_canonical(observed_data)).hexdigest()
                checks.append(
                    _check(
                        name,
                        "passed" if expected_digest == observed_digest else "failed",
                        "observed_scientific_json_equal"
                        if expected_digest == observed_digest
                        else "observed_scientific_json_differs",
                        expected=expected_digest,
                        observed=observed_digest,
                        details={"comparison": "Canonical scientific JSON, not packaging bytes"},
                    )
                )
            cancel()
    except SimulationCancelled:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, FloatingPointError):
        return [*checks, _check("execution", "inconclusive", "bounded_rerun_could_not_complete")]
    return checks
