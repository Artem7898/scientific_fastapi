"""Prepare deterministic independent streams and aggregate committed replicas.

An ensemble coordinator consumes no scientific worker. Only child simulations
enter the existing bounded queue; reading a status never runs scientific code.
"""

import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from ..schemas.ensemble import EnsembleRequest, EnsembleStatus
from ..schemas.langevin import LangevinEnsembleParameters

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def valid_ensemble_id(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def prepare_ensemble(
    request: EnsembleRequest, *, ensemble_id: str
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Materialize and persist a seed list before any child is accepted.

    Retry attempts reuse these actual uint64 seeds, independent of worker IDs,
    scheduling and NumPy's process-global RNG. Metadata contains no generated
    task ID or timestamp, so it is suitable for idempotency comparisons.
    """
    if not valid_ensemble_id(ensemble_id) or len(ensemble_id) > 100:
        raise ValueError("Invalid ensemble identifier")
    canonical_request = request.model_dump()
    # Adding an explicit default must not invalidate keys accepted before
    # backend selection existed. NumPy is the same experiment when omitted.
    fingerprint_request = {
        **canonical_request,
        "parameters": dict(canonical_request["parameters"]),
    }
    if fingerprint_request["parameters"]["backend"] == "numpy":
        del fingerprint_request["parameters"]["backend"]
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_request, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    root = np.random.SeedSequence(request.base_seed)
    streams = root.spawn(request.replicas)
    seeds = [int(stream.generate_state(1, dtype=np.uint64)[0]) for stream in streams]
    children = [
        (
            f"{ensemble_id}_r{index:03d}",
            {
                **({"model": request.model} if request.model != "ideal-gas-ou-v1" else {}),
                **request.parameters.model_dump(),
                "seed": seed,
            },
        )
        for index, seed in enumerate(seeds)
    ]
    metadata = {
        "request": canonical_request,
        "request_fingerprint": fingerprint,
        "seeds": seeds,
        "seed_policy": {
            "name": "numpy.SeedSequence.spawn",
            "version": 1,
            "entropy": request.base_seed,
            "pool_size": root.pool_size,
            "dtype": "uint64",
            "numpy_version": np.__version__,
            "spawn_keys": [list(stream.spawn_key) for stream in streams],
        },
    }
    return children, metadata


def _statistic(values: list[float]) -> dict[str, Any]:
    mean = math.fsum(values) / len(values)
    standard_deviation = (
        math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": (
            standard_deviation / math.sqrt(len(values)) if standard_deviation is not None else None
        ),
    }


def aggregate_ensemble(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Describe logical replicas once each, in their original index order.

    Failed/cancelled children remain visible. ``require_all`` never silently
    aggregates the survivors. Empirical sample SD/SE are unavailable for M=1;
    the separate equilibrium-model prediction is explicitly labelled.
    """
    metadata = snapshot["metadata"]
    request = EnsembleRequest.model_validate(metadata["request"])
    child_ids = snapshot["children"]
    seeds = metadata["seeds"]
    if (
        len(child_ids) != request.replicas
        or len(seeds) != request.replicas
        or len(set(child_ids)) != request.replicas
    ):
        raise ValueError("Ensemble snapshot has inconsistent replica records")
    terminal = snapshot["results"]
    statuses = snapshot.get("child_statuses", {})
    counts = dict.fromkeys(("pending", "running", "completed", "failed", "cancelled"), 0)
    children = []
    for index, task_id in enumerate(child_ids):
        result = terminal.get(task_id)
        status = result["status"] if result is not None else statuses.get(task_id, "pending")
        if status not in counts or (status in TERMINAL_STATUSES and result is None):
            raise ValueError("Ensemble snapshot is missing a terminal contribution")
        if result is not None and result.get("task_id") != task_id:
            raise ValueError("Ensemble contribution does not match its logical replica")
        counts[status] += 1
        children.append(
            {
                "replica_index": index,
                "task_id": task_id,
                "seed": seeds[index],
                "status": status,
                "result": result,
            }
        )
    finished = counts["completed"] + counts["failed"] + counts["cancelled"]
    if finished == request.replicas:
        status = (
            "failed" if counts["failed"] else "cancelled" if counts["cancelled"] else "completed"
        )
    else:
        status = "pending" if counts["pending"] == request.replicas else "running"
    statistics = None
    theoretical_error = None
    if status == "completed":
        statistics = {}
        metrics = (
            ("final_temperature", "total_energy")
            if request.model == "ideal-gas-ou-v1"
            else ("thermal_temperature", "kinetic_energy", "final_centered_msd")
        )
        if request.model == "langevin-free-v1" and all(
            child["result"].get("diffusion", {}).get("status") == "available" for child in children
        ):
            metrics = (*metrics, "diffusion_estimate")
        for metric in metrics:
            values = []
            for child in children:
                value = (
                    child["result"]["diffusion"]["estimate"]
                    if metric == "diffusion_estimate"
                    else child["result"][metric]
                )
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Ensemble contribution has a nonnumeric observable")
                if not math.isfinite(value):
                    raise ValueError("Ensemble contribution has a nonfinite observable")
                values.append(float(value))
            statistics[metric] = _statistic(values)
        if isinstance(request.parameters, LangevinEnsembleParameters):
            parameters = request.parameters
            exponent = -2 * parameters.friction * parameters.num_steps * parameters.time_step
            thermal_temperature = parameters.initial_temperature * math.exp(exponent) + (
                parameters.bath_temperature * -math.expm1(exponent)
            )
            theoretical_error = thermal_temperature / math.sqrt(
                (parameters.num_particles - 1) * request.replicas
            )
        else:
            theoretical_error = request.parameters.initial_temperature / math.sqrt(
                request.parameters.num_particles * request.replicas
            )
    return EnsembleStatus.model_validate(
        {
            "ensemble_id": snapshot["ensemble_id"],
            "status": status,
            "cancel_requested": snapshot.get("cancel_requested", False),
            "model": request.model,
            "failure_policy": request.failure_policy,
            "counts": {"requested": request.replicas, **counts},
            "children": children,
            "statistics": statistics,
            "theoretical_temperature_standard_error": theoretical_error,
        }
    ).model_dump()


def ensemble_manifest(snapshot: dict[str, Any]) -> dict[str, Any]:
    """A terminal reproducibility record independent of temporary child TTLs."""
    result = aggregate_ensemble(snapshot)
    if result["status"] not in TERMINAL_STATUSES:
        raise ValueError("Ensemble manifest is available only after all replicas finish")
    return {
        "schema_version": 1,
        "ensemble_id": snapshot["ensemble_id"],
        **snapshot["metadata"],
        "outputs": result,
        "statistics_method": {
            "contribution": "One committed terminal result per logical replica in replica-index order",
            "sample_standard_deviation": "ddof=1; unavailable for one replica",
            "standard_error": "Sample standard deviation divided by sqrt(n); unavailable for one replica",
            "theoretical_temperature_standard_error": (
                "T_target / sqrt(N * M), independent equilibrium 2D snapshots"
                if result["model"] == "ideal-gas-ou-v1"
                else "T_velocity(t) / sqrt((N - 1) * M), independent 2D Gaussian velocity fluctuations about sample flow"
            ),
            "failure_policy": "require_all; no survivor-only aggregate",
            "limitations": "No confidence interval or physical-model validation is implied",
        },
    }
