"""Execute a scientific experiment and persist its reproducibility record."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core.experiment_tracker import ExperimentTracker
from ..domain.langevin import LangevinParameters
from ..domain.lennard_jones import LennardJonesParameters
from ..domain.simulation import MODEL_VERSION, SimulationCancelled, SimulationParameters
from ..models.computation import create_simulator

if TYPE_CHECKING:
    from .checkpoint_service import CheckpointControl

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    """Record versions without depending on a checkout or exposing env vars."""
    try:
        package_version = importlib.metadata.distribution("scientific-api").metadata.get("Version")
        package_version = package_version or "uninstalled"
    except importlib.metadata.PackageNotFoundError:
        package_version = "uninstalled"
    package_root = Path(__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        source_digest.update(source.relative_to(package_root).as_posix().encode())
        source_digest.update(b"\0")
        source_digest.update(source.read_bytes())
    commit = None
    dirty = None
    try:
        checkout = package_root.parent.parent
        commit_result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = commit_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # A wheel normally runs outside a Git checkout. The package version
        # and source digest still identify the implementation used.
        pass
    lockfile = package_root.parent.parent / "uv.lock"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "package_version": package_version,
        "source_commit": commit,
        "source_dirty": dirty,
        "source_sha256": source_digest.hexdigest(),
        "lockfile_sha256": _sha256(lockfile) if lockfile.is_file() else None,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _write_state(path: Path, velocities: NDArray[np.float64]) -> None:
    """Publish a complete NPZ atomically; never expose a partial artifact."""
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, velocities=velocities)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".manifest-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SimulationCancelled("Simulation cancelled")


def _mark_tracking_failed(tracker: ExperimentTracker) -> None:
    mark_failed = getattr(tracker, "mark_failed", None)
    if callable(mark_failed):
        try:
            mark_failed()
        except Exception:
            logger.warning("Failed to mark experiment tracking as failed", exc_info=True)


def run_simulation(
    parameters: SimulationParameters | LangevinParameters | LennardJonesParameters,
    *,
    task_id: str,
    artifacts_dir: Path,
    attempt_id: str | None = None,
    tracker: ExperimentTracker | None = None,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_control: CheckpointControl | None = None,
) -> dict[str, Any]:
    """Compute a seeded trajectory and an auditable terminal-state artifact.

    The caller owns the stable task identifier and an optional unique attempt
    identifier. Queue workers must pass their lease owner as ``attempt_id`` so
    a stale worker cannot overwrite another attempt's artifacts. Artifact
    paths are relative to ``artifacts_dir``. Cancellation is cooperative at
    step boundaries. An exception, including tracking or persistence failure,
    is propagated to the caller instead of returning a successful experiment.
    """
    if isinstance(parameters, LennardJonesParameters):
        from .lennard_jones_service import run_lj_simulation

        return run_lj_simulation(
            parameters,
            task_id=task_id,
            artifacts_dir=artifacts_dir,
            attempt_id=attempt_id,
            tracker=tracker,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            checkpoint_control=checkpoint_control,
        )
    if isinstance(parameters, LangevinParameters):
        from .langevin_service import run_langevin_simulation

        return run_langevin_simulation(
            parameters,
            task_id=task_id,
            artifacts_dir=artifacts_dir,
            attempt_id=attempt_id,
            tracker=tracker,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            checkpoint_control=checkpoint_control,
        )
    if not isinstance(parameters, SimulationParameters):
        raise TypeError("parameters must be SimulationParameters")
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
        raise ValueError("task_id must contain 1-128 ASCII letters, digits, underscores or hyphens")
    if attempt_id is not None and (
        not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt_id)
    ):
        raise ValueError(
            "attempt_id must contain 1-128 ASCII letters, digits, underscores or hyphens"
        )
    _check_cancelled(cancel_event)
    artifacts_dir = Path(artifacts_dir).resolve()
    if checkpoint_control is not None:
        checkpoint_control.validate_run(
            parameters, task_id=task_id, attempt_id=attempt_id, artifacts_dir=artifacts_dir
        )
    task_dir = artifacts_dir / task_id
    if task_dir.is_symlink():
        raise ValueError("Artifact directory must not be a symlink")
    task_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.resolve().parent != artifacts_dir:
        raise ValueError("Artifact directory must remain inside artifacts_dir")
    run_dir = task_dir
    if attempt_id is not None:
        run_dir = task_dir / attempt_id
        if run_dir.is_symlink():
            raise ValueError("Attempt artifact directory must not be a symlink")
        run_dir.mkdir(exist_ok=True)
        if run_dir.resolve().parent != task_dir.resolve():
            raise ValueError("Attempt artifact directory must remain inside the task directory")

    tracking_started = False
    failure_in_flight = False
    try:
        if tracker is not None:
            tracker.start_run(run_name=f"sim_{task_id}")
            tracking_started = True
            tracker.log_params(
                {
                    **asdict(parameters),
                    "model_version": MODEL_VERSION,
                    "task_id": task_id,
                    **({"attempt_id": attempt_id} if attempt_id is not None else {}),
                }
            )
        loaded = checkpoint_control.loaded if checkpoint_control is not None else None
        if loaded is None:
            simulator = create_simulator(parameters)
            start_step = 0
            if checkpoint_control is not None:
                checkpoint_control.save_if_due(0, simulator, samples={}, next_sample=0)
        else:
            from ..models.checkpoint_state import restore_simulator

            if (
                type(loaded.completed_steps) is not int
                or not 0 <= loaded.completed_steps <= parameters.num_steps
                or type(loaded.next_sample) is not int
                or loaded.next_sample != 0
                or loaded.samples
            ):
                raise ValueError("Invalid OU checkpoint sampling state")
            simulator = restore_simulator(parameters, loaded.model_metadata, loaded.model_arrays)
            start_step = loaded.completed_steps
            loaded = None
            assert checkpoint_control is not None
            checkpoint_control.release_loaded()
        last_progress = monotonic()
        if progress_callback is not None:
            progress_callback(start_step, parameters.num_steps, "computing")
        for step in range(start_step + 1, parameters.num_steps + 1):
            _check_cancelled(cancel_event)
            simulator.advance()
            if checkpoint_control is not None:
                checkpoint_control.save_if_due(step, simulator, samples={}, next_sample=0)
            if progress_callback is not None:
                now = monotonic()
                if step == parameters.num_steps or now - last_progress >= 0.5:
                    progress_callback(step, parameters.num_steps, "computing")
                    last_progress = now
        _check_cancelled(cancel_event)
        # The workflow publishes a terminal snapshot. Intermediate reductions
        # did not contribute to the result; retain every stochastic transition
        # and cancellation boundary, but calculate observables only once.
        final_temperature, total_energy = simulator.observables()
        uncertainty = parameters.initial_temperature / math.sqrt(parameters.num_particles)
        metrics = {
            "final_temperature": final_temperature,
            "total_energy": total_energy,
            "temperature_standard_error": uncertainty,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Simulation produced nonfinite metrics")
        if tracker is not None:
            tracker.log_metrics(metrics)

        if progress_callback is not None:
            progress_callback(parameters.num_steps, parameters.num_steps, "saving_artifacts")
        _check_cancelled(cancel_event)
        state_path = run_dir / "final_state.npz"
        _write_state(state_path, simulator.velocities)
        relative_state = state_path.relative_to(artifacts_dir).as_posix()
        result = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "status": "completed",
            **metrics,
            "model_version": MODEL_VERSION,
            "seed": parameters.seed,
            "num_steps": parameters.num_steps,
            "units": {"temperature": "reduced", "energy": "reduced", "time": "reduced"},
            "artifacts": {
                "manifest": (run_dir / "manifest.json").relative_to(artifacts_dir).as_posix(),
                "final_state": relative_state,
            },
        }
        manifest = {
            "schema_version": 1,
            "result_envelope": {
                "schema_version": 1,
                "path": (run_dir / "envelope.json").relative_to(artifacts_dir).as_posix(),
            },
            "task_id": task_id,
            "attempt_id": attempt_id,
            "created_at": datetime.now(UTC).isoformat(),
            "parameters": asdict(parameters),
            "execution": simulator.execution_metadata(),
            "outputs": result,
            "model": {
                "version": MODEL_VERSION,
                "method": "exact Ornstein-Uhlenbeck transition",
                "dimensions": 2,
                "mass": 1.0,
                "boltzmann_constant": 1.0,
                "relaxation_rate": 1.0,
                "target_temperature": parameters.initial_temperature,
                "assumptions": [
                    "Independent noninteracting particles with Maxwellian initial velocities",
                    "Reduced units; no physical Kelvin or femtosecond conversion",
                    "No positions, boundaries, collisions or interparticle forces",
                ],
            },
            "uncertainty": {
                "method": "Exact equilibrium standard error T_target / sqrt(num_particles)",
                "scope": "One terminal snapshot with independent two-dimensional velocities",
                "excludes": [
                    "model discrepancy",
                    "parameter uncertainty",
                    "time-series uncertainty",
                ],
            },
            "rng": {"bit_generator": "PCG64", "seed": parameters.seed, "dtype": "float64"},
            "tracking": {"run_id": getattr(tracker, "run_id", None)},
            "environment": _environment(),
            "artifacts": [
                {
                    "path": relative_state,
                    "sha256": _sha256(state_path),
                    "format": "npz",
                    "arrays": {
                        "velocities": {"shape": [parameters.num_particles, 2], "dtype": "float64"}
                    },
                }
            ],
        }
        if checkpoint_control is not None:
            manifest["continuation"] = checkpoint_control.manifest_metadata()
        _check_cancelled(cancel_event)
        _write_manifest(run_dir / "manifest.json", manifest)
        from .result_envelope_service import publish_result_envelope

        publish_result_envelope(result, artifacts_dir=artifacts_dir)
        _check_cancelled(cancel_event)
        log_artifact = getattr(tracker, "log_artifact", None)
        if callable(log_artifact):
            log_artifact(state_path)
            log_artifact(run_dir / "manifest.json")
        logger.info("Simulation %s completed with T=%s", task_id, final_temperature)
        return result
    except BaseException:
        failure_in_flight = True
        if tracker is not None:
            _mark_tracking_failed(tracker)
        raise
    finally:
        if tracker is not None and tracking_started:
            try:
                tracker.end_run()
            except Exception:
                if failure_in_flight:
                    logger.warning(
                        "Failed to end experiment tracking after a run failure", exc_info=True
                    )
                else:
                    _mark_tracking_failed(tracker)
                    raise
