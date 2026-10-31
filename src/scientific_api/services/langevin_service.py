"""Execute and persist a position-aware, driven Langevin experiment."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.experiment_tracker import ExperimentTracker
from ..domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from ..models.langevin_engine import create_langevin_simulator, langevin_moments
from ..research.transport_analysis import analyze_transport
from .simulation_service import (
    _check_cancelled,
    _environment,
    _mark_tracking_failed,
    _sha256,
    _write_manifest,
)

if TYPE_CHECKING:
    from .checkpoint_service import CheckpointControl, LoadedCheckpoint

logger = logging.getLogger(__name__)
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _run_directory(root: Path, task_id: str, attempt_id: str | None) -> Path:
    for name, value in (("task_id", task_id), ("attempt_id", attempt_id)):
        if name == "attempt_id" and value is None:
            continue
        if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
            raise ValueError(
                f"{name} must contain 1-128 ASCII letters, digits, underscores or hyphens"
            )
    task_dir = root / task_id
    if task_dir.is_symlink():
        raise ValueError("Artifact directory must not be a symlink")
    task_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.resolve().parent != root:
        raise ValueError("Artifact directory must remain inside artifacts_dir")
    if attempt_id is None:
        return task_dir
    run_dir = task_dir / attempt_id
    if run_dir.is_symlink():
        raise ValueError("Attempt artifact directory must not be a symlink")
    run_dir.mkdir(exist_ok=True)
    if run_dir.resolve().parent != task_dir.resolve():
        raise ValueError("Attempt artifact directory must remain inside the task directory")
    return run_dir


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish all named arrays together, without exposing an incomplete ZIP."""
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Scientific state arrays must not contain Python objects")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".arrays-", suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            # NumPy <2.2 interprets an allow_pickle keyword as another array.
            # All arrays here are numeric; reject object dtype above instead.
            np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _array_record(path: Path, root: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "format": "npz",
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in arrays.items()
        },
    }


def _restore_sample_prefix(
    loaded: LoadedCheckpoint,
    sample_steps: np.ndarray,
    trajectory: np.ndarray,
    num_steps: int,
    *,
    series: dict[str, np.ndarray] | None = None,
) -> int:
    """Restore only initialized samples, preserving the original global schedule."""
    step = loaded.completed_steps
    if type(step) is not int or not 0 <= step <= num_steps:
        raise ValueError("Checkpoint step is outside the original simulation")
    next_sample = int(np.searchsorted(sample_steps, step, side="right"))
    expected_names = {"trajectory", *(series or {})}
    if (
        type(loaded.next_sample) is not int
        or loaded.next_sample != next_sample
        or set(loaded.samples) != expected_names
    ):
        raise ValueError("Checkpoint sampling cursor does not match its global schedule")
    prefix = loaded.samples["trajectory"]
    if (
        not isinstance(prefix, np.ndarray)
        or prefix.dtype != np.dtype(np.float64)
        or prefix.shape != (next_sample, *trajectory.shape[1:])
        or not prefix.flags.c_contiguous
        or not np.isfinite(prefix).all()
    ):
        raise ValueError("Checkpoint trajectory prefix is invalid")
    trajectory[:next_sample] = prefix
    for name, values in (series or {}).items():
        saved = loaded.samples[name]
        if (
            not isinstance(saved, np.ndarray)
            or saved.dtype != np.dtype(np.float64)
            or saved.shape != (next_sample,)
            or not saved.flags.c_contiguous
            or not np.isfinite(saved).all()
        ):
            raise ValueError("Checkpoint observable prefix is invalid")
        values[:next_sample] = saved
    return next_sample


def run_langevin_simulation(
    parameters: LangevinParameters,
    *,
    task_id: str,
    artifacts_dir: Path,
    attempt_id: str | None = None,
    tracker: ExperimentTracker | None = None,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_control: CheckpointControl | None = None,
) -> dict[str, Any]:
    """Persist terminal state, sampled unwrapped coordinates and transport data.

    Scientific state is local to this call. A queue lease owns an immutable
    attempt directory; only the task store can commit this return value. The
    final manifest is published after all other artifacts are complete.
    """
    if not isinstance(parameters, LangevinParameters):
        raise TypeError("parameters must be LangevinParameters")
    _check_cancelled(cancel_event)
    root = Path(artifacts_dir).resolve()
    if checkpoint_control is not None:
        checkpoint_control.validate_run(
            parameters, task_id=task_id, attempt_id=attempt_id, artifacts_dir=root
        )
    run_dir = _run_directory(root, task_id, attempt_id)
    sample_steps = np.asarray(parameters.sampling_steps(), dtype=np.int64)
    times = sample_steps.astype(np.float64) * parameters.time_step
    tracking_started = False
    failure_in_flight = False
    try:
        if tracker is not None:
            tracker.start_run(run_name=f"langevin_{task_id}")
            tracking_started = True
            tracker.log_params(
                {
                    **asdict(parameters),
                    "model_version": LANGEVIN_MODEL_VERSION,
                    "task_id": task_id,
                    **({"attempt_id": attempt_id} if attempt_id is not None else {}),
                }
            )
        loaded = checkpoint_control.loaded if checkpoint_control is not None else None
        if loaded is None:
            simulator = create_langevin_simulator(parameters)
        trajectory = np.empty((len(sample_steps), parameters.num_particles, 2), dtype=np.float64)
        if loaded is None:
            trajectory[0] = simulator.unwrapped_positions
            start_step, next_sample = 0, 1
            if checkpoint_control is not None:
                checkpoint_control.save_if_due(
                    0,
                    simulator,
                    samples={"trajectory": trajectory[:next_sample]},
                    next_sample=next_sample,
                )
        else:
            from ..models.checkpoint_state import restore_simulator

            next_sample = _restore_sample_prefix(
                loaded, sample_steps, trajectory, parameters.num_steps
            )
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
            if next_sample < len(sample_steps) and step == sample_steps[next_sample]:
                trajectory[next_sample] = simulator.unwrapped_positions
                next_sample += 1
            if checkpoint_control is not None:
                checkpoint_control.save_if_due(
                    step,
                    simulator,
                    samples={"trajectory": trajectory[:next_sample]},
                    next_sample=next_sample,
                )
            if progress_callback is not None:
                now = monotonic()
                if step == parameters.num_steps or now - last_progress >= 0.5:
                    progress_callback(step, parameters.num_steps, "computing")
                    last_progress = now
        _check_cancelled(cancel_event)
        if next_sample != len(sample_steps):
            raise RuntimeError("Simulation did not collect its complete sampling schedule")
        with np.errstate(over="raise", invalid="raise"):
            mean_velocity = simulator.velocities.mean(axis=0)
            thermal_velocities = simulator.velocities - mean_velocity
            kinetic_energy = float(
                0.5
                * parameters.mass
                * np.einsum("ij,ij->", simulator.velocities, simulator.velocities)
            )
            thermal_temperature = float(
                parameters.mass
                * np.einsum("ij,ij->", thermal_velocities, thermal_velocities)
                / (2 * (parameters.num_particles - 1))
            )
        theory = langevin_moments(parameters.num_steps * parameters.time_step, parameters)
        thermal_temperature_standard_error = (
            parameters.mass
            * theory["velocity_variance_per_component"]
            / math.sqrt(parameters.num_particles - 1)
        )
        transport = analyze_transport(trajectory, sample_steps, parameters=parameters)
        _check_cancelled(cancel_event)
        metrics = {
            "thermal_temperature": thermal_temperature,
            "thermal_temperature_standard_error": thermal_temperature_standard_error,
            "kinetic_energy": kinetic_energy,
            "final_msd": transport["msd"][-1],
            "final_centered_msd": transport["centered_msd"][-1],
        }
        if (
            not np.isfinite(mean_velocity).all()
            or not np.isfinite(trajectory).all()
            or not np.isfinite(simulator.velocities).all()
            or not all(math.isfinite(value) for value in metrics.values())
        ):
            raise FloatingPointError("Simulation produced nonfinite state or metrics")
        if tracker is not None:
            tracked_metrics = dict(metrics)
            for name in ("estimate", "standard_error"):
                value = transport["diffusion"][name]
                if value is not None:
                    tracked_metrics[f"diffusion_{name}"] = value
            tracker.log_metrics(tracked_metrics)
        if progress_callback is not None:
            progress_callback(parameters.num_steps, parameters.num_steps, "saving_artifacts")
        _check_cancelled(cancel_event)
        state_arrays = {
            "positions": simulator.positions,
            "unwrapped_positions": simulator.unwrapped_positions,
            "velocities": simulator.velocities,
        }
        if not all(np.isfinite(array).all() for array in state_arrays.values()):
            raise FloatingPointError("Simulation produced nonfinite terminal state")
        state_path = run_dir / "final_state.npz"
        _write_arrays(state_path, state_arrays)
        _check_cancelled(cancel_event)
        trajectory_arrays = {"positions": trajectory, "sample_steps": sample_steps, "times": times}
        trajectory_path = run_dir / "trajectory.npz"
        _write_arrays(trajectory_path, trajectory_arrays)
        _check_cancelled(cancel_event)
        transport_path = run_dir / "transport.json"
        _write_manifest(transport_path, transport)
        result = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "status": "completed",
            "model_version": LANGEVIN_MODEL_VERSION,
            "seed": parameters.seed,
            "num_steps": parameters.num_steps,
            **metrics,
            "mean_velocity": mean_velocity.tolist(),
            "mean_displacement": transport["mean_displacement"][-1],
            "diffusion": transport["diffusion"],
            "units": dict.fromkeys(
                ("temperature", "energy", "time", "length", "mass", "acceleration", "diffusion"),
                "reduced",
            ),
            "artifacts": {
                name: (run_dir / filename).relative_to(root).as_posix()
                for name, filename in (
                    ("manifest", "manifest.json"),
                    ("final_state", "final_state.npz"),
                    ("trajectory", "trajectory.npz"),
                    ("transport", "transport.json"),
                )
            },
        }
        manifest = {
            "schema_version": 1,
            "result_envelope": {
                "schema_version": 1,
                "path": (run_dir / "envelope.json").relative_to(root).as_posix(),
            },
            "task_id": task_id,
            "attempt_id": attempt_id,
            "created_at": datetime.now(UTC).isoformat(),
            "parameters": asdict(parameters),
            "execution": simulator.execution_metadata(),
            "outputs": result,
            "model": {
                "version": LANGEVIN_MODEL_VERSION,
                "method": "exact joint integrated Ornstein-Uhlenbeck transition",
                "dimensions": 2,
                "mass": parameters.mass,
                "boltzmann_constant": 1.0,
                "relaxation_rate": parameters.friction,
                "bath_temperature": parameters.bath_temperature,
                "initial_temperature": parameters.initial_temperature,
                "constant_acceleration": list(parameters.acceleration),
                "velocity_frame": "laboratory-frame friction; thermal moments subtract sample mean",
                "boundary": parameters.boundary,
                "box_size": list(parameters.box_size) if parameters.box_size is not None else None,
                "coordinate_policy": "unwrapped dynamics and sampled trajectory; periodic terminal positions are wrapped for presentation",
                "assumptions": [
                    "Independent noninteracting particles in two dimensions",
                    "Reduced units with Boltzmann constant equal to one",
                    "Gaussian initial velocities with specified temperature and mean",
                    "Constant acceleration; no interparticle forces or position-dependent potential",
                    "Periodic wrapping does not represent walls, sedimentation or a hydrostatic density profile",
                ],
            },
            "sampling": {
                "sample_stride": parameters.sample_stride,
                "sample_steps": sample_steps.tolist(),
                "includes_initial_and_terminal": True,
                "coordinate_convention": "unwrapped",
                "trajectory_array_bytes": trajectory.nbytes,
            },
            "uncertainty": {
                "thermal_temperature": {
                    "estimator": "mass * sum(|v - sample_mean(v)|^2) / (2 * (N - 1))",
                    "standard_error": "analytic transient temperature / sqrt(N - 1)",
                    "scope": "Independent Gaussian particle velocities at one terminal snapshot",
                },
                "msd": "Single initial-time origin; raw MSD and unbiased drift-centered displacement variance",
                "diffusion": transport["diffusion"],
                "independent_units": "Particles; samples within a particle trajectory remain correlated",
                "excludes": [
                    "model discrepancy",
                    "parameter uncertainty",
                    "numerical-platform differences",
                ],
            },
            "rng": {
                "bit_generator": "PCG64",
                "seed": parameters.seed,
                "dtype": "float64",
                "policy": "Local NumPy generator; joint position/velocity innovations; sampling consumes no random values",
            },
            "tracking": {"run_id": getattr(tracker, "run_id", None)},
            "environment": _environment(),
            "artifacts": [
                _array_record(state_path, root, state_arrays),
                _array_record(trajectory_path, root, trajectory_arrays),
                {
                    "path": transport_path.relative_to(root).as_posix(),
                    "sha256": _sha256(transport_path),
                    "bytes": transport_path.stat().st_size,
                    "format": "json",
                },
            ],
        }
        if checkpoint_control is not None:
            manifest["continuation"] = checkpoint_control.manifest_metadata()
        _check_cancelled(cancel_event)
        _write_manifest(run_dir / "manifest.json", manifest)
        from .result_envelope_service import publish_result_envelope

        publish_result_envelope(result, artifacts_dir=root)
        _check_cancelled(cancel_event)
        log_artifact = getattr(tracker, "log_artifact", None)
        if callable(log_artifact):
            for path in (state_path, trajectory_path, transport_path, run_dir / "manifest.json"):
                _check_cancelled(cancel_event)
                log_artifact(path)
        _check_cancelled(cancel_event)
        # Validate even when the caller does not persist the response itself.
        json.dumps(result, allow_nan=False)
        logger.info(
            "Langevin simulation %s completed with thermal T=%s", task_id, thermal_temperature
        )
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
