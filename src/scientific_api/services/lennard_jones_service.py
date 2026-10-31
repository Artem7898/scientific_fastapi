"""Execute a bounded, versioned LJ experiment and publish auditable artifacts."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.experiment_tracker import ExperimentTracker
from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from ..models.lennard_jones_engine import create_lj_simulator
from ..research.lj_analysis import ANALYSIS_VERSION, SERIES_NAMES, analyze_lj_observables
from .langevin_service import _array_record, _restore_sample_prefix, _run_directory, _write_arrays
from .simulation_service import (
    _check_cancelled,
    _environment,
    _mark_tracking_failed,
    _sha256,
    _write_manifest,
)

if TYPE_CHECKING:
    from .checkpoint_service import CheckpointControl

logger = logging.getLogger(__name__)


def run_lj_simulation(
    parameters: LennardJonesParameters,
    *,
    task_id: str,
    artifacts_dir: Path,
    attempt_id: str | None = None,
    tracker: ExperimentTracker | None = None,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_control: CheckpointControl | None = None,
) -> dict[str, Any]:
    """Compute one scientific attempt; only the queue may commit its result.

    Sampled coordinates and observables share one schedule. The terminal RDF
    reserves its pair budget before analysis. No scientific artifact is exposed
    as a completed task until every artifact and the manifest have been written.
    """
    if not isinstance(parameters, LennardJonesParameters):
        raise TypeError("parameters must be LennardJonesParameters")
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
            tracker.start_run(run_name=f"lj_{task_id}")
            tracking_started = True
            tracker.log_params(
                {
                    **asdict(parameters),
                    "model_version": LJ_MODEL_VERSION,
                    "task_id": task_id,
                    **({"attempt_id": attempt_id} if attempt_id is not None else {}),
                }
            )
        loaded = checkpoint_control.loaded if checkpoint_control is not None else None
        if loaded is None:
            simulator = create_lj_simulator(parameters)
        trajectory = np.empty((len(sample_steps), parameters.num_particles, 2), dtype=np.float64)
        series = {name: np.empty(len(sample_steps), dtype=np.float64) for name in SERIES_NAMES}
        if loaded is None:
            trajectory[0] = simulator.unwrapped_positions
            initial_observables = simulator.observables()
            for name in SERIES_NAMES:
                series[name][0] = initial_observables[name]
            start_step, next_sample = 0, 1
            if checkpoint_control is not None:
                checkpoint_control.save_if_due(
                    0,
                    simulator,
                    samples={
                        "trajectory": trajectory[:next_sample],
                        **{name: values[:next_sample] for name, values in series.items()},
                    },
                    next_sample=next_sample,
                )
        else:
            from ..models.checkpoint_state import restore_simulator

            next_sample = _restore_sample_prefix(
                loaded, sample_steps, trajectory, parameters.num_steps, series=series
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
                values = simulator.observables()
                for name in SERIES_NAMES:
                    series[name][next_sample] = values[name]
                next_sample += 1
            if checkpoint_control is not None:
                # Capture the completed transition and initialized samples
                # before terminal RDF work is charged to the model budget.
                checkpoint_control.save_if_due(
                    step,
                    simulator,
                    samples={
                        "trajectory": trajectory[:next_sample],
                        **{name: values[:next_sample] for name, values in series.items()},
                    },
                    next_sample=next_sample,
                )
            if progress_callback is not None:
                now = monotonic()
                if step == parameters.num_steps or now - last_progress >= 0.5:
                    progress_callback(step, parameters.num_steps, "computing")
                    last_progress = now
        _check_cancelled(cancel_event)
        if next_sample != len(sample_steps):
            raise RuntimeError("LJ run did not collect its complete sampling schedule")
        # The final frame is always sampled, even when the stride does not
        # divide the number of steps. Never infer final values from a mean.
        metrics = {
            name: float(series[name][-1])
            for name in (
                "thermal_temperature",
                "kinetic_energy",
                "potential_energy",
                "total_energy",
                "pressure",
            )
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("LJ simulation produced nonfinite terminal metrics")
        simulator.reserve_analysis_pairs(
            parameters.num_particles * (parameters.num_particles - 1) // 2
        )
        analysis = analyze_lj_observables(trajectory, sample_steps, series, parameters=parameters)
        _check_cancelled(cancel_event)
        diagnostics = simulator.diagnostics()
        if tracker is not None:
            tracker.log_metrics(metrics)
        if progress_callback is not None:
            progress_callback(parameters.num_steps, parameters.num_steps, "saving_artifacts")
        _check_cancelled(cancel_event)
        state_arrays = {
            "positions": simulator.positions,
            "unwrapped_positions": simulator.unwrapped_positions,
            "velocities": simulator.velocities,
        }
        if not all(np.isfinite(array).all() for array in state_arrays.values()):
            raise FloatingPointError("LJ simulation produced nonfinite terminal state")
        state_path = run_dir / "final_state.npz"
        _write_arrays(state_path, state_arrays)
        _check_cancelled(cancel_event)
        trajectory_arrays = {"positions": trajectory, "sample_steps": sample_steps, "times": times}
        trajectory_path = run_dir / "trajectory.npz"
        _write_arrays(trajectory_path, trajectory_arrays)
        _check_cancelled(cancel_event)
        analysis_path = run_dir / "observables.json"
        _write_manifest(analysis_path, analysis)
        _check_cancelled(cancel_event)
        result = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "status": "completed",
            "model_version": LJ_MODEL_VERSION,
            "seed": parameters.seed,
            "num_steps": parameters.num_steps,
            "ensemble": parameters.ensemble,
            **metrics,
            "diagnostics": diagnostics,
            "units": dict.fromkeys(
                ("temperature", "energy", "time", "length", "mass", "pressure"), "reduced"
            ),
            "artifacts": {
                name: (run_dir / filename).relative_to(root).as_posix()
                for name, filename in (
                    ("manifest", "manifest.json"),
                    ("final_state", "final_state.npz"),
                    ("trajectory", "trajectory.npz"),
                    ("observables", "observables.json"),
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
                "version": LJ_MODEL_VERSION,
                "dimensions": 2,
                "ensemble": parameters.ensemble,
                "integrator": "velocity-verlet" if parameters.ensemble == "nve" else "baoab",
                "mass": parameters.mass,
                "boltzmann_constant": 1.0,
                "epsilon": parameters.epsilon,
                "sigma": parameters.sigma,
                "box_size": list(parameters.box_size),
                "boundary": "periodic; minimum-image pair displacement",
                "coordinate_policy": "unwrapped dynamics/trajectory; terminal positions wrapped for presentation",
                "potential": {
                    "name": "Lennard-Jones 12-6 with shifted-force cutoff",
                    "bare_potential": "4*epsilon*((sigma/r)^12-(sigma/r)^6)",
                    "shifted_potential": "U(r)-U(rc)+(r-rc)*F(rc) for r<rc; zero otherwise",
                    "shifted_radial_force": "F(r)-F(rc) for r<rc; zero otherwise",
                    "cutoff_radius": parameters.cutoff_radius,
                    "tail_correction": None,
                },
                "thermostat": (
                    {
                        "scheme": "BAOAB with exact independent lab-frame OU substep",
                        "bath_temperature": parameters.bath_temperature,
                        "velocity_relaxation_rate": parameters.friction,
                    }
                    if parameters.ensemble == "nvt"
                    else None
                ),
                "initialization": {
                    "positions": "Regular rectangular lattice; no random position jitter",
                    "velocities": "PCG64 Gaussian directions; sample mean removed and kinetic temperature rescaled",
                    "degrees_of_freedom": 2 * (parameters.num_particles - 1),
                    "canonical_sample": False,
                },
                "assumptions": [
                    "Two-dimensional pairwise interactions; no external field or additional forces",
                    "Reduced units with k_B=1; no experimental material calibration",
                    "A finite-step integrator; exact OU substeps do not make the interacting dynamics exact",
                    "Equilibration is not inferred automatically from initial temperature or a short trajectory",
                ],
            },
            "sampling": {
                "sample_stride": parameters.sample_stride,
                "sample_steps": sample_steps.tolist(),
                "coordinate_convention": "unwrapped",
                "includes_initial_and_terminal": True,
                "trajectory_array_bytes": trajectory.nbytes,
                "equilibration_steps": parameters.equilibration_steps,
            },
            "analysis": {
                "version": ANALYSIS_VERSION,
                "rdf": analysis["rdf"]["normalization"],
                "rdf_snapshot": "terminal frame only",
                "energy_drift": analysis["energy_drift"],
            },
            "uncertainty": {
                "temperature": "Flow-subtracted kinetic estimator with 2*(N-1) degrees of freedom",
                "time_statistics": "Saved-frame mean and sample SD; not duration-weighted; correlated frames have no independent-sample SE",
                "rdf": "Descriptive terminal snapshot; pair/bin counts are not independent uncertainty units",
                "excludes": [
                    "residual equilibration bias",
                    "finite-step integration error",
                    "physical-model discrepancy",
                    "finite-size and cutoff effects",
                ],
            },
            "rng": {
                "bit_generator": "PCG64",
                "seed": parameters.seed,
                "dtype": "float64",
                "policy": "Local generator for initial velocity directions and optional NVT OU increments; sampling consumes no randomness",
            },
            "diagnostics": diagnostics,
            "tracking": {"run_id": getattr(tracker, "run_id", None)},
            "environment": _environment(),
            "artifacts": [
                _array_record(state_path, root, state_arrays),
                _array_record(trajectory_path, root, trajectory_arrays),
                {
                    "path": analysis_path.relative_to(root).as_posix(),
                    "sha256": _sha256(analysis_path),
                    "bytes": analysis_path.stat().st_size,
                    "format": "json",
                },
            ],
        }
        if checkpoint_control is not None:
            manifest["continuation"] = checkpoint_control.manifest_metadata()
        json.dumps(result, allow_nan=False)
        _check_cancelled(cancel_event)
        _write_manifest(run_dir / "manifest.json", manifest)
        from .result_envelope_service import publish_result_envelope

        publish_result_envelope(result, artifacts_dir=root)
        _check_cancelled(cancel_event)
        log_artifact = getattr(tracker, "log_artifact", None)
        if callable(log_artifact):
            for path in (state_path, trajectory_path, analysis_path, run_dir / "manifest.json"):
                _check_cancelled(cancel_event)
                log_artifact(path)
        _check_cancelled(cancel_event)
        logger.info("LJ simulation %s completed with energy=%s", task_id, metrics["total_energy"])
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
                    logger.warning("Failed to end tracking after an LJ run failure", exc_info=True)
                else:
                    _mark_tracking_failed(tracker)
                    raise
