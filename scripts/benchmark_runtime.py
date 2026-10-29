"""Measure the actual seeded CPU simulation path, separately from kernel benchmarks.

Each backend runs in a fresh child process. Its first run includes optional
backend import/JIT and writes real NPZ/manifest artifacts. Warm samples include
PCG64, evolution, observables, environment capture and artifact persistence.
HTTP, queue waiting, tracking, and SSE are deliberately outside this measurement.
Temporary artifacts are validated against NumPy and removed after measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402

from scientific_api.domain.simulation import (  # noqa: E402
    MODEL_VERSION,
    SimulationParameters,
)
from scientific_api.models.computation import (  # noqa: E402
    BackendUnavailable,
    create_simulator,
)
from scientific_api.services import simulation_service  # noqa: E402

TOLERANCE = {"rtol": 1e-11, "atol": 1e-11}


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
    }


def _validate(
    velocities: np.ndarray,
    metrics: tuple[float, float],
    reference: np.ndarray,
    reference_metrics: tuple[float, float],
) -> float:
    if velocities.shape != reference.shape or velocities.dtype != np.float64:
        raise RuntimeError("Backend returned an incorrect final array contract")
    if not np.isfinite(velocities).all() or not np.isfinite(metrics).all():
        raise RuntimeError("Backend returned a nonfinite final state")
    if not np.allclose(velocities, reference, **TOLERANCE):
        raise RuntimeError("Final velocities differ from the seeded NumPy reference")
    if not np.allclose(metrics, reference_metrics, **TOLERANCE):
        raise RuntimeError("Final observables differ from the seeded NumPy reference")
    return float(np.max(np.abs(velocities - reference)))


def _timed_run(parameters: SimulationParameters, directory: Path, task_id: str) -> dict[str, Any]:
    """Time persistence inside the real service without adding per-step timers."""
    durations = {
        "state_persistence_seconds": 0.0,
        "manifest_persistence_seconds": 0.0,
        "environment_capture_seconds": 0.0,
    }
    originals = {
        "_write_state": simulation_service._write_state,
        "_write_manifest": simulation_service._write_manifest,
        "_environment": simulation_service._environment,
    }
    names = {
        "_write_state": "state_persistence_seconds",
        "_write_manifest": "manifest_persistence_seconds",
        "_environment": "environment_capture_seconds",
    }

    def wrapper(function, name):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                durations[name] += time.perf_counter() - start

        return timed

    for name, function in originals.items():
        setattr(simulation_service, name, wrapper(function, names[name]))
    try:
        start = time.perf_counter()
        result = simulation_service.run_simulation(
            parameters, task_id=task_id, artifacts_dir=directory
        )
        durations["wall_seconds"] = time.perf_counter() - start
    finally:
        for name, function in originals.items():
            setattr(simulation_service, name, function)
    with np.load(directory / result["artifacts"]["final_state"], allow_pickle=False) as state:
        velocities = state["velocities"].copy()
    manifest = json.loads((directory / result["artifacts"]["manifest"]).read_text())
    if manifest["execution"]["backend"] != parameters.backend:
        raise RuntimeError("Persisted execution backend differs from the requested backend")
    if manifest["parameters"] != asdict(parameters):
        raise RuntimeError("Persisted parameters do not identify the requested backend")
    shutil.rmtree(directory / task_id)
    return {
        "durations": durations,
        "velocities": velocities,
        "manifest": manifest,
        "metrics": (result["final_temperature"], result["total_energy"]),
    }


def _profile(parameters: SimulationParameters) -> tuple[dict[str, float], np.ndarray, tuple]:
    start = time.perf_counter()
    simulator = create_simulator(parameters)
    initialization = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(parameters.num_steps):
        simulator.advance()
    evolution = time.perf_counter() - start
    start = time.perf_counter()
    metrics = simulator.observables()
    observables = time.perf_counter() - start
    # A separate diagnostic, not a decomposition of the evolution sample.
    rng = np.random.Generator(np.random.PCG64(parameters.seed))
    rng.normal(0.0, np.sqrt(parameters.initial_temperature), size=(parameters.num_particles, 2))
    start = time.perf_counter()
    for _ in range(parameters.num_steps):
        rng.standard_normal((parameters.num_particles, 2))
    rng_only = time.perf_counter() - start
    return (
        {
            "initialization_seconds": initialization,
            "evolution_including_rng_seconds": evolution,
            "terminal_observables_seconds": observables,
            "rng_only_seconds": rng_only,
        },
        simulator.velocities,
        metrics,
    )


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


def _measure_backend(parameters: SimulationParameters, repeats: int) -> dict[str, Any]:
    # No create_simulator() call occurs before this first real service run.
    with tempfile.TemporaryDirectory(prefix="scientific-runtime-benchmark-") as temporary:
        directory = Path(temporary)
        try:
            cold = _timed_run(parameters, directory, "cold")
        except BackendUnavailable as error:
            return {"status": "skipped", "reason": str(error)}
        reference = create_simulator(
            SimulationParameters(**{**asdict(parameters), "backend": "numpy"})
        )
        for _ in range(parameters.num_steps):
            reference.advance()
        reference_metrics = reference.observables()
        max_error = _validate(
            cold["velocities"], cold["metrics"], reference.velocities, reference_metrics
        )
        profile_samples: list[dict[str, float]] = []
        full_samples: list[dict[str, float]] = []
        for index in range(repeats):
            profile, velocities, metrics = _profile(parameters)
            max_error = max(
                max_error, _validate(velocities, metrics, reference.velocities, reference_metrics)
            )
            profile_samples.append(profile)
            sample = _timed_run(parameters, directory, f"warm_{index}")
            max_error = max(
                max_error,
                _validate(
                    sample["velocities"], sample["metrics"], reference.velocities, reference_metrics
                ),
            )
            # Repeatability is tested independently of cross-backend tolerance.
            if not np.array_equal(sample["velocities"], cold["velocities"]):
                raise RuntimeError("Backend did not repeat its own seeded trajectory exactly")
            full_samples.append(sample["durations"])
        result = {
            "status": "passed",
            "execution": cold["manifest"]["execution"],
            "cold_run": cold["durations"],
            "warm_full_run": {
                key: _summary([item[key] for item in full_samples]) for key in full_samples[0]
            },
            "warm_profile": {
                key: _summary([item[key] for item in profile_samples]) for key in profile_samples[0]
            },
            "validation": {
                "reference": "numpy",
                **TOLERANCE,
                "max_absolute_velocity_error": max_error,
                "same_backend_repeatability": "exact final arrays",
                "final_array_sha256": hashlib.sha256(cold["velocities"].tobytes()).hexdigest(),
            },
            "environment": cold["manifest"]["environment"],
            "peak_worker_rss_bytes": _peak_rss_bytes(),
        }
    return result


def _cpu_metadata() -> dict[str, Any]:
    model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    affinity = getattr(os, "sched_getaffinity", None)
    quota_cores = None
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    if quota_file.is_file():
        quota, period = quota_file.read_text().split()
        if quota != "max":
            quota_cores = int(quota) / int(period)
    return {
        "cpu_model": model,
        "cpu_affinity_count": len(affinity(0)) if affinity else None,
        "cpu_quota_cores": quota_cores,
    }


def run_benchmark(
    *,
    particles: int = 2000,
    steps: int = 100,
    repeats: int = 5,
    seed: int = 42,
    temperature: float = 300.0,
    time_step: float = 0.001,
    backends: tuple[str, ...] = ("numpy", "numba"),
) -> dict[str, Any]:
    parameters = SimulationParameters(temperature, particles, time_step, steps, seed)
    if type(repeats) is not int or not 1 <= repeats <= 30:
        raise ValueError("repeats must be an integer in 1..30")
    if not backends or len(set(backends)) != len(backends) or set(backends) - {"numpy", "numba"}:
        raise ValueError("Choose unique runtime backends from numpy,numba")
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "production-runtime",
        "model_version": MODEL_VERSION,
        "workload": {
            **{key: value for key, value in asdict(parameters).items() if key != "backend"},
            "repeats": repeats,
            "particle_steps_per_run": particles * steps,
            "velocity_array_bytes": particles * 2 * 8,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            **_cpu_metadata(),
            "numpy": np.__version__,
            "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "thread_limits": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        },
        "methodology": {
            "isolation": "One fresh child process per backend; no concurrent benchmark workloads",
            "cold_run": "First service call in that process, including optional import/JIT; common Python/NumPy/service imports and interpreter startup excluded",
            "warm_full_run": "Real run_simulation including PCG64, evolution, final observables, checksums, environment capture, compressed NPZ/manifest and fsync; no tracking",
            "warm_profile": "Separate initialized engine runs with per-stage timers, not a decomposition of full-run samples; RNG-only is a separate diagnostic, not additive",
            "validation": "Persisted arrays and final observables checked outside timing; every warm run repeats the same seed exactly; numerical cross-backend tolerance is not a physical validation",
            "memory": "No pre-generated steps-by-particles noise; peak RSS is whole child process including imports/reference/validation; not incremental engine allocation",
            "excluded": "HTTP, queue wait, Redis, SSE, tracking, remote storage, child startup and artifact reads/numerical validation",
            "interpretation": "Local measurements on one CPU; skipped backends have no performance result; p95 over small samples is descriptive",
        },
        "backends": {},
    }
    for name in backends:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_measure-backend",
            name,
            "--particles",
            str(particles),
            "--steps",
            str(steps),
            "--repeats",
            str(repeats),
            "--seed",
            str(seed),
            "--temperature",
            str(temperature),
            "--time-step",
            str(time_step),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        if process.returncode:
            raise RuntimeError(f"{name} runtime measurement failed: {process.stderr[-4000:]}")
        try:
            report["backends"][name] = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{name} runtime measurement did not return JSON") from error
    numpy_result = report["backends"].get("numpy", {})
    if numpy_result.get("status") == "passed":
        for result in report["backends"].values():
            if result["status"] == "passed":
                result["measured_speedup_over_numpy"] = {
                    "warm_full_run": numpy_result["warm_full_run"]["wall_seconds"]["median_seconds"]
                    / result["warm_full_run"]["wall_seconds"]["median_seconds"],
                    "warm_evolution": numpy_result["warm_profile"][
                        "evolution_including_rng_seconds"
                    ]["median_seconds"]
                    / result["warm_profile"]["evolution_including_rng_seconds"]["median_seconds"],
                }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--backends", default="numpy,numba")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-backends",
        action="store_true",
        help="Fail when a requested optional backend is unavailable",
    )
    parser.add_argument("--_measure-backend", choices=("numpy", "numba"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    options = {
        "particles": args.particles,
        "steps": args.steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "temperature": args.temperature,
        "time_step": args.time_step,
    }
    try:
        if args._measure_backend:
            parameters = SimulationParameters(
                args.temperature,
                args.particles,
                args.time_step,
                args.steps,
                args.seed,
                args._measure_backend,
            )
            print(json.dumps(_measure_backend(parameters, args.repeats), allow_nan=False))
            return
        if args.output is None:
            parser.error("--output is required")
        if args.output.exists():
            parser.error("output already exists; choose a new report path")
        report = run_benchmark(
            **options, backends=tuple(name.strip() for name in args.backends.split(","))
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Runtime benchmark failed: {error}\n")
    skipped = False
    for name, result in report["backends"].items():
        if result["status"] == "passed":
            print(
                f"{name}: cold={result['cold_run']['wall_seconds']:.6f}s "
                f"warm full={result['warm_full_run']['wall_seconds']['median_seconds']:.6f}s "
                f"warm evolution={result['warm_profile']['evolution_including_rng_seconds']['median_seconds']:.6f}s"
            )
        else:
            skipped = True
            print(f"{name}: skipped ({result['reason']})")
    print(f"Runtime benchmark report: {args.output.resolve()}")
    if skipped and args.require_backends:
        parser.exit(1, "A requested runtime backend was skipped\n")


if __name__ == "__main__":
    main()
