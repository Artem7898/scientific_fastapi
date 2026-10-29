"""Measure an identical seeded OU workload across optional execution backends.

Example: python scripts/benchmark.py --backends numpy,threads,processes --output /tmp/benchmark.json
Numba and JAX are loaded only when explicitly requested. Input generation and
reference computation are outside timed samples; process transfer and JAX device
synchronization are included. Compilation and worker startup are reported as warmup.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import sys
import sysconfig
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Respect explicit user limits and use one native thread per Python worker otherwise.
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

AVAILABLE_BACKENDS = {"numpy", "threads", "processes", "numba", "jax"}
MAX_INPUT_BYTES = 256 * 1024 * 1024


def runtime_gil_enabled() -> bool:
    query = getattr(sys, "_is_gil_enabled", None)
    return bool(query()) if query else True


def numpy_kernel(arguments: tuple[np.ndarray, np.ndarray, float, float]) -> np.ndarray:
    initial, shocks, decay, amplitude = arguments
    velocities = initial.copy()
    for shock in shocks:
        velocities *= decay
        velocities += amplitude * shock
    return velocities


def validate_outputs(outputs: list[np.ndarray], reference: list[np.ndarray]) -> float:
    if len(outputs) != len(reference):
        raise RuntimeError("Backend returned an incorrect number of independent batches")
    maximum_error = 0.0
    for output, expected in zip(outputs, reference, strict=True):
        if output.shape != expected.shape or not np.isfinite(output).all():
            raise RuntimeError("Backend returned an invalid final state")
        if not np.allclose(output, expected, rtol=1e-11, atol=1e-11):
            raise RuntimeError("Backend failed numerical equivalence against NumPy")
        maximum_error = max(maximum_error, float(np.max(np.abs(output - expected))))
    return maximum_error


def benchmark_backend(
    name: str,
    batches: list[tuple[np.ndarray, np.ndarray, float, float]],
    reference: list[np.ndarray],
    repeats: int,
    workers: int,
) -> dict[str, Any]:
    executor = None
    backend_metadata: dict[str, Any] = {}
    warmup_start = time.perf_counter()
    try:
        if name == "numpy":

            def execute() -> list[np.ndarray]:
                return [numpy_kernel(batch) for batch in batches]
        elif name in {"threads", "processes"}:
            if name == "threads":
                executor = ThreadPoolExecutor(max_workers=workers)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                )

            def execute() -> list[np.ndarray]:
                assert executor is not None
                return list(executor.map(numpy_kernel, batches))
        elif name == "numba":
            try:
                import numba
            except ImportError:
                return {
                    "status": "skipped",
                    "reason": "Install scientific-api[numba] or scientific-api[benchmark] for Numba",
                }

            from scientific_api.models.numba_engine import update_velocities

            def compiled(initial, shocks, decay, amplitude):
                # Reuse the production single-step kernel. Input generation is
                # still deliberately excluded from this kernel-only benchmark.
                velocities = initial.copy()
                for shock in shocks:
                    update_velocities(velocities, shock, decay, amplitude)
                return velocities

            def execute() -> list[np.ndarray]:
                return [compiled(*batch) for batch in batches]

            backend_metadata["version"] = numba.__version__
            backend_metadata["fastmath"] = False
            backend_metadata["kernel"] = "production ou-numba-v1 single-step update"
            backend_metadata["device_platforms"] = ["cpu"]
        elif name == "jax":
            try:
                import jax
                from jax import numpy as jnp
            except ImportError:
                return {"status": "skipped", "reason": "Install scientific-api[benchmark] for JAX"}
            jax.config.update("jax_enable_x64", True)

            @jax.jit
            def compiled(initial, shocks, decay, amplitude):
                def step(velocities, shock):
                    return decay * velocities + amplitude * shock, None

                return jax.lax.scan(step, initial, shocks)[0]

            def execute() -> list[np.ndarray]:
                return [
                    np.asarray(compiled(*(jnp.asarray(item) for item in batch)).block_until_ready())
                    for batch in batches
                ]

            backend_metadata["version"] = jax.__version__
            backend_metadata["devices"] = [str(device) for device in jax.devices()]
            backend_metadata["device_platforms"] = sorted(
                {device.platform for device in jax.devices()}
            )
            backend_metadata["gpu_available"] = any(
                device.platform == "gpu" for device in jax.devices()
            )
            backend_metadata["scope"] = (
                "experimental pre-generated kernel; not a production runtime backend"
            )
            backend_metadata["float64"] = True
        else:
            raise ValueError(f"Unknown backend: {name}")

        warmup_outputs = execute()
        warmup_seconds = time.perf_counter() - warmup_start
        maximum_error = validate_outputs(warmup_outputs, reference)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            outputs = execute()
            samples.append(time.perf_counter() - start)
            maximum_error = max(maximum_error, validate_outputs(outputs, reference))
        return {
            "status": "passed",
            "gil_enabled": runtime_gil_enabled(),
            "warmup_seconds": warmup_seconds,
            "samples_seconds": samples,
            "median_seconds": float(np.median(samples)),
            "p95_seconds": float(np.percentile(samples, 95)),
            "max_absolute_error": maximum_error,
            "validation": {"reference": "numpy", "rtol": 1e-11, "atol": 1e-11},
            **backend_metadata,
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def run_benchmark(
    *,
    particles: int = 2000,
    steps: int = 100,
    repeats: int = 5,
    seed: int = 42,
    workers: int = 2,
    backends: tuple[str, ...] = ("numpy", "threads", "processes"),
) -> dict[str, Any]:
    if not 1 <= particles <= 1_000_000 or not 1 <= steps <= 10_000:
        raise ValueError("particles must be 1..1000000 and steps must be 1..10000")
    if not 1 <= repeats <= 100 or not 1 <= workers <= 32:
        raise ValueError("repeats must be 1..100 and workers must be 1..32")
    if not backends or len(set(backends)) != len(backends) or set(backends) - AVAILABLE_BACKENDS:
        raise ValueError("Choose unique backends from numpy,threads,processes,numba,jax")
    if not 0 <= seed <= 2**64 - workers:
        raise ValueError("seed and per-batch seeds must fit unsigned 64-bit integers")
    input_bytes = workers * (steps + 1) * particles * 2 * 8
    if input_bytes > MAX_INPUT_BYTES:
        raise ValueError(
            "Pre-generated inputs exceed the 256 MiB limit; reduce particles, steps, or workers"
        )

    temperature, time_step = 300.0, 0.001
    decay = math.exp(-time_step)
    amplitude = math.sqrt(-math.expm1(-2.0 * time_step) * temperature)
    batches = []
    for index in range(workers):
        rng = np.random.Generator(np.random.PCG64(seed + index))
        values = rng.standard_normal((steps + 1, particles, 2), dtype=np.float64)
        initial = values[0] * math.sqrt(temperature)
        batches.append((initial, values[1:], decay, amplitude))
    reference = [numpy_kernel(batch) for batch in batches]

    report: dict[str, Any] = {
        "schema_version": 1,
        "model_version": "ideal-gas-ou-v1",
        "workload": {
            "particles_per_batch": particles,
            "steps": steps,
            "independent_batches": workers,
            "particle_steps_per_sample": particles * steps * workers,
            "input_bytes": input_bytes,
            "temperature": temperature,
            "time_step": time_step,
            "seed": seed,
            "rng": "PCG64",
            "batch_seed_policy": "seed + batch_index",
            "dtype": "float64",
            "repeats": repeats,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "gil_enabled": runtime_gil_enabled(),
            "thread_limits": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        },
        "methodology": {
            "timed": "identical pre-generated inputs; process IPC and JAX device transfers/synchronization included",
            "excluded": "input generation, NumPy reference, numerical validation, warmup/compilation, worker shutdown",
            "parallelism": "NumPy, Numba and JAX process batches serially; threads/processes use workers concurrent batches",
            "interpretation": "Local measurements only; skipped backends are not validated and are not performance results",
        },
        "backends": {},
    }
    try:
        report["environment"]["package_version"] = importlib.metadata.version("scientific-api")
    except importlib.metadata.PackageNotFoundError:
        report["environment"]["package_version"] = "not-installed"

    for name in backends:
        report["backends"][name] = benchmark_backend(name, batches, reference, repeats, workers)
    # Optional native extensions can enable the GIL on a free-threaded build.
    report["environment"]["gil_enabled_after_backends"] = runtime_gil_enabled()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backends", default="numpy,threads,processes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        parser.error("output already exists; choose a new report path")
    try:
        report = run_benchmark(
            particles=args.particles,
            steps=args.steps,
            repeats=args.repeats,
            seed=args.seed,
            workers=args.workers,
            backends=tuple(name.strip() for name in args.backends.split(",")),
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, result in report["backends"].items():
        if result["status"] == "passed":
            print(
                f"{name}: median={result['median_seconds']:.6f}s p95={result['p95_seconds']:.6f}s"
            )
        else:
            print(f"{name}: skipped ({result['reason']})")
    print(f"Benchmark report: {output.resolve()}")


if __name__ == "__main__":
    main()
