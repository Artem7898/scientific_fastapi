"""Compare the actual short-horizon LJ engines at equal density and seed.

Each variant gets a fresh process and an untimed NumPy/direct reference.
The first initialization/evolution includes optional import and compilation.
Warm measurements include initialization, neighbor construction/rebuilding,
integration and terminal observables. HTTP, queueing, RDF and artifact I/O
are outside this engine benchmark. Long chaotic trajectories are not expected
to remain numerically close across implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
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

from scientific_api.domain.lennard_jones import (  # noqa: E402
    LJ_MODEL_VERSION,
    LennardJonesParameters,
)
from scientific_api.models.computation import BackendUnavailable  # noqa: E402
from scientific_api.models.lennard_jones_engine import create_lj_simulator  # noqa: E402
from scientific_api.services.simulation_service import _environment  # noqa: E402

VARIANTS = ("numpy-direct", "numpy-verlet", "numba-direct", "numba-verlet")
TOLERANCE = {"rtol": 1e-10, "atol": 1e-10}


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": values,
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
    }


def _hardware() -> dict[str, Any]:
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    quota = None
    try:
        value, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if value != "max":
            quota = int(value) / int(period)
    except (OSError, ValueError):
        pass
    return {
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cgroup_cpu_quota_cores": quota,
        "platform": platform.platform(),
        "thread_limits": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _run(parameters: LennardJonesParameters) -> dict[str, Any]:
    start = time.perf_counter()
    simulator = create_lj_simulator(parameters)
    initialized = time.perf_counter()
    for _ in range(parameters.num_steps):
        simulator.advance()
    advanced = time.perf_counter()
    observables = simulator.observables()
    finished = time.perf_counter()
    return {
        "durations": {
            "initialization_seconds": initialized - start,
            "evolution_seconds": advanced - initialized,
            "terminal_observables_seconds": finished - advanced,
            "wall_seconds": finished - start,
        },
        "positions": simulator.unwrapped_positions.copy(),
        "velocities": simulator.velocities.copy(),
        "forces": simulator.forces.copy(),
        "observables": observables,
        "diagnostics": simulator.diagnostics(),
        "execution": simulator.execution_metadata(),
    }


def _validate(sample: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    errors = {}
    for name in ("positions", "velocities", "forces"):
        actual, expected = sample[name], reference[name]
        if actual.shape != expected.shape or actual.dtype != np.float64:
            raise RuntimeError(f"{name} violates the final state contract")
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, **TOLERANCE):
            raise RuntimeError(f"Short-horizon {name} differs from the NumPy/direct reference")
        errors[name] = float(np.max(np.abs(actual - expected)))
    if sample["observables"].keys() != reference["observables"].keys():
        raise RuntimeError("Observable names differ from the reference")
    errors["observables"] = 0.0
    for name, expected in reference["observables"].items():
        actual = sample["observables"][name]
        if not math.isfinite(actual) or not np.isclose(actual, expected, **TOLERANCE):
            raise RuntimeError(f"Short-horizon {name} differs from the reference")
        errors["observables"] = max(errors["observables"], abs(actual - expected))
    return errors


def _peak_rss() -> int | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


def _measure(parameters: LennardJonesParameters, repeats: int) -> dict[str, Any]:
    try:
        cold = _run(parameters)
    except BackendUnavailable as error:
        return {"status": "skipped", "reason": str(error)}
    reference_parameters = LennardJonesParameters(
        **{**asdict(parameters), "backend": "numpy", "interaction_backend": "direct"}
    )
    reference = _run(reference_parameters)
    maximum_errors = _validate(cold, reference)
    samples = []
    for _ in range(repeats):
        sample = _run(parameters)
        errors = _validate(sample, reference)
        maximum_errors = {name: max(value, errors[name]) for name, value in maximum_errors.items()}
        for name in ("positions", "velocities", "forces"):
            if not np.array_equal(sample[name], cold[name]):
                raise RuntimeError("A variant did not exactly repeat its own seeded state")
        if sample["observables"] != cold["observables"]:
            raise RuntimeError("A variant did not exactly repeat its own observables")
        samples.append(sample["durations"])
    return {
        "status": "passed",
        "execution": cold["execution"],
        "cold_call": cold["durations"],
        "warm_call": {name: _summary([item[name] for item in samples]) for name in samples[0]},
        "diagnostics": cold["diagnostics"],
        "final_observables": cold["observables"],
        "validation": {
            "reference": "numpy-direct",
            **TOLERANCE,
            "max_absolute_errors": maximum_errors,
            "same_variant_repeatability": "exact final arrays and observables",
            "state_sha256": {
                name: hashlib.sha256(cold[name].tobytes()).hexdigest()
                for name in ("positions", "velocities", "forces")
            },
        },
        "environment": {
            **_environment(),
            "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "gil_enabled_after_backend": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        },
        "peak_process_rss_bytes": _peak_rss(),
    }


def run_benchmark(
    *,
    particles: int = 64,
    steps: int = 20,
    repeats: int = 5,
    seed: int = 42,
    density: float = 4 / 9,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer in 1..100 for short-horizon comparisons")
    if type(particles) is not int or not 36 <= particles <= 2048:
        raise ValueError("particles must be an integer in 36..2048")
    if type(repeats) is not int or not 1 <= repeats <= 30:
        raise ValueError("repeats must be an integer in 1..30")
    if isinstance(density, bool) or not isinstance(density, (float, int)) or not 0 < density <= 1:
        raise ValueError("density must be finite and in (0,1]")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("Choose unique variants from " + ",".join(VARIANTS))
    length = math.sqrt(particles / density)
    parameters = LennardJonesParameters(
        num_particles=particles,
        box_size=(length, length),
        num_steps=steps,
        time_step=0.002,
        seed=seed,
        interaction_backend="direct",
        sample_stride=steps,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "lj-engine",
        "model_version": LJ_MODEL_VERSION,
        "workload": {
            **{
                key: value
                for key, value in asdict(parameters).items()
                if key not in {"backend", "interaction_backend"}
            },
            "density": density,
            "repeats": repeats,
        },
        "hardware": _hardware(),
        "methodology": {
            "isolation": "One fresh subprocess per variant; variants measured serially",
            "cold_call": "First engine initialization, integration and terminal observables; optional import/JIT included; Python/common NumPy/package imports excluded",
            "warm_call": "New same-seed engine per sample; lattice/velocities, initial force, neighbor construction/rebuilding, integration and terminal observables included",
            "excluded": "HTTP, queue waiting, tracking, RDF, trajectory sampling, artifact I/O, reference generation, numerical validation and process startup",
            "reference": "NumPy direct force engine; identical initial parameters and PCG64 seed; only short-horizon numerical agreement is tested",
            "interpretation": "Local measurements at fixed density and timestep; long chaotic MD paths need not remain close across implementations; no universal speed or GPU claim",
            "memory": "Peak RSS covers imports, JIT, reference and validation state, not incremental neighbor-list allocation",
        },
        "variants": {},
    }
    for variant in variants:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_measure-variant",
            variant,
            "--particles",
            str(particles),
            "--steps",
            str(steps),
            "--repeats",
            str(repeats),
            "--seed",
            str(seed),
            "--density",
            str(density),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        if process.returncode:
            raise RuntimeError(f"{variant} failed: {process.stderr[-4000:]}")
        try:
            report["variants"][variant] = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{variant} did not return JSON") from error
    baseline = report["variants"].get("numpy-direct", {})
    if baseline.get("status") == "passed":
        for result in report["variants"].values():
            if result["status"] == "passed":
                result["warm_speed_ratio_over_numpy_direct"] = (
                    baseline["warm_call"]["wall_seconds"]["median_seconds"]
                    / result["warm_call"]["wall_seconds"]["median_seconds"]
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=4 / 9)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-backends", action="store_true")
    parser.add_argument("--_measure-variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args._measure_variant:
            backend, interaction = args._measure_variant.split("-", 1)
            length = math.sqrt(args.particles / args.density)
            parameters = LennardJonesParameters(
                num_particles=args.particles,
                box_size=(length, length),
                num_steps=args.steps,
                time_step=0.002,
                seed=args.seed,
                backend=backend,
                interaction_backend=interaction,
                sample_stride=args.steps,
            )
            print(json.dumps(_measure(parameters, args.repeats), allow_nan=False))
            return
        if args.output is None:
            parser.error("--output is required")
        if args.output.exists():
            parser.error("output already exists; choose a new report path")
        report = run_benchmark(
            particles=args.particles,
            steps=args.steps,
            repeats=args.repeats,
            seed=args.seed,
            density=args.density,
            variants=tuple(name.strip() for name in args.variants.split(",")),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"LJ benchmark failed: {error}\n")
    skipped = False
    for variant, result in report["variants"].items():
        if result["status"] == "passed":
            print(
                f"{variant}: cold={result['cold_call']['wall_seconds']:.6f}s "
                f"warm={result['warm_call']['wall_seconds']['median_seconds']:.6f}s "
                f"pairs={result['diagnostics']['total_pair_checks']}"
            )
        else:
            skipped = True
            print(f"{variant}: skipped ({result['reason']})")
    print(f"LJ benchmark report: {args.output.resolve()}")
    if skipped and args.require_backends:
        parser.exit(1, "A requested optional backend was skipped\n")


if __name__ == "__main__":
    main()
