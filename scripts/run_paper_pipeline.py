"""Generate a reproducible dataset, uncertainty figure, and checksummed artifacts.

Example: python scripts/run_paper_pipeline.py --output-dir /tmp/paper-run --seed 42
Install the ``research`` extra to enable plotting. No notebook runtime is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_paper_artifacts(
    output_dir: Path,
    *,
    seed: int = 0,
    num_points: int = 12,
    num_particles: int = 2000,
    num_steps: int = 100,
    time_step: float = 0.001,
    temperature_start: float = 100.0,
    temperature_end: float = 1000.0,
) -> Path:
    """Write a new output directory; preserve existing artifacts and source files."""
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install scientific-api[research] to generate figures") from exc

    from scientific_api.domain.simulation import SimulationParameters
    from scientific_api.services.simulation_service import run_simulation

    if not 1 <= num_points <= 1000:
        raise ValueError("num_points must be between 1 and 1000")
    if not 0 <= seed <= 2**64 - num_points:
        raise ValueError("seed and derived per-point seeds must fit unsigned 64-bit integers")
    temperatures = np.linspace(temperature_start, temperature_end, num_points)
    parameters = [
        SimulationParameters(
            initial_temperature=float(temperature),
            num_particles=num_particles,
            time_step=time_step,
            num_steps=num_steps,
            seed=seed + index,
        )
        for index, temperature in enumerate(temperatures)
    ]
    if num_points * num_particles * num_steps > 100_000_000:
        raise ValueError("The complete sweep must not exceed 100 million particle steps")

    output_dir = output_dir.resolve()
    source_root = Path(__file__).resolve().parent.parent
    for source_directory in (source_root / "scripts", source_root / "src"):
        if output_dir.is_relative_to(source_directory):
            raise ValueError("Choose an output directory outside scripts/ and src/")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty; existing artifacts are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, parameter in enumerate(parameters):
        task_id = f"point_{index:04d}_seed_{parameter.seed}"
        result = run_simulation(parameter, task_id=task_id, artifacts_dir=output_dir / "runs")
        rows.append({"parameters": asdict(parameter), "result": result})

    dataset = {
        "schema_version": 1,
        "experiment": "ideal_gas_equilibrium_temperature_sweep",
        "seed_policy": "point_seed = base_seed + point_index",
        "base_seed": seed,
        "rows": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    final_temperatures = [row["result"]["final_temperature"] for row in rows]
    errors = [row["result"]["temperature_standard_error"] for row in rows]
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.errorbar(
        temperatures,
        final_temperatures,
        yerr=errors,
        fmt="o",
        capsize=3,
        label="Final temperature ± model-predicted standard error",
    )
    axes.plot(temperatures, temperatures, "--", label="Equilibrium target")
    axes.set(
        xlabel="Target temperature (reduced units)",
        ylabel="Final temperature (reduced units)",
        title=f"OU thermostat equilibrium (N={num_particles}, steps={num_steps})",
    )
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "figure_1_temperature.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "dataset": "results.json",
        "figure": "figure_1_temperature.png",
        "uncertainty": (
            "Model-predicted equilibrium standard error T_target/sqrt(N) for a 2D Maxwellian snapshot; "
            "excludes model discrepancy and trajectory uncertainty"
        ),
        "reproducibility": (
            "Compare numerical results and per-run provenance in the same locked environment. "
            "PNG rendering and archive hashes need not match across platforms."
        ),
        "artifacts": [
            {"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=12)
    parser.add_argument("--num-particles", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--temperature-start", type=float, default=100.0)
    parser.add_argument("--temperature-end", type=float, default=1000.0)
    args = parser.parse_args()
    try:
        manifest = generate_paper_artifacts(**vars(args))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Paper pipeline failed: {exc}\n")
    print(f"Paper artifacts and SHA-256 manifest: {manifest}")


if __name__ == "__main__":
    main()
