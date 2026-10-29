"""Independent canonical reference for three interacting 2D LJ particles.

The reference integrates the four relative coordinates after fixing particle
one at the origin by translation invariance. It does not import the production
potential, force, neighbour list or integrator. Midpoint-grid differences are
resolution estimates, not rigorous error bounds. The optional MD comparison
uses independent replica means, never a standard error from correlated frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PHYSICS = {
    "model_version": "lennard-jones-2d-v1",
    "dimensions": 2,
    "num_particles": 3,
    "box_size": [6.0, 6.0],
    "epsilon": 1.0,
    "sigma": 1.0,
    "mass": 1.0,
    "bath_temperature": 1.0,
    "cutoff_radius": 2.5,
    "cutoff_policy": "shifted-force",
    "boltzmann_constant": 1.0,
}


def _independent_pair(radius_squared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bare analytic formula and its radial derivative, with the stated cutoff."""
    cutoff = PHYSICS["cutoff_radius"]
    potential = np.zeros_like(radius_squared)
    virial = np.zeros_like(radius_squared)
    active = (radius_squared > 0) & (radius_squared < cutoff**2)
    radius = np.sqrt(radius_squared[active])
    inverse_sixth = radius_squared[active] ** -3
    cutoff_potential = 4 * (cutoff**-12 - cutoff**-6)
    cutoff_radial_force = 24 * (2 * cutoff**-13 - cutoff**-7)
    potential[active] = (
        4 * (inverse_sixth**2 - inverse_sixth)
        - cutoff_potential
        + (radius - cutoff) * cutoff_radial_force
    )
    virial[active] = 24 * (2 * inverse_sixth**2 - inverse_sixth) - radius * cutoff_radial_force
    # Exactly coincident quadrature sites have zero Boltzmann weight.
    potential[radius_squared == 0] = np.inf
    return potential, virial


def midpoint_reference(grid: int) -> dict[str, Any]:
    """Integrate 4D periodic coordinates in bounded 64-by-grid² chunks."""
    if type(grid) is not int or not 8 <= grid <= 160:
        raise ValueError("grid must be an integer in 8..160")
    length = PHYSICS["box_size"][0]
    axis = (np.arange(grid) + 0.5) * length / grid - length / 2
    coordinates = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    origin_u, origin_w = _independent_pair(np.einsum("ij,ij->i", coordinates, coordinates))
    normalization = weighted_u = weighted_u2 = weighted_w = 0.0
    for start in range(0, len(coordinates), 64):
        differences = coordinates[start : start + 64, None, :] - coordinates[None, :, :]
        differences -= length * np.rint(differences / length)
        energy, virial = _independent_pair(np.einsum("ijk,ijk->ij", differences, differences))
        energy += origin_u[start : start + 64, None] + origin_u[None, :]
        virial += origin_w[start : start + 64, None] + origin_w[None, :]
        weights = np.exp(-energy / PHYSICS["bath_temperature"])
        finite = np.isfinite(energy)
        normalization += float(np.sum(weights))
        weighted_u += float(np.sum(energy[finite] * weights[finite]))
        weighted_u2 += float(np.sum(energy[finite] ** 2 * weights[finite]))
        weighted_w += float(np.sum(virial * weights))
    mean_u = weighted_u / normalization
    return {
        "grid_per_coordinate": grid,
        "relative_coordinate_points": grid**4,
        "potential_energy": mean_u,
        "potential_energy_variance": weighted_u2 / normalization - mean_u**2,
        "virial": weighted_w / normalization,
    }


def generate_reference(grids: tuple[int, ...] = (64, 96, 128)) -> dict[str, Any]:
    if len(grids) < 3 or any(left >= right for left, right in zip(grids, grids[1:])):
        raise ValueError("Use at least three strictly increasing grid sizes")
    measurements = [midpoint_reference(grid) for grid in grids]
    estimates = {}
    for observable, floor in (("potential_energy", 0.001), ("virial", 0.01)):
        changes = [
            abs(right[observable] - left[observable])
            for left, right in zip(measurements[-3:-1], measurements[-2:])
        ]
        estimates[observable] = {
            "value": measurements[-1][observable],
            "successive_grid_differences": changes,
            "absolute_resolution_allowance": max(floor, 4 * max(changes)),
        }
    return {
        "schema_version": 1,
        "physics": PHYSICS,
        "method": "Periodic 4D midpoint quadrature with particle one fixed at the origin",
        "normalization": "Boltzmann exp(-U/T); translational volume and quadrature volume cancel in ratios",
        "resolution_interpretation": "Successive-grid differences and a conservative allowance, not a rigorous bound",
        "limitations": [
            "Three particles at one specified 2D state point with a shifted-force cutoff",
            "No long-range tail correction or 3D fluid reference is imported",
            "Production overlap rejection below 0.5 sigma has negligible canonical weight at T=1",
            "Agreement does not establish equilibration for every density, temperature or timestep",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source": {
            "path": "scripts/validate_lj_reference.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "production_imports": False,
        },
        "grids": measurements,
        "estimates": estimates,
    }


def validate_md(reference: dict[str, Any], *, replicas: int = 12) -> dict[str, Any]:
    """Compare interacting BAOAB samples with independent position quadrature."""
    from scientific_api.domain.lennard_jones import LennardJonesParameters
    from scientific_api.models.lennard_jones_engine import create_lj_simulator

    if reference["physics"] != PHYSICS:
        raise ValueError("The reference does not describe the supported physical state point")
    if type(replicas) is not int or not 4 <= replicas <= 32:
        raise ValueError("replicas must be an integer in 4..32")
    replica_means = []
    for index in range(replicas):
        parameters = LennardJonesParameters(
            num_particles=3,
            box_size=(6.0, 6.0),
            initial_temperature=1.0,
            ensemble="nvt",
            bath_temperature=1.0,
            friction=1.0,
            time_step=0.005,
            num_steps=10_000,
            equilibration_steps=4000,
            sample_stride=20,
            seed=1729 + index,
            interaction_backend="direct",
            backend="numba",
        )
        simulator = create_lj_simulator(parameters)
        observations = []
        for step in range(1, parameters.num_steps + 1):
            simulator.advance()
            if step >= parameters.equilibration_steps and step % parameters.sample_stride == 0:
                observations.append(simulator.observables())
        replica_means.append(
            {
                "seed": parameters.seed,
                "samples": len(observations),
                **{
                    key: float(np.mean([item[key] for item in observations]))
                    for key in (
                        "potential_energy",
                        "virial",
                        "kinetic_energy",
                        "thermal_temperature",
                        "pressure",
                    )
                },
            }
        )
    checks = {}
    expected = {
        "potential_energy": reference["estimates"]["potential_energy"]["value"],
        "virial": reference["estimates"]["virial"]["value"],
        "kinetic_energy": 3.0,
        "thermal_temperature": 1.0,
        "pressure": (6 + reference["estimates"]["virial"]["value"]) / 72,
    }
    for observable, target in expected.items():
        values = np.asarray([item[observable] for item in replica_means])
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / math.sqrt(replicas))
        resolution = (
            reference["estimates"].get(observable, {}).get("absolute_resolution_allowance", 0)
        )
        if observable == "pressure":
            resolution = reference["estimates"]["virial"]["absolute_resolution_allowance"] / 72
        # A small explicit finite-dt allowance is not claimed as an error bound.
        scale = (
            1 / 36 if observable == "pressure" else 1 if observable == "thermal_temperature" else 3
        )
        timestep_allowance = 8 * parameters.time_step**2 * scale
        tolerance = 6 * standard_error + resolution + timestep_allowance
        checks[observable] = {
            "mean": mean,
            "reference": target,
            "replica_standard_error": standard_error,
            "reference_resolution_allowance": resolution,
            "finite_timestep_allowance": timestep_allowance,
            "acceptance_tolerance": tolerance,
            "passed": abs(mean - target) <= tolerance,
        }
    return {
        "schema_version": 1,
        "physics": PHYSICS,
        "parameters": {
            "backend": "numba",
            "interaction_backend": "direct",
            "integrator": "baoab",
            "time_step": 0.005,
            "num_steps": 10_000,
            "equilibration_steps": 4000,
            "sample_stride": 20,
            "replicas": replicas,
            "seed_policy": "1729 + replica_index",
        },
        "uncertainty": "Sample SD of independent replica means divided by sqrt(replicas); time frames are not independent samples",
        "acceptance": "Six replica standard errors plus reference resolution and stated finite-timestep allowances; a consistency check, not a confidence interval",
        "checks": checks,
        "replica_means": replica_means,
        "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids", default="64,96,128")
    parser.add_argument("--reference", type=Path, help="Validate MD against an existing reference")
    parser.add_argument("--replicas", type=int, default=12)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output already exists; choose a new path")
    try:
        if arguments.reference:
            report = validate_md(
                json.loads(arguments.reference.read_text()), replicas=arguments.replicas
            )
        else:
            report = generate_reference(tuple(int(value) for value in arguments.grids.split(",")))
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"LJ reference validation failed: {error}\n")
    print(f"LJ canonical report: {arguments.output.resolve()}")
    if report.get("status") == "failed":
        parser.exit(1, "Interacting canonical validation failed; inspect the report\n")


if __name__ == "__main__":
    main()
