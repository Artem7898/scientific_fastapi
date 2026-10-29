"""Descriptive 2D LJ observables and a finite-N terminal radial distribution.

Time frames from one interacting trajectory are correlated. This module does
not turn their sample standard deviation into an independent-sample SE, apply
free-particle diffusion formulas, or claim that a run is equilibrated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters

ANALYSIS_VERSION = "lj-observables-2d-v1"
RDF_VERSION = "lj-rdf-2d-v1"
SERIES_NAMES = (
    "thermal_temperature",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "pressure",
    "virial",
)
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024


def _positions(values: NDArray[np.float64]) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 2
        or values.shape[1] != 2
        or not 3 <= values.shape[0] <= 4096
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite native float64 coordinates with shape (N, 2), N=3..4096")


def radial_distribution(
    positions: NDArray[np.float64], *, box_size: tuple[float, float], bins: int = 64
) -> dict[str, Any]:
    """Describe one periodic snapshot using unordered pairs and 2D annular area.

    The finite-particle denominator uses N*(N-1), rather than N*N. Limiting r
    to half the shortest box side keeps every annulus inside the minimum-image
    rectangle. Pair rows are processed separately, without an N by N array.
    The caller reserves the reported pair checks before invoking this analysis.
    """
    _positions(positions)
    if (
        not isinstance(box_size, (tuple, list))
        or len(box_size) != 2
        or any(
            isinstance(length, bool)
            or not isinstance(length, (int, float))
            or not math.isfinite(length)
            or length <= 0
            for length in box_size
        )
    ):
        raise ValueError("box_size must contain two finite positive lengths")
    if type(bins) is not int or not 8 <= bins <= 256:
        raise ValueError("bins must be an integer in 8..256")
    box = np.asarray(box_size, dtype=np.float64)
    particles = len(positions)
    maximum_radius = float(np.min(box)) / 2.0
    edges = np.linspace(0.0, maximum_radius, bins + 1, dtype=np.float64)
    if not (np.diff(edges) > 0).all():
        raise ValueError("RDF bins cannot be represented by positive finite widths")
    counts = np.zeros(bins, dtype=np.int64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for particle in range(particles - 1):
                displacement = positions[particle] - positions[particle + 1 :]
                displacement -= box * np.rint(displacement / box)
                radius = np.hypot(displacement[:, 0], displacement[:, 1])
                counts += np.histogram(radius[radius <= maximum_radius], bins=edges)[0]
            area = float(np.prod(box))
            pair_checks = particles * (particles - 1) // 2
            expected_counts = pair_checks * math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / area
            distribution = counts / expected_counts
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("RDF cannot be represented as finite float64 values") from error
    if not np.isfinite(distribution).all():
        raise ValueError("RDF normalization is not finite")
    return {
        "analysis_version": RDF_VERSION,
        "dimensions": 2,
        "sample_size": particles,
        "snapshot": "terminal frame only; no equilibration or time-averaging claim",
        "distance_convention": "periodic minimum image",
        "maximum_radius": maximum_radius,
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "expected_uniform_counts": expected_counts.tolist(),
        "g_r": distribution.tolist(),
        "normalization": "unordered pairs / [N*(N-1)/2 * pi*(r_outer^2-r_inner^2)/area]",
        "pair_checks": pair_checks,
        "limitations": [
            "A single correlated interacting snapshot does not establish canonical equilibrium",
            "No uncertainty estimate is inferred from RDF bins or particle pairs",
            "Two-dimensional geometry; no three-dimensional tail correction is applied",
        ],
    }


def analyze_lj_observables(
    positions: NDArray[np.float64],
    sample_steps: NDArray[np.int64],
    observable_series: Mapping[str, NDArray[np.float64]],
    *,
    parameters: LennardJonesParameters,
) -> dict[str, Any]:
    """Reanalyze saved sampled observables and terminal coordinates, without dynamics."""
    if not isinstance(parameters, LennardJonesParameters):
        raise TypeError("parameters must be LennardJonesParameters")
    expected_steps = np.asarray(parameters.sampling_steps(), dtype=np.int64)
    if (
        not isinstance(sample_steps, np.ndarray)
        or sample_steps.dtype != np.dtype(np.int64)
        or not np.array_equal(sample_steps, expected_steps)
    ):
        raise ValueError("sample_steps must match the declared int64 sampling schedule")
    frames = len(expected_steps)
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.dtype(np.float64)
        or positions.shape != (frames, parameters.num_particles, 2)
        or positions.nbytes > MAX_TRAJECTORY_BYTES
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Expected a bounded finite float64 trajectory with shape (frames,N,2)")
    if not isinstance(observable_series, Mapping) or set(observable_series) != set(SERIES_NAMES):
        raise ValueError("Observable series must contain exactly the six supported observables")
    series = {}
    for name in SERIES_NAMES:
        values = observable_series[name]
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(np.float64)
            or values.shape != (frames,)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{name} must be a finite float64 sampled series")
        if name in {"thermal_temperature", "kinetic_energy"} and (values < 0).any():
            raise ValueError(f"{name} must not be negative")
        series[name] = values
    scale = parameters.epsilon * parameters.num_particles
    if not np.allclose(
        series["kinetic_energy"] + series["potential_energy"],
        series["total_energy"],
        rtol=1e-12,
        atol=1e-12 * scale,
    ):
        raise ValueError("Total energy disagrees with kinetic plus potential energy")
    pressure = (2 * series["kinetic_energy"] + series["virial"]) / (
        2 * parameters.box_size[0] * parameters.box_size[1]
    )
    if not np.allclose(pressure, series["pressure"], rtol=1e-12, atol=1e-12 * scale):
        raise ValueError("Pressure disagrees with the two-dimensional kinetic/virial definition")
    selected = sample_steps >= parameters.equilibration_steps
    production = {}
    for name, values in series.items():
        samples = values[selected]
        mean = math.fsum(float(value) for value in samples) / len(samples)
        standard_deviation = (
            math.sqrt(
                math.fsum((float(value) - mean) ** 2 for value in samples) / (len(samples) - 1)
            )
            if len(samples) > 1
            else None
        )
        production[name] = {"mean": mean, "sample_standard_deviation": standard_deviation}
    initial_energy = float(series["total_energy"][0])
    reference_scale = max(abs(initial_energy), scale)
    changes = series["total_energy"] - initial_energy
    nve = parameters.ensemble == "nve"
    energy_drift = {
        "applicable": nve,
        "initial_total_energy": initial_energy,
        "reference_scale": reference_scale,
        "normalization": "max(abs(initial_total_energy), epsilon*num_particles)",
        "maximum_absolute_change": float(np.max(np.abs(changes))) if nve else None,
        "maximum_normalized_absolute_change": (
            float(np.max(np.abs(changes))) / reference_scale if nve else None
        ),
        "terminal_change": float(changes[-1]) if nve else None,
        "terminal_normalized_change": float(changes[-1]) / reference_scale if nve else None,
        "interpretation": (
            "Sampled NVE energy changes; this is not a time-step convergence test"
            if nve
            else "Not an NVE diagnostic: the NVT thermostat exchanges energy with the system"
        ),
    }
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_version": LJ_MODEL_VERSION,
        "dimensions": 2,
        "sample_size": parameters.num_particles,
        "ensemble": parameters.ensemble,
        "coordinate_convention": "unwrapped trajectory; periodic minimum-image RDF",
        "sample_steps": sample_steps.tolist(),
        "times": (sample_steps.astype(np.float64) * parameters.time_step).tolist(),
        "series": {name: values.tolist() for name, values in series.items()},
        "production": {
            "requested_equilibration_steps": parameters.equilibration_steps,
            "first_sample_step": int(sample_steps[selected][0]),
            "last_sample_step": int(sample_steps[selected][-1]),
            "sample_count": int(np.count_nonzero(selected)),
            "statistics": production,
            "method": "Saved-frame mean and sample SD; not duration-weighted; correlated frames have no independent-sample SE",
            "equilibration": "User-selected exclusion; no automatic equilibration test was performed",
        },
        "energy_drift": energy_drift,
        "rdf": radial_distribution(
            positions[-1], box_size=parameters.box_size, bins=parameters.rdf_bins
        ),
        "definitions": {
            "thermal_temperature": "mass*sum(|v-sample_mean(v)|^2)/(2*(N-1))",
            "kinetic_energy": "mass*sum(|v|^2)/2; includes center-of-mass motion",
            "potential_energy": "Sum of shifted-force pair potentials below the cutoff; signed",
            "total_energy": "kinetic_energy + potential_energy; signed",
            "pressure": "(2*kinetic_energy + virial)/(2*area); signed",
        },
        "limitations": [
            "Interacting particles and time frames are correlated; no free-gas standard error is reused",
            "No diffusion coefficient is inferred using the independent Langevin-particle formula",
            "The final RDF and descriptive time statistics do not establish equilibration",
            "No three-dimensional tail correction or dimensional experimental calibration is applied",
        ],
    }
