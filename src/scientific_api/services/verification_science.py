"""Independent bounded observable checks on already integrity-verified bytes.

Only domain contracts and NumPy are imported. These checks do not call the
production integrators, force implementations or research-analysis functions.
Passing verifies the stated deterministic identities, not equilibrium, model
adequacy, integration accuracy, authorship or stochastic equivalence.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from ..domain.experiments import ExperimentParameters
from ..domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from ..domain.simulation import MODEL_VERSION, SimulationCancelled, SimulationParameters

MAX_NPZ_BYTES = 128 * 1024 * 1024
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_PAIR_CHECKS = 2_000_000
NUMERICAL_RTOL = 1e-10
NUMERICAL_ATOL = 1e-11
SCIENCE_POLICY = "independent-observables-v1"
_SERIES = (
    "thermal_temperature",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "pressure",
    "virial",
)


class VerificationArtifactError(ValueError):
    """Saved numerical bytes do not meet their finite, bounded contract."""


def _cancel(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def decode_verification_npz(
    raw: bytes,
    contract: dict[str, tuple[tuple[int, ...], np.dtype]],
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    """Decode only allowlisted C-order arrays after a bounded header preflight.

    The same byte snapshot is preflighted and decoded. NPY object arrays,
    duplicate/nested members, encryption and optional compression codecs are
    rejected before calling NumPy's array allocator.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_NPZ_BYTES:
        raise VerificationArtifactError("NUMERICAL_ARTIFACT_TOO_LARGE")
    if not isinstance(contract, dict) or not 1 <= len(contract) <= 16:
        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_CONTRACT")
    expected: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    size = 0
    for name, (shape, raw_dtype) in contract.items():
        dtype = np.dtype(raw_dtype)
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(shape, tuple)
            or not shape
            or any(type(value) is not int or value < 0 for value in shape)
            or dtype not in {np.dtype(np.float64), np.dtype(np.int64)}
        ):
            raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_CONTRACT")
        size += math.prod(shape) * dtype.itemsize
        if size > MAX_DECODED_BYTES:
            raise VerificationArtifactError("NUMERICAL_ARTIFACT_TOO_LARGE")
        expected[name + ".npy"] = shape, dtype
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {item.filename for item in members} != set(
                expected
            ):
                raise VerificationArtifactError("INVALID_NUMERICAL_ARCHIVE_MEMBERS")
            for member in members:
                _cancel(check_cancelled)
                shape, dtype = expected[member.filename]
                array_bytes = math.prod(shape) * dtype.itemsize
                if (
                    member.flag_bits & 1
                    or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or member.file_size > array_bytes + MAX_HEADER_BYTES + 12
                ):
                    raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in {(1, 0), (2, 0)}:
                        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
                    width = 2 if version == (1, 0) else 4
                    length = stream.read(width)
                    if len(length) != width:
                        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
                    header_bytes = int.from_bytes(length, "little")
                    if (
                        not 0 < header_bytes <= MAX_HEADER_BYTES
                        or member.file_size != 8 + width + header_bytes + array_bytes
                    ):
                        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
                    header = stream.read(header_bytes)
                    if len(header) != header_bytes:
                        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    actual_shape, fortran, actual_dtype = reader(
                        io.BytesIO(length + header), max_header_size=MAX_HEADER_BYTES
                    )
                    if actual_shape != shape or fortran or actual_dtype != dtype:
                        raise VerificationArtifactError("INVALID_NUMERICAL_ARRAY_HEADER")
        result = {}
        with np.load(
            io.BytesIO(raw), allow_pickle=False, max_header_size=MAX_HEADER_BYTES
        ) as archive:
            for filename, (shape, dtype) in expected.items():
                _cancel(check_cancelled)
                name = filename[:-4]
                values = archive[name]
                if values.shape != shape or values.dtype != dtype or not np.isfinite(values).all():
                    raise VerificationArtifactError("NONFINITE_OR_INVALID_NUMERICAL_ARRAY")
                values.setflags(write=False)
                result[name] = values
        return result
    except (VerificationArtifactError, SimulationCancelled):
        raise
    except (
        ValueError,
        OSError,
        EOFError,
        RuntimeError,
        OverflowError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise VerificationArtifactError("INVALID_NUMERICAL_ARCHIVE") from error


def _json_artifact(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
        raise VerificationArtifactError("INVALID_NUMERICAL_JSON")

    def constant(value):
        raise VerificationArtifactError("NONFINITE_NUMERICAL_JSON")

    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise VerificationArtifactError("NONFINITE_NUMERICAL_JSON")
        return parsed

    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise VerificationArtifactError("DUPLICATE_NUMERICAL_JSON_KEY")
            result[name] = value
        return result

    try:
        decoded = json.loads(
            raw, parse_constant=constant, parse_float=floating, object_pairs_hook=pairs
        )
    except (ValueError, RecursionError, UnicodeError) as error:
        raise VerificationArtifactError("INVALID_NUMERICAL_JSON") from error
    if not isinstance(decoded, dict):
        raise VerificationArtifactError("INVALID_NUMERICAL_JSON")
    return decoded


def _check(name: str, passed: bool, *, reason: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "reason": reason, **fields}


def _numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise VerificationArtifactError("INVALID_SAVED_OBSERVABLE")
    return array.astype(np.float64, copy=False)


def _near(
    name: str, observed: Any, expected: Any, *, scale: float = 1.0, **details: Any
) -> dict[str, Any]:
    try:
        left, right = _numeric(observed), _numeric(expected)
        if left.shape != right.shape:
            return _check(name, False, reason="OBSERVABLE_SHAPE_MISMATCH")
        absolute = NUMERICAL_ATOL * abs(float(scale))
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(left - right)
            error = float(np.max(difference)) if difference.size else 0.0
            passed = bool(np.allclose(left, right, rtol=NUMERICAL_RTOL, atol=absolute))
        output = _check(
            name,
            passed,
            reason="INDEPENDENT_NUMERICAL_AGREEMENT" if passed else "NUMERICAL_DISAGREEMENT",
            details={
                "rtol": NUMERICAL_RTOL,
                "atol": absolute,
                "shape": list(right.shape),
                "max_absolute_error": error if math.isfinite(error) else None,
                **details,
            },
        )
        if right.size <= 2:
            output["expected"] = float(right) if right.ndim == 0 else right.tolist()
            output["observed"] = float(left) if left.ndim == 0 else left.tolist()
        return output
    except (ValueError, TypeError, OverflowError) as error:
        return _check(name, False, reason="INVALID_SAVED_OBSERVABLE")


def _steps(parameters: LangevinParameters | LennardJonesParameters) -> np.ndarray:
    # Derive the schedule independently of the production sampling helper.
    schedule = list(range(0, parameters.num_steps + 1, parameters.sample_stride))
    if schedule[-1] != parameters.num_steps:
        schedule.append(parameters.num_steps)
    return np.asarray(schedule, dtype=np.int64)


def _wrapped(values: np.ndarray, box_size) -> np.ndarray:
    box = np.asarray(box_size, dtype=np.float64)
    wrapped = np.remainder(values, box)
    return np.where((wrapped >= box) | (wrapped < 0), 0.0, wrapped)


def _magnitude(value: Any) -> float:
    magnitude = float(np.max(np.abs(value)))
    if not math.isfinite(magnitude):
        raise VerificationArtifactError("NONFINITE_INDEPENDENT_REFERENCE")
    return max(magnitude, np.finfo(float).tiny)


def _terminal_checks(parameters, result, state):
    velocities = state["velocities"]
    mass = 1.0 if isinstance(parameters, SimulationParameters) else parameters.mass
    kinetic = float(0.5 * mass * np.sum(np.square(velocities), dtype=np.float64))
    if isinstance(parameters, SimulationParameters):
        return [
            _near(
                "terminal_kinetic_energy",
                result.get("total_energy"),
                kinetic,
                scale=_magnitude(kinetic),
            ),
            _near(
                "terminal_temperature",
                result.get("final_temperature"),
                kinetic / parameters.num_particles,
                scale=_magnitude(kinetic / parameters.num_particles),
            ),
            _near(
                "declared_snapshot_standard_error",
                result.get("temperature_standard_error"),
                parameters.initial_temperature / math.sqrt(parameters.num_particles),
                scale=parameters.initial_temperature / math.sqrt(parameters.num_particles),
                scope="Formula for independent equilibrium velocities; no equilibrium goodness-of-fit test",
            ),
        ]
    mean = np.mean(velocities, axis=0)
    thermal = (
        mass
        * float(np.sum(np.square(velocities - mean), dtype=np.float64))
        / (2 * (parameters.num_particles - 1))
    )
    checks = [
        _near(
            "terminal_kinetic_energy",
            result.get("kinetic_energy"),
            kinetic,
            scale=_magnitude(kinetic),
        ),
        _near(
            "terminal_thermal_temperature",
            result.get("thermal_temperature"),
            thermal,
            scale=_magnitude(thermal),
        ),
    ]
    if isinstance(parameters, LangevinParameters):
        exponent = -2 * parameters.friction * parameters.num_steps * parameters.time_step
        temperature = math.fsum(
            (
                parameters.initial_temperature * math.exp(exponent),
                parameters.bath_temperature * -math.expm1(exponent),
            )
        )
        checks += [
            _near(
                "terminal_mean_velocity",
                result.get("mean_velocity"),
                mean,
                scale=math.sqrt(max(thermal, np.finfo(float).tiny) / parameters.mass),
            ),
            _near(
                "declared_transient_temperature_error",
                result.get("thermal_temperature_standard_error"),
                temperature / math.sqrt(parameters.num_particles - 1),
                scale=temperature / math.sqrt(parameters.num_particles - 1),
                scope="Independent Gaussian particles at one snapshot, not a confidence interval",
            ),
        ]
    return checks


def _diffusion_checks(parameters, positions, times, centered_msd, saved):
    if not isinstance(saved, dict):
        return [_check("diffusion_contract", False, reason="INVALID_SAVED_OBSERVABLE")]
    checks = []
    theoretical = parameters.bath_temperature / (parameters.mass * parameters.friction)
    checks.append(
        _near(
            "declared_diffusion_coefficient",
            saved.get("theoretical"),
            theoretical,
            scale=theoretical,
        )
    )
    minimum = max(
        5.0,
        math.log(
            (1 + abs(parameters.initial_temperature / parameters.bath_temperature - 1)) / 0.01
        ),
    )
    start = parameters.diffusion_fit_start_step
    steps = _steps(parameters)
    if start is None:
        admissible = np.flatnonzero(parameters.friction * times >= minimum)
        start = int(steps[admissible[0]]) if len(admissible) else None
    end = parameters.diffusion_fit_end_step
    end = parameters.num_steps if end is None else end
    indices = (
        np.flatnonzero((steps >= start) & (steps <= end))
        if start is not None
        else np.empty(0, dtype=np.int64)
    )
    reason = None
    if len(indices) < 3:
        reason = "insufficient_diffusive_samples"
    elif parameters.friction * times[indices[0]] < minimum:
        reason = "initial_transient_not_relaxed"
    if reason is not None:
        checks.append(
            _check(
                "diffusion_applicability",
                saved.get("status") == "unavailable"
                and saved.get("reason") == reason
                and saved.get("estimate") is None
                and saved.get("standard_error") is None,
                reason="FINITE_WINDOW_APPLICABILITY_CHECKED",
                expected={"status": "unavailable", "reason": reason},
                observed={"status": saved.get("status"), "reason": saved.get("reason")},
            )
        )
        return checks
    selected_times = times[indices]
    span = float(selected_times[-1] - selected_times[0])
    normalized = (selected_times - selected_times[0]) / span
    centered_time = normalized - np.mean(normalized)
    weights = centered_time / (span * float(np.dot(centered_time, centered_time)))
    response = -np.expm1(-parameters.friction * selected_times) / parameters.friction
    analytic_msd = (
        4 * theoretical * (selected_times - response)
        + 2
        * (parameters.initial_temperature - parameters.bath_temperature)
        / parameters.mass
        * response**2
    )
    delta = selected_times - selected_times[0]
    exponential = math.exp(-parameters.friction * float(selected_times[0]))
    contrast = (
        (parameters.initial_temperature - parameters.bath_temperature)
        / parameters.mass
        / parameters.friction**2
    )
    # Remove the large time-independent initial-temperature offset analytically,
    # rather than subtracting two MSD values that can both be around 1e15.
    correction = (4 * theoretical / parameters.friction - 4 * contrast) * exponential * np.expm1(
        -parameters.friction * delta
    ) + 2 * contrast * exponential**2 * np.expm1(-2 * parameters.friction * delta)
    expected_finite = theoretical + float(np.dot(weights, correction)) / 4
    # The recorded producer used full MSD subtraction. An error below this
    # isolated cancellation bound cannot support a precise numeric verdict.
    analytic_roundoff = (
        128
        * np.finfo(float).eps
        * float(np.sum(np.abs(weights)))
        * float(np.max(np.abs(analytic_msd)))
        / 4
    )
    bias = (expected_finite - theoretical) / theoretical
    if abs(bias) > 0.01:
        checks.append(
            _check(
                "diffusion_applicability",
                saved.get("status") == "unavailable"
                and saved.get("reason") == "finite_window_bias_exceeds_tolerance"
                and saved.get("estimate") is None
                and saved.get("standard_error") is None,
                reason="FINITE_WINDOW_APPLICABILITY_CHECKED",
            )
        )
        return checks
    checks.append(
        _check(
            "diffusion_applicability",
            saved.get("status") == "available" and saved.get("reason") is None,
            reason="FINITE_WINDOW_APPLICABILITY_CHECKED",
        )
    )
    selected_msd = np.asarray(centered_msd)[indices]
    estimate = float(np.dot(weights, selected_msd - selected_msd[0])) / 4
    intercept = float(np.mean(selected_msd)) - 4 * estimate * float(np.mean(selected_times))
    particles = parameters.num_particles
    deleted_estimates = np.zeros(particles, dtype=np.float64)
    baseline = None
    # Use deleted-sample sufficient statistics, rather than the production
    # contribution formula. Center first to avoid large-drift cancellation.
    for index, weight in zip(indices, weights, strict=True):
        displacement = positions[index] - positions[0]
        residual = displacement - np.mean(displacement, axis=0)
        norm = np.sum(residual**2, axis=1)
        residual_sum = np.sum(residual, axis=0)
        deleted = (
            float(np.sum(norm))
            - norm
            - np.sum((residual_sum - residual) ** 2, axis=1) / (particles - 1)
        ) / (particles - 2)
        if baseline is None:
            baseline = deleted.copy()
        deleted_estimates += float(weight) * (deleted - baseline) / 4
    standard_error = math.sqrt(
        (particles - 1)
        / particles
        * float(np.sum((deleted_estimates - np.mean(deleted_estimates)) ** 2))
    )
    checks += [
        _near(
            "diffusion_estimate",
            saved.get("estimate"),
            estimate,
            scale=_magnitude(estimate),
            scope="Centered-MSD finite-window OLS slope / 4; no confidence interval",
        ),
        _near(
            "diffusion_intercept",
            saved.get("intercept"),
            intercept,
            scale=max(float(np.max(np.abs(selected_msd))), np.finfo(float).tiny),
        ),
        _near(
            "diffusion_particle_jackknife_error",
            saved.get("standard_error"),
            standard_error,
            scale=_magnitude(standard_error),
            scope="Whole particles are deleted; time frames remain correlated; finite-window bias and model discrepancy are excluded",
        ),
        _check(
            "diffusion_fit_window",
            saved.get("fit_start_step") == int(steps[indices[0]])
            and saved.get("fit_end_step") == int(steps[indices[-1]])
            and saved.get("sample_count") == len(indices),
            reason="DECLARED_WINDOW_MATCHES_SAVED_SAMPLES",
        ),
    ]
    for name, recorded, expected, bound in (
        (
            "diffusion_analytic_window_bias",
            saved.get("relative_finite_window_bias"),
            bias,
            analytic_roundoff / theoretical,
        ),
        (
            "diffusion_analytic_window_estimate",
            saved.get("expected_finite_window_estimate"),
            expected_finite,
            analytic_roundoff,
        ),
    ):
        checked = _near(
            name, recorded, expected, scale=1.0 if name.endswith("bias") else _magnitude(expected)
        )
        if (
            checked["status"] == "failed"
            and isinstance(recorded, (int, float))
            and not isinstance(recorded, bool)
            and math.isfinite(recorded)
            and abs(recorded - expected) <= bound
        ):
            checked["status"] = "inconclusive"
            checked["reason"] = "ANALYTIC_WINDOW_CANCELLATION_LIMIT"
            checked["details"]["roundoff_bound"] = bound
            checked["details"]["scope"] = (
                "Isolated full-MSD subtraction conditioning; other observable tolerances are unchanged"
            )
        checks.append(checked)
    return checks


def _langevin_checks(parameters, result, trajectory, report, check_cancelled):
    positions, times = trajectory["positions"], trajectory["times"]
    means, raw_msd, centered, raw_error = [], [], [], []
    particles = parameters.num_particles
    for frame in positions:
        _cancel(check_cancelled)
        displacement = frame - positions[0]
        mean = np.mean(displacement, axis=0)
        squared = np.sum(displacement**2, axis=1)
        residual = displacement - mean
        means.append(mean.tolist())
        raw_msd.append(float(np.mean(squared)))
        centered.append(float(np.sum(residual**2)) / (particles - 1))
        raw_error.append(float(np.std(squared, ddof=1)) / math.sqrt(particles))
    scale = _magnitude(raw_msd)
    centered_scale = _magnitude(centered)
    fluctuation_scale = math.sqrt(centered_scale)
    checks = [
        _check(
            "initial_coordinates",
            np.array_equal(
                positions[0],
                np.broadcast_to(np.asarray(parameters.initial_position), positions[0].shape),
            ),
            reason="DECLARED_INITIAL_POSITION_CHECKED",
        ),
        _near(
            "mean_displacement_series",
            report.get("mean_displacement"),
            means,
            scale=fluctuation_scale,
        ),
        _near("raw_msd_series", report.get("msd"), raw_msd, scale=scale),
        _near("centered_msd_series", report.get("centered_msd"), centered, scale=centered_scale),
        _near(
            "raw_msd_snapshot_error",
            report.get("msd_standard_error"),
            raw_error,
            scale=_magnitude(raw_error),
            scope="Independent particles at each saved snapshot, not independent times",
        ),
        _near(
            "terminal_mean_displacement",
            result.get("mean_displacement"),
            means[-1],
            scale=fluctuation_scale,
        ),
        _near("terminal_raw_msd", result.get("final_msd"), raw_msd[-1], scale=scale),
        _near(
            "terminal_centered_msd",
            result.get("final_centered_msd"),
            centered[-1],
            scale=_magnitude(centered[-1]),
        ),
        _check(
            "diffusion_result_binding",
            report.get("diffusion") == result.get("diffusion"),
            reason="RESULT_AND_ANALYSIS_AGREE",
        ),
    ]
    checks.extend(
        _diffusion_checks(parameters, positions, times, centered, report.get("diffusion"))
    )
    return checks


def _lj_pair_reference(parameters, positions, check_cancelled):
    particles = parameters.num_particles
    pair_count = particles * (particles - 1) // 2
    if pair_count > MAX_PAIR_CHECKS:
        return None
    box = np.asarray(parameters.box_size)
    cutoff = parameters.cutoff_radius
    sigma = parameters.sigma
    epsilon = parameters.epsilon
    ratio = sigma / cutoff
    cutoff_potential = 4 * epsilon * (ratio**12 - ratio**6)
    cutoff_force = 24 * epsilon / cutoff * (2 * ratio**12 - ratio**6)
    edges = np.linspace(0, float(min(parameters.box_size)) / 2, parameters.rdf_bins + 1)
    counts = np.zeros(parameters.rdf_bins, dtype=np.int64)
    energies, virials = [], []
    for first in range(particles - 1):
        _cancel(check_cancelled)
        separation = positions[first + 1 :] - positions[first]
        separation -= box * np.rint(separation / box)
        radius = np.hypot(separation[:, 0], separation[:, 1])
        if (radius < 0.5 * sigma).any():
            raise VerificationArtifactError("TERMINAL_PAIR_SEPARATION_INVALID")
        active = radius[radius < cutoff]
        sixth = (sigma / active) ** 6
        energies.append(
            float(
                np.sum(
                    4 * epsilon * (sixth**2 - sixth)
                    - cutoff_potential
                    + (active - cutoff) * cutoff_force
                )
            )
        )
        virials.append(float(np.sum(24 * epsilon * (2 * sixth**2 - sixth) - active * cutoff_force)))
        included = radius[radius <= edges[-1]]
        bins = np.minimum(
            np.searchsorted(edges, included, side="right") - 1, parameters.rdf_bins - 1
        )
        counts += np.bincount(bins, minlength=parameters.rdf_bins)
    expected_counts = pair_count * math.pi * np.diff(edges**2) / math.prod(parameters.box_size)
    return math.fsum(energies), math.fsum(virials), edges, counts, expected_counts


def _lj_checks(parameters, result, state, trajectory, report, check_cancelled):
    frames, particles = len(trajectory["sample_steps"]), parameters.num_particles
    raw_series = report.get("series")
    if not isinstance(raw_series, dict) or set(raw_series) != set(_SERIES):
        raise VerificationArtifactError("INVALID_SAMPLED_OBSERVABLES")
    series = {name: _numeric(raw_series[name]) for name in _SERIES}
    if any(values.shape != (frames,) for values in series.values()):
        raise VerificationArtifactError("INVALID_SAMPLED_OBSERVABLES")
    energy_scale = parameters.epsilon * particles
    area = math.prod(parameters.box_size)
    columns = math.ceil(math.sqrt(particles * parameters.box_size[0] / parameters.box_size[1]))
    rows = math.ceil(particles / columns)
    indices = np.arange(particles)
    lattice = np.column_stack(
        (
            (indices % columns + 0.5) * parameters.box_size[0] / columns,
            (indices // columns + 0.5) * parameters.box_size[1] / rows,
        )
    )
    checks = [
        _near(
            "initial_lattice", trajectory["positions"][0], lattice, scale=max(parameters.box_size)
        ),
        _near(
            "initial_thermal_temperature",
            series["thermal_temperature"][0],
            parameters.initial_temperature,
            scale=parameters.initial_temperature,
        ),
        _near(
            "initial_kinetic_energy",
            series["kinetic_energy"][0],
            (particles - 1) * parameters.initial_temperature,
            scale=(particles - 1) * parameters.initial_temperature,
        ),
        _check(
            "nonnegative_kinetic_observables",
            bool(
                (series["thermal_temperature"] >= 0).all() and (series["kinetic_energy"] >= 0).all()
            ),
            reason="KINETIC_NONNEGATIVITY_CHECKED",
        ),
        _near(
            "sampled_energy_closure",
            series["total_energy"],
            series["kinetic_energy"] + series["potential_energy"],
            scale=energy_scale,
            scope="Algebraic identities on saved samples; intermediate energies are not independently recomputed",
        ),
        _near(
            "sampled_pressure_closure",
            series["pressure"],
            (2 * series["kinetic_energy"] + series["virial"]) / (2 * area),
            scale=energy_scale / area,
        ),
    ]
    for name in _SERIES:
        if name != "virial":
            checks.append(
                _near(
                    "terminal_series_" + name,
                    series[name][-1],
                    result.get(name),
                    scale=_magnitude(result.get(name)),
                )
            )
    production = report.get("production")
    if not isinstance(production, dict) or not isinstance(production.get("statistics"), dict):
        raise VerificationArtifactError("INVALID_PRODUCTION_SUMMARY")
    chosen = trajectory["sample_steps"] >= parameters.equilibration_steps
    checks.append(
        _check(
            "production_sample_window",
            production.get("sample_count") == int(np.count_nonzero(chosen))
            and production.get("first_sample_step") == int(trajectory["sample_steps"][chosen][0])
            and production.get("last_sample_step") == parameters.num_steps,
            reason="USER_SELECTED_SAVED_SAMPLE_WINDOW_CHECKED",
        )
    )
    for name, values in series.items():
        observed = production["statistics"].get(name)
        if not isinstance(observed, dict):
            raise VerificationArtifactError("INVALID_PRODUCTION_SUMMARY")
        selected = values[chosen]
        mean = math.fsum(float(value) for value in selected) / len(selected)
        checks.append(
            _near(
                "production_mean_" + name,
                observed.get("mean"),
                mean,
                scale=_magnitude(mean),
            )
        )
        if len(selected) > 1:
            deviation = math.sqrt(
                math.fsum((float(value) - mean) ** 2 for value in selected) / (len(selected) - 1)
            )
            checks.append(
                _near(
                    "production_sample_sd_" + name,
                    observed.get("sample_standard_deviation"),
                    deviation,
                    scale=_magnitude(deviation),
                    scope="Descriptive correlated-frame sample SD, not an independent-sample standard error",
                )
            )
        else:
            checks.append(
                _check(
                    "production_sample_sd_" + name,
                    observed.get("sample_standard_deviation") is None,
                    reason="SINGLE_SAMPLE_SD_IS_UNAVAILABLE",
                )
            )
    drift = report.get("energy_drift")
    if not isinstance(drift, dict):
        raise VerificationArtifactError("INVALID_ENERGY_DIAGNOSTIC")
    nve = parameters.ensemble == "nve"
    changes = series["total_energy"] - series["total_energy"][0]
    reference = max(abs(float(series["total_energy"][0])), energy_scale)
    checks.append(
        _check(
            "energy_change_applicability",
            drift.get("applicable") is nve,
            reason="THERMOSTAT_SPECIFIC_DIAGNOSTIC_CHECKED",
        )
    )
    expected_drift = {
        "initial_total_energy": float(series["total_energy"][0]),
        "reference_scale": reference,
        "maximum_absolute_change": float(np.max(np.abs(changes))) if nve else None,
        "maximum_normalized_absolute_change": float(np.max(np.abs(changes))) / reference
        if nve
        else None,
        "terminal_change": float(changes[-1]) if nve else None,
        "terminal_normalized_change": float(changes[-1]) / reference if nve else None,
    }
    for name, expected in expected_drift.items():
        checks.append(
            _near(
                "energy_change_" + name,
                drift.get(name),
                expected,
                scale=1.0 if "normalized" in name else energy_scale,
                scope="Saved-sample diagnostic, not a convergence or integration-accuracy verdict",
            )
            if expected is not None
            else _check(
                "energy_change_" + name,
                drift.get(name) is None,
                reason="NVT_ENERGY_CONSERVATION_IS_NOT_REQUIRED",
            )
        )
    reference_pairs = _lj_pair_reference(parameters, state["unwrapped_positions"], check_cancelled)
    if reference_pairs is None:
        checks.append(
            {
                "name": "terminal_pair_reference",
                "status": "unsupported",
                "reason": "INDEPENDENT_PAIR_BUDGET_EXCEEDED",
                "details": {
                    "required_pair_checks": particles * (particles - 1) // 2,
                    "maximum_pair_checks": MAX_PAIR_CHECKS,
                },
            }
        )
        return checks
    potential, virial, edges, counts, uniform = reference_pairs
    kinetic = float(parameters.mass / 2 * np.sum(state["velocities"] ** 2))
    checks += [
        _near(
            "terminal_potential_reference",
            result.get("potential_energy"),
            potential,
            scale=energy_scale,
            reference="Independent direct shifted-force LJ sum on terminal coordinates",
        ),
        _near("terminal_virial_reference", series["virial"][-1], virial, scale=energy_scale),
        _near(
            "terminal_total_energy_reference",
            result.get("total_energy"),
            kinetic + potential,
            scale=energy_scale,
        ),
        _near(
            "terminal_pressure_reference",
            result.get("pressure"),
            (2 * kinetic + virial) / (2 * area),
            scale=energy_scale / area,
        ),
    ]
    rdf = report.get("rdf")
    if not isinstance(rdf, dict):
        raise VerificationArtifactError("INVALID_RADIAL_DISTRIBUTION")
    checks += [
        _near("rdf_bin_edges", rdf.get("bin_edges"), edges, scale=float(edges[-1])),
        _check(
            "rdf_pair_counts",
            np.array_equal(_numeric(rdf.get("counts")), counts),
            reason="INDEPENDENT_TERMINAL_PAIR_HISTOGRAM_CHECKED",
        ),
        _near(
            "rdf_uniform_normalization",
            rdf.get("expected_uniform_counts"),
            uniform,
            scale=float(max(np.max(uniform), np.finfo(float).tiny)),
        ),
        _near(
            "rdf_distribution",
            rdf.get("g_r"),
            counts / uniform,
            scope="One terminal 2D snapshot, not equilibrium or uncertainty validation",
        ),
        _check(
            "rdf_finite_particle_contract",
            rdf.get("dimensions") == 2
            and rdf.get("sample_size") == particles
            and rdf.get("pair_checks") == particles * (particles - 1) // 2,
            reason="TWO_DIMENSIONAL_FINITE_N_NORMALIZATION_CHECKED",
        ),
    ]
    return checks


def verify_scientific_result(
    parameters: ExperimentParameters,
    result: dict[str, Any],
    artifacts: dict[str, bytes],
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Check declared deterministic observables; the caller verifies digests."""
    checks: list[dict[str, Any]] = []
    if not isinstance(
        parameters, (SimulationParameters, LangevinParameters, LennardJonesParameters)
    ):
        return [{"name": "model_support", "status": "unsupported", "reason": "MODEL_NOT_SUPPORTED"}]
    model = (
        LJ_MODEL_VERSION
        if isinstance(parameters, LennardJonesParameters)
        else LANGEVIN_MODEL_VERSION
        if isinstance(parameters, LangevinParameters)
        else MODEL_VERSION
    )
    try:
        _cancel(check_cancelled)
        manifest = _json_artifact(artifacts["manifest"])
        canonical_parameters = json.loads(json.dumps(asdict(parameters), allow_nan=False))
        checks.append(
            _check(
                "source_scientific_contract",
                result.get("status") == "completed"
                and result.get("model_version") == model
                and result.get("seed") == parameters.seed
                and result.get("num_steps") == parameters.num_steps
                and manifest.get("parameters") == canonical_parameters
                and manifest.get("outputs") == result
                and isinstance(manifest.get("model"), dict)
                and manifest["model"].get("version") == model,
                reason="DECLARED_MODEL_INPUTS_AND_RESULT_CHECKED",
            )
        )
        shape = (parameters.num_particles, 2)
        state_contract = {"velocities": (shape, np.dtype(np.float64))}
        if not isinstance(parameters, SimulationParameters):
            state_contract.update(
                {
                    name: (shape, np.dtype(np.float64))
                    for name in ("positions", "unwrapped_positions")
                }
            )
        state = decode_verification_npz(
            artifacts["final_state"], state_contract, check_cancelled=check_cancelled
        )
        checks.append(
            _check(
                "terminal_array_contract", True, reason="BOUNDED_FINITE_NUMERICAL_ARRAYS_CHECKED"
            )
        )
        checks.extend(_terminal_checks(parameters, result, state))
        if isinstance(parameters, SimulationParameters):
            return checks
        steps = _steps(parameters)
        trajectory = decode_verification_npz(
            artifacts["trajectory"],
            {
                "positions": ((len(steps), parameters.num_particles, 2), np.dtype(np.float64)),
                "sample_steps": ((len(steps),), np.dtype(np.int64)),
                "times": ((len(steps),), np.dtype(np.float64)),
            },
            check_cancelled=check_cancelled,
        )
        checks += [
            _check(
                "trajectory_sampling_contract",
                np.array_equal(trajectory["sample_steps"], steps)
                and np.array_equal(
                    trajectory["times"], steps.astype(np.float64) * parameters.time_step
                ),
                reason="DECLARED_STEP_AND_TIME_SCHEDULE_CHECKED",
            ),
            _check(
                "terminal_trajectory_binding",
                np.array_equal(trajectory["positions"][-1], state["unwrapped_positions"]),
                reason="TERMINAL_COORDINATES_MATCH_SAVED_TRAJECTORY",
            ),
        ]
        periodic = (
            isinstance(parameters, LennardJonesParameters) or parameters.boundary == "periodic"
        )
        expected_position = (
            _wrapped(state["unwrapped_positions"], parameters.box_size)
            if periodic
            else state["unwrapped_positions"]
        )
        checks.append(
            _check(
                "presentation_coordinates",
                np.array_equal(expected_position, state["positions"]),
                reason="DECLARED_COORDINATE_CONVENTION_CHECKED",
            )
        )
        report = _json_artifact(
            artifacts[
                "observables" if isinstance(parameters, LennardJonesParameters) else "transport"
            ]
        )
        checks += [
            _check(
                "analysis_model_contract",
                report.get("model_version") == model
                and report.get("dimensions") == 2
                and report.get("sample_size") == parameters.num_particles,
                reason="MODEL_SPECIFIC_ANALYSIS_CONTRACT_CHECKED",
            ),
            _near("analysis_sample_steps", report.get("sample_steps"), steps),
            _near(
                "analysis_sample_times",
                report.get("times"),
                trajectory["times"],
                scale=float(trajectory["times"][-1]),
            ),
        ]
        _cancel(check_cancelled)
        checks.extend(
            _lj_checks(parameters, result, state, trajectory, report, check_cancelled)
            if isinstance(parameters, LennardJonesParameters)
            else _langevin_checks(parameters, result, trajectory, report, check_cancelled)
        )
        return checks
    except SimulationCancelled:
        raise
    except (
        VerificationArtifactError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ) as error:
        reason = (
            str(error)
            if isinstance(error, VerificationArtifactError)
            else "INVALID_NUMERICAL_ARTIFACT"
        )
        checks.append(_check("numerical_artifact_validation", False, reason=reason))
        return checks
