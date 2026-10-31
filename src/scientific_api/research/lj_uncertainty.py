"""Conditional uncertainty for one saved, temporally correlated LJ trajectory.

The resampling unit is a block of saved time frames, never a particle.  Passing
the diagnostics is not an equilibration test.  In particular, positive-sequence
IAT truncation is a diagnostic heuristic for molecular dynamics; the ordinary
reversible-Markov-chain positivity theorem is not assumed for this integrator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..domain.lennard_jones import (
    LJ_MODEL_VERSION,
    MAX_LJ_TRAJECTORY_FRAMES,
    LennardJonesParameters,
)
from .lj_analysis import SERIES_NAMES
from .student_t import student_t_critical

UNCERTAINTY_VERSION = "lj-correlated-uncertainty-v1"
MIN_SAMPLES = 128
MIN_BLOCKS = 8
MAX_BLOCKS = 50
MIN_EFFECTIVE_SAMPLES = 20.0
MIN_BOOTSTRAP_SAMPLES = 200
MAX_BOOTSTRAP_SAMPLES = 5000
CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)
_ASSUMPTIONS = [
    "The user assumes the retained series is stationary and representative of the intended state",
    "Dependence decays within the observed window and the chosen time-block length",
    "A finite second moment and an approximately Gaussian, weakly dependent batch-mean approximation are appropriate",
]
_LIMITATIONS = [
    "Diagnostics can detect some drift or unresolved dependence; passing them does not prove equilibrium or stationarity",
    "Positive-sequence IAT truncation is heuristic for these molecular dynamics integrators, not a guaranteed conservative bound",
    "A finite saved window can miss slow modes or later positive autocorrelation lobes",
    "Primary intervals use an approximate batch-means Student-t construction; residual batch dependence and finite-block variance bias can reduce coverage",
    "Intervals are pointwise and conditional for a saved-frame mean; not simultaneous, predictive, or physical-model error bounds",
    "Secondary moving-block percentile intervals are exploratory and can under-cover even when their Monte Carlo quantiles are adequately resolved",
    "No particle-independence assumption or square-root-of-particle-count standard error is used",
    "No uncertainty is inferred for a terminal RDF or unsaved trajectory frames",
    "Ordinary overlapping moving blocks give finite-window endpoint weighting; bootstrap center and bias are reported",
]


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def select_lj_production(
    observables: Mapping[str, Any],
    parameters: LennardJonesParameters,
    *,
    burn_in_step: int | None = None,
) -> dict[str, Any]:
    """Copy only actual saved regular-grid production frames, without dynamics.

    An extra final sample whose gap is shorter than ``sample_stride`` is always
    excluded, including when it is the only frame after the burn-in threshold.
    ``sampling`` is JSON safe; returned array copies are read-only.
    """
    if not isinstance(parameters, LennardJonesParameters):
        raise TypeError("parameters must be LennardJonesParameters")
    if not isinstance(observables, Mapping):
        raise ValueError("observables must be a mapping")
    if observables.get("model_version") != LJ_MODEL_VERSION:
        raise ValueError("observables model_version does not match the LJ model")
    threshold = parameters.equilibration_steps
    if burn_in_step is not None:
        threshold = _integer(burn_in_step, "burn_in_step", threshold, parameters.num_steps)
    expected = parameters.sampling_steps()
    source_steps = observables.get("sample_steps")
    if (
        not isinstance(source_steps, (list, tuple, np.ndarray))
        or (isinstance(source_steps, np.ndarray) and source_steps.ndim != 1)
        or len(source_steps) != len(expected)
    ):
        raise ValueError("sample_steps must match the saved sampling schedule")
    if isinstance(source_steps, np.ndarray):
        if source_steps.dtype != np.dtype(np.int64) or source_steps.ndim != 1:
            raise ValueError("sample_steps must be native int64 with one dimension")
    elif any(type(step) is not int for step in source_steps):
        raise ValueError("sample_steps must contain integers")
    steps = np.asarray(source_steps, dtype=np.int64)
    if len(steps) > MAX_LJ_TRAJECTORY_FRAMES or not np.array_equal(steps, expected):
        raise ValueError("sample_steps must match the bounded saved sampling schedule")
    source_series = observables.get("series")
    if not isinstance(source_series, Mapping) or set(source_series) != set(SERIES_NAMES):
        raise ValueError("series must contain exactly the six LJ observable names")
    regular = steps % parameters.sample_stride == 0
    selected = regular & (steps >= threshold)
    arrays: dict[str, NDArray[np.float64]] = {}
    for name in SERIES_NAMES:
        values = source_series[name]
        if (
            not isinstance(values, (list, tuple, np.ndarray))
            or (isinstance(values, np.ndarray) and values.ndim != 1)
            or len(values) != len(steps)
        ):
            raise ValueError(f"{name} must match the saved frame count")
        if isinstance(values, np.ndarray):
            if values.dtype != np.dtype(np.float64) or values.ndim != 1:
                raise ValueError(f"{name} must be native one-dimensional float64")
        elif any(
            isinstance(value, bool) or not isinstance(value, (float, int)) for value in values
        ):
            raise ValueError(f"{name} must contain finite numbers")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{name} must contain finite float64 numbers") from error
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite float64 numbers")
        if name in {"thermal_temperature", "kinetic_energy"} and (array < 0).any():
            raise ValueError(f"{name} must not be negative")
        arrays[name] = array[selected].copy()
        arrays[name].flags.writeable = False
    retained_steps = steps[selected].copy()
    retained_steps.flags.writeable = False
    return {
        "sample_steps": retained_steps,
        "series": arrays,
        "sampling": {
            "source_sample_count": len(steps),
            "retained_sample_count": len(retained_steps),
            "requested_burn_in_step": burn_in_step,
            "physics_equilibration_steps": parameters.equilibration_steps,
            "effective_burn_in_step": threshold,
            "first_retained_step": int(retained_steps[0]) if len(retained_steps) else None,
            "last_retained_step": int(retained_steps[-1]) if len(retained_steps) else None,
            "sample_stride": parameters.sample_stride,
            "sample_interval": parameters.sample_stride * parameters.time_step,
            "excluded_irregular_terminal_steps": steps[~regular].tolist(),
            "excluded_burn_in_sample_count": int(np.count_nonzero(regular & (steps < threshold))),
            "selection": "Saved regular-grid frames at or after the burn-in threshold; no interpolation or duration weighting",
        },
    }


def _autocorrelation(
    standardized: NDArray[np.float64], check_cancelled: Callable[[], None] | None
) -> tuple[float, dict[str, Any]]:
    """Biased autocovariance, exact Geyer paired initial-positive/monotone sum."""
    count = len(standardized)
    denominator = float(np.dot(standardized, standardized))
    maximum_lag = count // 2
    rho = np.empty(maximum_lag + 1, dtype=np.float64)
    rho[0] = 1.0
    for lag in range(1, maximum_lag + 1):
        if check_cancelled is not None and lag % 32 == 0:
            check_cancelled()
        rho[lag] = float(np.dot(standardized[:-lag], standardized[lag:])) / denominator
    pairs = rho[: 2 * ((maximum_lag + 1) // 2)].reshape(-1, 2).sum(axis=1)
    retained: list[float] = []
    stopped_at: int | None = None
    previous = math.inf
    for index, value in enumerate(pairs):
        if value <= 0:
            stopped_at = index
            break
        previous = min(previous, float(value))
        retained.append(previous)
    raw_iat = -1.0 + 2.0 * math.fsum(retained)
    iat = max(1.0, raw_iat)
    tail = pairs[stopped_at + 1 :] if stopped_at is not None else np.empty(0)
    # A deliberately stringent heuristic; no finite-window tail check can
    # exclude unobserved long-time correlations. Report its threshold openly.
    bartlett_inflation = math.sqrt(
        1.0 + 2.0 * math.fsum(float(value) ** 2 for value in rho[1 : 2 * len(retained)])
    )
    tail_threshold = 4.0 * math.sqrt(2.0 / count) * bartlett_inflation
    tail_maximum = float(np.max(tail)) if len(tail) else None
    tail_reemergence = tail_maximum is not None and tail_maximum > tail_threshold
    late_maximum = float(np.max(np.abs(rho[count // 4 :])))
    return iat, {
        "method": "Biased ACF; Gamma_k=rho(2k)+rho(2k+1); initial positive and monotone paired sequence; g=max(1,-1+2*sum(Gamma_k))",
        "maximum_examined_lag": maximum_lag,
        "retained_pair_count": len(retained),
        "first_nonpositive_pair": stopped_at,
        "unclamped_integrated_autocorrelation_time": raw_iat,
        "negative_correlation_benefit_used": False,
        "positive_sequence_truncated": stopped_at is not None,
        "positive_tail_reemergence": tail_reemergence,
        "maximum_later_pair_sum": tail_maximum,
        "positive_tail_threshold": tail_threshold,
        "tail_threshold_bartlett_inflation": bartlett_inflation,
        "maximum_late_absolute_correlation": late_maximum,
        "persistent_late_correlation": late_maximum > tail_threshold,
    }


def _drift_diagnostics(standardized: NDArray[np.float64], iat: float) -> dict[str, Any]:
    count = len(standardized)
    chunks = np.array_split(standardized, 4)
    means = [float(np.mean(chunk)) for chunk in chunks]
    variances = [float(np.var(chunk, ddof=1)) for chunk in chunks]
    minimum_variance = min(variances)
    maximum_variance = max(variances)
    variance_ratio = maximum_variance / minimum_variance if minimum_variance > 0 else None
    time = np.linspace(-0.5, 0.5, count, dtype=np.float64)
    norm = float(np.dot(time, time))
    slope = float(np.dot(time, standardized)) / norm
    trend_z = abs(slope) * math.sqrt(norm / iat)
    trend_explained_variance = slope * slope * norm / (count - 1)
    end_difference = abs(means[-1] - means[0])
    end_z = end_difference / math.sqrt(iat * (1 / len(chunks[0]) + 1 / len(chunks[-1])))
    flags: list[str] = []
    if trend_z > 4.5 or trend_explained_variance > 0.5:
        flags.append("linear_drift_detected")
    if end_z > 4.5:
        flags.append("end_window_mean_shift_detected")
    if minimum_variance == 0 or (variance_ratio is not None and variance_ratio > 4.0):
        flags.append("quarter_variance_shift_detected")
    return {
        "status": "flags_detected" if flags else "no_flags_detected",
        "flags": flags,
        "linear_trend_z_heuristic": trend_z,
        "linear_trend_explained_variance": trend_explained_variance,
        "end_window_mean_shift_z_heuristic": end_z,
        "quarter_variance_ratio": variance_ratio,
        "mean_drift_z_threshold": 4.5,
        "linear_trend_explained_variance_threshold": 0.5,
        "quarter_variance_ratio_threshold": 4.0,
        "interpretation": "Heuristic drift screening with estimated temporal dependence; no stationarity or equilibrium proof",
    }


def _block_means(
    centered: NDArray[np.float64],
    block_length: int,
    samples: int,
    seed: int,
    check_cancelled: Callable[[], None] | None,
) -> NDArray[np.float64]:
    """Ordinary moving-block mean bootstrap, O(B*n/b+n) storage/work.

    A fresh private PCG64 with the same seed for each metric gives the same
    block starts when block lengths match.  The final block is truncated to
    exactly n frames; valid starts remain 0..n-b, including for that block.
    """
    count = len(centered)
    full_blocks, remainder = divmod(count, block_length)
    prefix = np.concatenate(([0.0], np.cumsum(centered, dtype=np.float64)))
    generator = np.random.Generator(np.random.PCG64(seed))
    output = np.empty(samples, dtype=np.float64)
    # Batching bounds temporary index/sum matrices even at the largest request.
    for begin in range(0, samples, 128):
        if check_cancelled is not None:
            check_cancelled()
        batch = min(128, samples - begin)
        starts = generator.integers(
            0, count - block_length + 1, size=(batch, full_blocks + bool(remainder))
        )
        totals = np.sum(
            prefix[starts[:, :full_blocks] + block_length] - prefix[starts[:, :full_blocks]],
            axis=1,
        )
        if remainder:
            last = starts[:, -1]
            totals += prefix[last + remainder] - prefix[last]
        output[begin : begin + batch] = totals / count
    return output


def _metric(
    values: NDArray[np.float64],
    *,
    assume_stationary: bool,
    block_length: int | None,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence_level: float,
    check_cancelled: Callable[[], None] | None,
) -> dict[str, Any]:
    count = len(values)
    report: dict[str, Any] = {
        "status": "unavailable",
        "reason": "insufficient_samples",
        "mean": None,
        "sample_standard_deviation": None,
        "integrated_autocorrelation_time_samples": None,
        "effective_sample_size": None,
        "correlated_mean_standard_error": None,
        "iat_mean_standard_error": None,
        "confidence_interval": None,
        "exploratory_bootstrap_percentile": None,
        "exploratory_bootstrap_reason": "primary_interval_unavailable",
        "bootstrap_expected_tail_samples": bootstrap_samples * (1 - confidence_level) / 2,
        "block_length": None,
        "diagnostics": {},
    }
    if not count:
        return report
    # Scale BEFORE subtraction/squaring.  This preserves tiny physical units,
    # avoids overflow for signed extreme values and never uses epsilon*N or an
    # unrelated observable as a numerical floor.
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        report.update(mean=0.0, sample_standard_deviation=0.0, reason="flat_series")
        return report
    normalized = values / scale
    offset = float(normalized[0])
    differences = normalized - offset
    mean_difference = math.fsum(float(value) for value in differences) / count
    mean_normalized = offset + mean_difference
    centered = differences - mean_difference
    squared = math.fsum(float(value) ** 2 for value in centered)
    sd_normalized = math.sqrt(squared / (count - 1)) if count > 1 else 0.0
    mean = mean_normalized * scale
    sd = sd_normalized * scale
    if not math.isfinite(mean) or not math.isfinite(sd):
        report["reason"] = "unrepresentable_statistics"
        return report
    report.update(mean=mean, sample_standard_deviation=sd if count > 1 else None)
    if sd_normalized <= 32 * np.finfo(np.float64).eps:
        report["reason"] = "flat_series"
        return report
    if count < MIN_SAMPLES:
        return report
    standardized = centered / sd_normalized
    iat, correlation = _autocorrelation(standardized, check_cancelled)
    effective = count / iat
    drift = _drift_diagnostics(standardized, iat)
    recommended = max(
        math.ceil(8 * iat), math.ceil(count / MAX_BLOCKS), math.ceil(count ** (1 / 3))
    )
    chosen = recommended if block_length is None else block_length
    report.update(
        integrated_autocorrelation_time_samples=iat,
        effective_sample_size=effective,
        block_length=chosen,
        diagnostics={
            "autocorrelation": correlation,
            "stationarity_screen": drift,
            "recommended_block_length": recommended,
            "complete_block_count": count // chosen,
            "user_assumes_stationary": assume_stationary,
            "expected_bootstrap_tail_samples": bootstrap_samples * (1 - confidence_level) / 2,
        },
    )
    if drift["flags"]:
        report["reason"] = "nonstationarity_detected"
    elif correlation["positive_tail_reemergence"]:
        report["reason"] = "unresolved_positive_correlation_tail"
    elif (
        not correlation["positive_sequence_truncated"] or correlation["persistent_late_correlation"]
    ):
        report["reason"] = "unresolved_correlation_window"
    elif effective < MIN_EFFECTIVE_SAMPLES:
        report["reason"] = "insufficient_effective_samples"
    elif chosen < math.ceil(8 * iat):
        report["reason"] = "block_length_too_short_for_correlation"
    elif count // chosen < MIN_BLOCKS:
        report["reason"] = "insufficient_time_blocks"
    elif count // chosen > MAX_BLOCKS:
        report["reason"] = "too_many_time_blocks"
    elif not assume_stationary:
        report["reason"] = "stationary_assumption_required"
    else:
        full_blocks, remainder = divmod(count, chosen)
        batch_values = np.mean(
            centered[: full_blocks * chosen].reshape(full_blocks, chosen), axis=1
        )
        batch_center = math.fsum(float(value) for value in batch_values) / full_blocks
        batch_sd_normalized = math.sqrt(
            math.fsum((float(value) - batch_center) ** 2 for value in batch_values)
            / (full_blocks - 1)
        )
        if batch_sd_normalized <= 32 * np.finfo(np.float64).eps * sd_normalized:
            report["reason"] = "flat_batch_means"
            return report
        standard_error_normalized = batch_sd_normalized * math.sqrt(chosen / count)
        critical = student_t_critical(confidence_level, full_blocks - 1)
        half_width_normalized = critical * standard_error_normalized
        lower = (mean_normalized - half_width_normalized) * scale
        upper = (mean_normalized + half_width_normalized) * scale
        standard_error = standard_error_normalized * scale
        iat_standard_error = sd * math.sqrt(iat / count)
        prefix_mean = (mean_normalized + batch_center) * scale
        batch_sd = batch_sd_normalized * scale
        if not all(
            math.isfinite(value)
            for value in (lower, upper, standard_error, iat_standard_error, prefix_mean, batch_sd)
        ):
            report["reason"] = "unrepresentable_statistics"
            return report
        report["diagnostics"]["batch_means"] = {
            "method": "Non-overlapping complete prefix batches; variance of batch means scaled by block_length/n",
            "complete_batch_count": full_blocks,
            "degrees_of_freedom": full_blocks - 1,
            "critical_value": critical,
            "variance_sample_count": full_blocks * chosen,
            "variance_excluded_remainder_count": remainder,
            "mean_sample_count": count,
            "mean_uses_remainder": True,
            "complete_batches_mean": prefix_mean,
            "batch_means_sample_standard_deviation": batch_sd,
            "standard_error_formula": "sample_sd(complete_batch_means)*sqrt(block_length/n)",
            "interpretation": "Approximate Gaussian weakly dependent batches; residual correlation and finite-block bias are not guaranteed absent",
        }
        report.update(
            status="available",
            reason="conditional_on_stationarity_and_resolved_time_dependence",
            correlated_mean_standard_error=standard_error,
            iat_mean_standard_error=iat_standard_error,
            confidence_interval={
                "lower": lower,
                "upper": upper,
                "level": confidence_level,
                "nominal_level": confidence_level,
                "coverage_guarantee": False,
                "method": "Approximate non-overlapping batch-means Student-t interval centered on the full saved-frame arithmetic mean",
                "degrees_of_freedom": full_blocks - 1,
            },
        )
        if bootstrap_samples * (1 - confidence_level) / 2 < 10.0 - 1e-12:
            report["exploratory_bootstrap_reason"] = "bootstrap_quantile_resolution_insufficient"
            return report
        bootstrap = _block_means(
            centered, chosen, bootstrap_samples, bootstrap_seed, check_cancelled
        )
        alpha = (1 - confidence_level) / 2
        quantiles = np.quantile(bootstrap, (alpha, 1 - alpha), method="linear")
        bootstrap_lower, bootstrap_upper = (
            (float(value) + mean_normalized) * scale for value in quantiles
        )
        prefix = np.concatenate(([0.0], np.cumsum(centered, dtype=np.float64)))
        starts = np.arange(count - chosen + 1)
        expected_offset = full_blocks * float(np.mean(prefix[starts + chosen] - prefix[starts]))
        if remainder:
            expected_offset += float(np.mean(prefix[starts + remainder] - prefix[starts]))
        expected_offset /= count
        report["diagnostics"]["bootstrap_center"] = {
            "expected_mean": (mean_normalized + expected_offset) * scale,
            "endpoint_weighting_bias": expected_offset * scale,
            "empirical_mean": (mean_normalized + float(np.mean(bootstrap))) * scale,
            "interpretation": "Ordinary moving-block finite-window endpoint weighting; percentile endpoints are approximate",
        }
        if not all(
            math.isfinite(value)
            for value in (
                bootstrap_lower,
                bootstrap_upper,
                report["diagnostics"]["bootstrap_center"]["expected_mean"],
                report["diagnostics"]["bootstrap_center"]["endpoint_weighting_bias"],
                report["diagnostics"]["bootstrap_center"]["empirical_mean"],
            )
        ):
            report["diagnostics"].pop("bootstrap_center")
            report["exploratory_bootstrap_reason"] = "unrepresentable_statistics"
            return report
        report["exploratory_bootstrap_percentile"] = {
            "lower": bootstrap_lower,
            "upper": bootstrap_upper,
            "level": confidence_level,
            "nominal_level": confidence_level,
            "coverage_guarantee": False,
            "method": "Exploratory ordinary moving-block bootstrap percentile of the saved-frame arithmetic mean",
        }
        report["exploratory_bootstrap_reason"] = "exploratory_percentile_bootstrap"
    return report


def analyze_lj_uncertainty(
    observables: Mapping[str, Any],
    parameters: LennardJonesParameters,
    *,
    burn_in_step: int | None = None,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    block_length: int | None = None,
    assume_stationary: bool = False,
    observable_names: tuple[str, ...] = SERIES_NAMES,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Analyze a bounded immutable saved artifact; never mutate physics/RNG state."""
    _integer(bootstrap_seed, "bootstrap_seed", 0, 2**64 - 1)
    _integer(bootstrap_samples, "bootstrap_samples", MIN_BOOTSTRAP_SAMPLES, MAX_BOOTSTRAP_SAMPLES)
    if isinstance(confidence_level, bool) or confidence_level not in CONFIDENCE_LEVELS:
        raise ValueError("confidence_level must be 0.9, 0.95, or 0.99")
    if type(assume_stationary) is not bool:
        raise ValueError("assume_stationary must be a boolean")
    if block_length is not None:
        _integer(block_length, "block_length", 1, MAX_LJ_TRAJECTORY_FRAMES)
    if (
        not isinstance(observable_names, tuple)
        or not 1 <= len(observable_names) <= len(SERIES_NAMES)
        or any(not isinstance(name, str) for name in observable_names)
        or len(set(observable_names)) != len(observable_names)
        or any(name not in SERIES_NAMES for name in observable_names)
    ):
        raise ValueError("observable_names must be a nonempty tuple of unique supported names")
    if check_cancelled is not None:
        check_cancelled()
    selected = select_lj_production(observables, parameters, burn_in_step=burn_in_step)
    metrics = {}
    for name in observable_names:
        if check_cancelled is not None:
            check_cancelled()
        metrics[name] = _metric(
            selected["series"][name],
            assume_stationary=assume_stationary,
            block_length=block_length,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            check_cancelled=check_cancelled,
        )
    return {
        "analysis_version": UNCERTAINTY_VERSION,
        "model_version": LJ_MODEL_VERSION,
        "sampling": selected["sampling"],
        "metrics": metrics,
        "bootstrap": {
            "generator": "PCG64",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "requested_block_length": block_length,
            "resampling_unit": "contiguous blocks of saved regular-grid time frames",
            "randomness": "Private generator; no physics or global NumPy RNG consumption",
        },
        "policy": {
            "minimum_samples": MIN_SAMPLES,
            "minimum_complete_blocks": MIN_BLOCKS,
            "maximum_complete_blocks": MAX_BLOCKS,
            "minimum_effective_samples": MIN_EFFECTIVE_SAMPLES,
            "minimum_expected_bootstrap_tail_samples": 10,
            "default_block_length": "max(ceil(8*g), ceil(n/50), ceil(n**(1/3)))",
            "primary_interval": "Approximate non-overlapping batch-means Student-t; full mean, prefix complete batches for variance",
            "primary_standard_error": "sample_sd(complete_batch_means)*sqrt(block_length/n)",
            "secondary_interval": "Exploratory ordinary moving-block bootstrap percentile; no calibrated coverage claim",
            "assume_stationary": assume_stationary,
            "effective_sample_size": "n/g; g=max(1,estimated IAT), never particle count",
        },
        "assumptions": list(_ASSUMPTIONS),
        "limitations": list(_LIMITATIONS),
    }
