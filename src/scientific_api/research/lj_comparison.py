"""Finite-window LJ replica uncertainty and matched saved-run sensitivity.

The independent statistical units here are entire seeded simulations. Sampled
particles, frames and RDF bins never become additional replicas. Nothing in
this module executes dynamics, establishes equilibration or estimates a true
zero-time-step or infinite-cutoff limit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from .lj_analysis import SERIES_NAMES
from .student_t import student_t_critical as _student_t_critical

COMPARISON_VERSION = "lj-replica-comparison-v1"
MIN_INTERVAL_REPLICAS = 8
MAX_REPLICAS = 50
MAX_SENSITIVITY_GROUPS = 8
MIN_BOOTSTRAP_TAIL_SAMPLES = 10
_CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)
_TAIL_PROBABILITIES = {0.9: 0.05, 0.95: 0.025, 0.99: 0.005}
_UNITS = {
    "thermal_temperature": "energy/k_B; k_B=1",
    "kinetic_energy": "energy",
    "potential_energy": "energy",
    "total_energy": "energy",
    "pressure": "energy/area",
    "virial": "energy",
}


@dataclass(frozen=True, slots=True)
class LJReplicaSource:
    """One explicitly requested logical replica, already pinned by its caller."""

    source_id: str
    attempt_id: str
    status: str
    parameters: LennardJonesParameters
    sample_steps: NDArray[np.int64]
    series: Mapping[str, NDArray[np.float64]]


def _options(
    burn_in_time: float | None,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence_level: float,
    observable_names: tuple[str, ...],
) -> None:
    if burn_in_time is not None and (
        isinstance(burn_in_time, bool)
        or not isinstance(burn_in_time, (int, float))
        or not math.isfinite(burn_in_time)
        or burn_in_time < 0
    ):
        raise ValueError("burn_in_time must be a finite nonnegative model time")
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed <= 2**64 - 1:
        raise ValueError("bootstrap_seed must be an integer in 0..2**64-1")
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 10_000:
        raise ValueError("bootstrap_samples must be an integer in 100..10000")
    if isinstance(confidence_level, bool) or confidence_level not in _CONFIDENCE_LEVELS:
        raise ValueError("confidence_level must be 0.90, 0.95 or 0.99")
    if (
        not isinstance(observable_names, tuple)
        or not 1 <= len(observable_names) <= len(SERIES_NAMES)
        or any(not isinstance(name, str) for name in observable_names)
        or len(set(observable_names)) != len(observable_names)
        or any(name not in SERIES_NAMES for name in observable_names)
    ):
        raise ValueError("observable_names must select distinct supported observables")


def _cancel(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _source_mean(
    source: LJReplicaSource,
    burn_in_time: float | None,
    names: tuple[str, ...],
) -> tuple[list[float], dict[str, Any]]:
    parameters = source.parameters
    if not isinstance(parameters, LennardJonesParameters):
        raise ValueError("Every replica must have validated LJ parameters")
    steps = source.sample_steps
    if (
        not isinstance(steps, np.ndarray)
        or steps.dtype != np.dtype(np.int64)
        or not np.array_equal(steps, np.asarray(parameters.sampling_steps(), dtype=np.int64))
    ):
        raise ValueError("Replica sample_steps do not match its archived sampling schedule")
    if not isinstance(source.series, Mapping) or set(source.series) != set(SERIES_NAMES):
        raise ValueError("Replica series must contain exactly the six saved LJ observables")
    for name, values in source.series.items():
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(np.float64)
            or values.shape != steps.shape
            or not np.isfinite(values).all()
            or (name in {"thermal_temperature", "kinetic_energy"} and (values < 0).any())
        ):
            raise ValueError("Replica series must be finite native float64 sampled observables")
    times = steps.astype(np.float64) * parameters.time_step
    archived_burn_in = parameters.equilibration_steps * parameters.time_step
    threshold = max(archived_burn_in, burn_in_time or 0.0)
    # A saved terminal frame can be shorter than sample_stride. Exclude it to
    # retain the same regular saved-frame estimator as the single-run analysis.
    regular = steps % parameters.sample_stride == 0
    tolerance = 16.0 * math.ulp(max(threshold, float(times[-1])))
    selected = regular & (times >= threshold - tolerance)
    if not np.any(selected):
        raise ValueError("Replica has no regular production samples after the chosen exclusion")
    selected_steps = steps[selected]
    selected_times = times[selected]
    means = [
        math.fsum(float(value) / len(selected_steps) for value in source.series[name][selected])
        for name in names
    ]
    if not all(math.isfinite(value) for value in means):
        raise ValueError("Replica means must be representable as finite float64 values")
    return means, {
        "sample_steps": selected_steps.tolist(),
        "sample_times": selected_times.tolist(),
        "retained_samples": len(selected_steps),
        "archived_equilibration_time": archived_burn_in,
        "requested_burn_in_time": burn_in_time,
        "effective_burn_in_time": threshold,
        "saved_frames": len(steps),
        "dropped_irregular_terminal": bool(not regular[-1]),
        "method": "Arithmetic mean of retained regularly spaced saved frames; no unseen frames reconstructed",
    }


def _validated_group(
    sources: Sequence[LJReplicaSource],
    *,
    burn_in_time: float | None,
    names: tuple[str, ...],
    check_cancelled: Callable[[], None] | None,
) -> tuple[list[LJReplicaSource], NDArray[np.float64], dict[str, Any]]:
    if not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= MAX_REPLICAS:
        raise ValueError("An exact requested replica group must contain 1..50 sources")
    if any(not isinstance(source, LJReplicaSource) for source in sources):
        raise ValueError("Every requested replica must be an LJReplicaSource")
    if any(source.status != "completed" for source in sources):
        raise ValueError(
            "All requested replicas must be completed; survivor aggregation is forbidden"
        )
    for field in ("source_id", "attempt_id"):
        identifiers = [getattr(source, field) for source in sources]
        if any(
            not isinstance(value, str) or not value or len(value) > 128 for value in identifiers
        ):
            raise ValueError(
                "Replica source and attempt identities must be bounded nonempty strings"
            )
        if len(set(identifiers)) != len(sources):
            raise ValueError("Duplicate replica source or attempt identity")
    if any(not isinstance(source.parameters, LennardJonesParameters) for source in sources):
        raise ValueError("Every replica must have validated LJ parameters")
    seeds = [source.parameters.seed for source in sources]
    if len(set(seeds)) != len(sources):
        raise ValueError(
            "Replica physical seeds must be distinct; task identifiers do not create independence"
        )
    ordered = sorted(sources, key=lambda source: source.parameters.seed)
    protocol = asdict(ordered[0].parameters)
    protocol.pop("seed")
    rows, sampling = [], None
    for source in ordered:
        _cancel(check_cancelled)
        other = asdict(source.parameters)
        other.pop("seed")
        if other != protocol:
            raise ValueError("Replica physics, execution backend and sampling protocol must match")
        mean, window = _source_mean(source, burn_in_time, names)
        if sampling is not None and sampling != window:
            raise ValueError("Replica production sampling windows must match")
        sampling = window
        rows.append(mean)
    assert sampling is not None
    return ordered, np.asarray(rows, dtype=np.float64), sampling


def _summaries(
    values: NDArray[np.float64],
    *,
    names: tuple[str, ...],
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence_level: float,
    check_cancelled: Callable[[], None] | None,
) -> dict[str, Any]:
    count = len(values)
    critical = (
        _student_t_critical(confidence_level, count - 1) if count >= MIN_INTERVAL_REPLICAS else None
    )
    expected_tail_samples = bootstrap_samples * _TAIL_PROBABILITIES[confidence_level]
    resolved_tails = expected_tail_samples >= MIN_BOOTSTRAP_TAIL_SAMPLES
    bootstrap = None
    varying = np.any(values != values[0], axis=0)
    if count >= MIN_INTERVAL_REPLICAS and resolved_tails and varying.any():
        bootstrap = np.empty((bootstrap_samples, len(names)), dtype=np.float64)
        generator = np.random.Generator(np.random.PCG64(bootstrap_seed))
        for start in range(0, bootstrap_samples, 128):
            _cancel(check_cancelled)
            stop = min(start + 128, bootstrap_samples)
            indices = generator.integers(0, count, size=(stop - start, count))
            # Use the same whole-replica resample for all observables. Scaling
            # before summing avoids overflowing an otherwise finite mean.
            for column in range(len(names)):
                if not varying[column]:
                    continue
                bootstrap[start:stop, column] = np.sum(
                    values[indices, column] / count, axis=1, dtype=np.float64
                )
    result = {}
    for column, name in enumerate(names):
        _cancel(check_cancelled)
        observed = values[:, column]
        mean = math.fsum(float(value) / count for value in observed)
        scale = max(abs(float(value) - mean) for value in observed)
        deviation = (
            scale
            * math.sqrt(
                math.fsum(((float(value) - mean) / scale) ** 2 for value in observed) / (count - 1)
            )
            if count > 1 and scale > 0.0
            else 0.0
            if count > 1
            else None
        )
        standard_error = deviation / math.sqrt(count) if deviation is not None else None
        interval = exploratory = None
        reason = (
            "insufficient_replicas" if count < MIN_INTERVAL_REPLICAS else "unresolved_variation"
        )
        exploratory_reason = reason
        if critical is not None and standard_error is not None and scale > 0.0:
            half_width = critical * standard_error
            interval = {
                "lower": mean - half_width,
                "upper": mean + half_width,
                "confidence_level": confidence_level,
                "method": "Student-t on independent whole-replica means",
                "degrees_of_freedom": count - 1,
                "critical_value": critical,
                "assumption": "Independent identically distributed replica means with approximately normal sampling distribution",
                "coverage_guarantee": False,
            }
            if resolved_tails:
                assert bootstrap is not None
                alpha = _TAIL_PROBABILITIES[confidence_level]
                lower, upper = np.quantile(
                    bootstrap[:, column], [alpha, 1.0 - alpha], method="linear"
                )
                exploratory = {
                    "lower": float(lower),
                    "upper": float(upper),
                    "nominal_level": confidence_level,
                    "method": "Exploratory percentile bootstrap of whole-replica means",
                    "coverage_guarantee": False,
                }
                exploratory_reason = "exploratory_percentile_bootstrap"
            else:
                exploratory_reason = "bootstrap_quantile_resolution_insufficient"
            reason = "conditional_approximate_normal_replica_means"
        numbers = [mean, *([] if deviation is None else [deviation, standard_error])]
        if interval is not None:
            numbers.extend([interval["lower"], interval["upper"]])
        if exploratory is not None:
            numbers.extend([exploratory["lower"], exploratory["upper"]])
        if any(value is None or not math.isfinite(value) for value in numbers):
            raise ValueError("Replica statistics cannot be represented as finite float64 values")
        result[name] = {
            "replica_means": observed.tolist(),
            "replicas": count,
            "mean": mean,
            "sample_standard_deviation": deviation,
            "standard_error": standard_error,
            "standard_error_interpretation": "Descriptive across independent seeded finite-window replica means",
            "confidence_interval": interval,
            "interval_reason": reason,
            "exploratory_bootstrap_percentile": exploratory,
            "exploratory_bootstrap_reason": exploratory_reason,
            "bootstrap_expected_tail_samples": expected_tail_samples,
            "bootstrap_minimum_tail_samples": MIN_BOOTSTRAP_TAIL_SAMPLES,
            "unit": _UNITS[name],
        }
    return result


def _limitations() -> list[str]:
    return [
        "Each independent unit is one whole logical seeded replica; particles, time frames, RDF bins and execution attempts are not replicas",
        "All explicitly requested replicas are required; no incomplete or failed replica is silently excluded",
        "The estimand is the expected saved finite-window mean under the declared seeded initialization and dynamics, not an established equilibrium or canonical thermodynamic expectation",
        "Distinct seeds operationally identify random realizations; metadata cannot prove physical independence",
        "Student-t intervals require approximately normal independent identically distributed replica means; eight replicas is a conservative reporting policy, not a coverage guarantee",
        "Percentile bootstrap intervals are exploratory and can substantially under-cover at small replica counts",
        "Exploratory bootstrap intervals require at least ten expected resamples in each nominal tail; this resolution threshold does not establish calibrated coverage",
        "Intervals are pointwise per observable and contrast; no simultaneous multiple-comparison coverage is claimed",
        "Unresolved between-replica variation suppresses intervals; identical saved means do not establish zero physical uncertainty",
    ]


def analyze_lj_replicas(
    sources: Sequence[LJReplicaSource],
    *,
    burn_in_time: float | None = None,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    observable_names: tuple[str, ...] = SERIES_NAMES,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Analyze exactly the requested completed independent LJ replicas."""
    _options(burn_in_time, bootstrap_seed, bootstrap_samples, confidence_level, observable_names)
    _cancel(check_cancelled)
    ordered, values, sampling = _validated_group(
        sources,
        burn_in_time=burn_in_time,
        names=observable_names,
        check_cancelled=check_cancelled,
    )
    return {
        "analysis_version": COMPARISON_VERSION,
        "model_version": LJ_MODEL_VERSION,
        "mode": "replicas",
        "replica_count": len(ordered),
        "failure_policy": "require_all_requested",
        "independent_unit": "whole logical seeded replica",
        "minimum_interval_replicas": MIN_INTERVAL_REPLICAS,
        "sources": [
            {
                "source_id": source.source_id,
                "attempt_id": source.attempt_id,
                "seed": source.parameters.seed,
            }
            for source in ordered
        ],
        "parameters_without_seed": {
            key: value for key, value in asdict(ordered[0].parameters).items() if key != "seed"
        },
        "sampling": sampling,
        "bootstrap": {
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "bit_generator": "PCG64",
            "resampling_unit": "whole logical seeded replica mean",
            "expected_tail_samples": bootstrap_samples * _TAIL_PROBABILITIES[confidence_level],
            "minimum_tail_samples": MIN_BOOTSTRAP_TAIL_SAMPLES,
        },
        "statistics": _summaries(
            values,
            names=observable_names,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            check_cancelled=check_cancelled,
        ),
        "limitations": _limitations(),
    }


def _same_times(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(a - b) <= 32.0 * max(math.ulp(a), math.ulp(b)) for a, b in zip(left, right, strict=True)
    )


def analyze_lj_sensitivity(
    groups: Sequence[Sequence[LJReplicaSource]],
    *,
    parameter: Literal["time_step", "cutoff_radius"],
    burn_in_time: float | None = None,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    observable_names: tuple[str, ...] = SERIES_NAMES,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Describe paired saved-run changes across one declared parameter only."""
    _options(burn_in_time, bootstrap_seed, bootstrap_samples, confidence_level, observable_names)
    if parameter not in {"time_step", "cutoff_radius"}:
        raise ValueError("Sensitivity parameter must be time_step or cutoff_radius")
    if not isinstance(groups, (list, tuple)) or not 2 <= len(groups) <= MAX_SENSITIVITY_GROUPS:
        raise ValueError("Sensitivity requires 2..8 explicitly requested replica groups")
    prepared = [
        _validated_group(
            group,
            burn_in_time=burn_in_time,
            names=observable_names,
            check_cancelled=check_cancelled,
        )
        for group in groups
    ]
    identities = [source for group, _, _ in prepared for source in group]
    for field in ("source_id", "attempt_id"):
        if len({getattr(source, field) for source in identities}) != len(identities):
            raise ValueError("A source or attempt cannot occur in multiple sensitivity groups")
    seeds = [source.parameters.seed for source in prepared[0][0]]
    first_parameters = prepared[0][0][0].parameters
    excluded = {"seed", parameter}
    if parameter == "time_step":
        excluded.update(("num_steps", "sample_stride", "equilibration_steps"))
    protocol = {
        key: value for key, value in asdict(first_parameters).items() if key not in excluded
    }
    values, reports = [], []
    first_sampling = prepared[0][2]
    for ordered, _, sampling in prepared:
        _cancel(check_cancelled)
        parameters = ordered[0].parameters
        if [source.parameters.seed for source in ordered] != seeds:
            raise ValueError("Sensitivity groups must contain the same complete physical seed set")
        other = {key: value for key, value in asdict(parameters).items() if key not in excluded}
        if other != protocol:
            raise ValueError(
                "Sensitivity may vary only the declared parameter and matched time-grid fields"
            )
        if not _same_times(first_sampling["sample_times"], sampling["sample_times"]):
            raise ValueError("Sensitivity production saved-frame physical timelines must match")
        for name in ("effective_burn_in_time", "archived_equilibration_time"):
            if not _same_times([first_sampling[name]], [sampling[name]]):
                raise ValueError("Sensitivity physical exclusions must match")
        if not _same_times(
            [first_parameters.num_steps * first_parameters.time_step],
            [parameters.num_steps * parameters.time_step],
        ):
            raise ValueError("Sensitivity physical simulation durations must match")
        value = float(getattr(parameters, parameter))
        if value in values:
            raise ValueError("Sensitivity groups must have distinct declared parameter values")
        values.append(value)
        reports.append(
            analyze_lj_replicas(
                ordered,
                burn_in_time=burn_in_time,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                observable_names=observable_names,
                check_cancelled=check_cancelled,
            )
        )
    contrasts = []
    for index in range(1, len(prepared)):
        _cancel(check_cancelled)
        with np.errstate(over="raise", invalid="raise"):
            try:
                differences = prepared[index][1] - prepared[0][1]
            except FloatingPointError as error:
                raise ValueError(
                    "Paired differences cannot be represented as finite values"
                ) from error
        contrasts.append(
            {
                "reference_group": 0,
                "comparison_group": index,
                "reference_parameter_value": values[0],
                "comparison_parameter_value": values[index],
                "method": "Matched physical-seed whole-replica differences; comparison minus reference",
                "statistics": _summaries(
                    differences,
                    names=observable_names,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    check_cancelled=check_cancelled,
                ),
            }
        )
    return {
        "analysis_version": COMPARISON_VERSION,
        "model_version": LJ_MODEL_VERSION,
        "mode": "sensitivity",
        "parameter": parameter,
        "parameter_values": values,
        "reference_group": 0,
        "group_count": len(groups),
        "matched_replica_count": len(seeds),
        "matched_seeds": seeds,
        "groups": reports,
        "contrasts": contrasts,
        "interpretation": "Observed finite-window changes across saved runs; no true convergence, bias correction, equivalence or error-to-limit claim",
        "limitations": [
            *_limitations(),
            "Changing shifted-force cutoff changes the physical model; finite cutoff sensitivity is not a tail correction or infinite-cutoff estimate",
            "Shared physical seeds define matching; trajectories may decorrelate and matching is not a cross-backend equivalence guarantee",
            "A contrast interval containing zero does not establish equivalence or convergence",
        ],
    }
