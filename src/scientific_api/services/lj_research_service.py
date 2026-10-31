"""Immutable, bounded uncertainty reports from committed saved LJ results."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.task_store import TaskStore
from ..domain.checkpoint import checkpoint_parameters, checkpoint_parameters_sha256
from ..domain.experiments import parameters_from_payload
from ..domain.lennard_jones import LennardJonesParameters
from ..research.lj_comparison import (
    COMPARISON_VERSION,
    LJReplicaSource,
    analyze_lj_replicas,
    analyze_lj_sensitivity,
)
from ..research.lj_uncertainty import UNCERTAINTY_VERSION, analyze_lj_uncertainty
from ..schemas.lj_research import (
    LJResearchReport,
    LJResearchRequest,
    LJResearchSource,
    ResearchAvailability,
)
from ..schemas.verification import VerificationSource
from .lj_research_source import (
    MAX_RESEARCH_SOURCE_BYTES,
    ResearchSourceError,
    _validate_pin,
    load_lj_research_source,
    prepare_lj_research_sources,
)
from .simulation_service import _environment
from .verification_service import VERIFICATION_ID, canonical_json

POLICY_VERSION = "lj-uncertainty-report-v1"
MAX_REPORT_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024


class ResearchError(ValueError):
    def __init__(self, code: str, status_code: int = 409):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def prepare_lj_research(backend: TaskStore, request: LJResearchRequest) -> dict[str, Any]:
    try:
        sources = prepare_lj_research_sources(backend, request.task_ids)
    except ResearchSourceError as error:
        raise ResearchError(error.code, error.status_code) from error
    payload = {
        "policy_version": POLICY_VERSION,
        "source_id": request.task_ids[0],
        "options": request.model_dump(mode="json"),
        "sources": sources,
    }
    encoded = canonical_json(payload)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ResearchError("RESEARCH_INPUT_LIMIT_EXCEEDED", 413)
    return json.loads(encoded)


def _validate_payload(payload: dict[str, Any], options: LJResearchRequest) -> None:
    if len(canonical_json(payload)) > MAX_REQUEST_BYTES:
        raise ResearchError("RESEARCH_INPUT_LIMIT_EXCEEDED", 413)
    pins = payload["sources"]
    if (
        payload.get("policy_version") != POLICY_VERSION
        or payload.get("source_id") != options.task_ids[0]
        or not isinstance(pins, list)
        or len(pins) != len(options.task_ids)
    ):
        raise ResearchError("RESEARCH_REQUEST_INTEGRITY_FAILED")
    total_bytes = 0
    source_ids = []
    for pin in pins:
        try:
            source, _, size = _validate_pin(pin)
        except (ValueError, KeyError, TypeError) as error:
            raise ResearchError("RESEARCH_REQUEST_INTEGRITY_FAILED") from error
        total_bytes += size
        source_ids.append(source.task_id)
        if total_bytes > MAX_RESEARCH_SOURCE_BYTES:
            raise ResearchError("RESEARCH_INPUT_LIMIT_EXCEEDED", 413)
    if source_ids != options.task_ids:
        raise ResearchError("RESEARCH_REQUEST_INTEGRITY_FAILED")


def _burn_in_step(parameters: LennardJonesParameters, time: float | None) -> int | None:
    if time is None:
        return None
    duration = parameters.num_steps * parameters.time_step
    tolerance = 16 * math.ulp(max(abs(time), abs(duration)))
    if time > duration + tolerance:
        raise ResearchError("RESEARCH_BURN_IN_OUTSIDE_RUN")
    # Select only recorded samples. ULP-aware physical-time selection avoids
    # excluding 0.3 merely because 3*0.1 is represented as 0.30000000000000004.
    for step in parameters.sampling_steps():
        saved_time = step * parameters.time_step
        tolerance = 16 * math.ulp(max(abs(time), abs(saved_time)))
        if step >= parameters.equilibration_steps and saved_time + tolerance >= time:
            return step
    raise ResearchError("RESEARCH_BURN_IN_OUTSIDE_RUN")


def _availability(statistics: dict[str, Any], options: LJResearchRequest) -> ResearchAvailability:
    """Count actual requested estimates, never arbitrary nested interval fields."""
    if statistics.get("model_version") != "lennard-jones-2d-v1":
        raise ValueError("Missing research model coverage")
    rows: list[dict[str, Any]] = []

    def add_metrics(
        metrics: Any, *, timeseries: bool = False, replica_count: int | None = None
    ) -> None:
        if not isinstance(metrics, dict) or set(metrics) != set(options.observables):
            raise ValueError("Missing requested observable coverage")
        for row in metrics.values():
            if not isinstance(row, dict) or "confidence_interval" not in row:
                raise ValueError("Missing conditional interval availability")
            if replica_count is not None and (
                type(row.get("replicas")) is not int
                or row["replicas"] != replica_count
                or not isinstance(row.get("replica_means"), list)
                or len(row["replica_means"]) != replica_count
            ):
                raise ValueError("Reported independent-unit counts disagree")
            interval = row["confidence_interval"]
            if interval is not None:
                if not isinstance(interval, dict) or not all(
                    type(interval.get(name)) in {float, int} and math.isfinite(interval[name])
                    for name in ("lower", "upper")
                ):
                    raise ValueError("Invalid finite interval")
                if (
                    interval["lower"] > interval["upper"]
                    or interval.get("level", interval.get("confidence_level"))
                    != options.confidence_level
                    or interval.get("coverage_guarantee") is not False
                    or not isinstance(interval.get("method"), str)
                ):
                    raise ValueError("Interval contradicts requested statistical policy")
            if timeseries and (
                row.get("status") not in {"available", "unavailable"}
                or (row["status"] == "available") != (interval is not None)
                or (interval is not None and not options.assume_stationary)
            ):
                raise ValueError("Time-series interval requires the declared assumptions")
            if interval is not None and timeseries:
                diagnostics = row.get("diagnostics", {})
                sampling = statistics.get("sampling", {})
                correlation = diagnostics.get("autocorrelation", {})
                if (
                    sampling.get("retained_sample_count", 0) < 128
                    or row.get("effective_sample_size", 0) < 20
                    or diagnostics.get("complete_block_count", 0) < 8
                    or diagnostics.get("complete_block_count", 0) > 50
                    or interval.get("degrees_of_freedom") != diagnostics["complete_block_count"] - 1
                    or row.get("block_length", 0)
                    < math.ceil(8 * row["integrated_autocorrelation_time_samples"])
                    or diagnostics.get("user_assumes_stationary") is not True
                    or diagnostics.get("stationarity_screen", {}).get("flags") != []
                    or correlation.get("positive_sequence_truncated") is not True
                    or correlation.get("positive_tail_reemergence") is not False
                    or correlation.get("persistent_late_correlation") is not False
                ):
                    raise ValueError("Time-series interval contradicts applicability diagnostics")
            if interval is not None and not timeseries:
                if (
                    row.get("replicas", 0) < 8
                    or interval.get("degrees_of_freedom") != row["replicas"] - 1
                ):
                    raise ValueError("Replica interval contradicts independent-unit count")
            rows.append(row)

    if options.kind == "timeseries":
        if (
            statistics.get("analysis_version") != UNCERTAINTY_VERSION
            or not isinstance(statistics.get("sampling"), dict)
            or statistics.get("bootstrap", {}).get("seed") != options.bootstrap_seed
            or statistics.get("bootstrap", {}).get("samples") != options.bootstrap_samples
        ):
            raise ValueError("Missing time-series policy coverage")
        add_metrics(statistics.get("metrics"), timeseries=True)
    else:
        if (
            statistics.get("analysis_version") != COMPARISON_VERSION
            or statistics.get("mode") != options.kind
        ):
            raise ValueError("Missing comparison policy coverage")
        if options.kind == "replicas":
            if statistics.get("replica_count") != len(options.task_ids):
                raise ValueError("Missing requested replica coverage")
            add_metrics(statistics.get("statistics"), replica_count=len(options.task_ids))
        else:
            groups, contrasts = statistics.get("groups"), statistics.get("contrasts")
            if (
                not isinstance(groups, list)
                or not 2 <= len(groups) <= 8
                or statistics.get("group_count") != len(groups)
                or statistics.get("parameter") != options.varied_parameter
                or not isinstance(contrasts, list)
                or len(contrasts) != len(groups) - 1
                or sum(group.get("replica_count", 0) for group in groups) != len(options.task_ids)
            ):
                raise ValueError("Missing requested sensitivity group coverage")
            for group in groups:
                add_metrics(group.get("statistics"), replica_count=group["replica_count"])
            for contrast in contrasts:
                add_metrics(
                    contrast.get("statistics"), replica_count=statistics["matched_replica_count"]
                )
    intervals = [row["confidence_interval"] is not None for row in rows]
    if all(intervals):
        return "available"
    return "partial" if any(intervals) else "unavailable"


def _implementation() -> dict[str, Any]:
    package = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for name in (
        "research/lj_uncertainty.py",
        "research/lj_comparison.py",
        "research/student_t.py",
        "services/lj_research_service.py",
        "services/lj_research_source.py",
    ):
        digest.update(name.encode("ascii") + b"\x00")
        digest.update((package / name).read_bytes())
    return {"sha256": digest.hexdigest(), "environment": _environment()}


def compute_lj_research(
    payload: dict[str, Any],
    *,
    analysis_id: str,
    analysis_attempt_id: str,
    artifacts_dir: Path,
    check_cancelled: Callable[[], None],
) -> LJResearchReport:
    started = datetime.now(UTC).isoformat()
    options = LJResearchRequest.model_validate(payload["options"])
    pins = payload["sources"]
    _validate_payload(payload, options)
    snapshots = []
    for pin in pins:
        check_cancelled()
        try:
            snapshots.append(
                load_lj_research_source(
                    pin, artifacts_dir=artifacts_dir, check_cancelled=check_cancelled
                )
            )
        except ResearchSourceError as error:
            raise ResearchError(error.code, error.status_code) from error
    shared = {
        "bootstrap_seed": options.bootstrap_seed,
        "bootstrap_samples": options.bootstrap_samples,
        "confidence_level": options.confidence_level,
        "observable_names": tuple(options.observables),
        "check_cancelled": check_cancelled,
    }
    check_cancelled()
    if options.kind == "timeseries":
        snapshot = snapshots[0]
        statistics = analyze_lj_uncertainty(
            snapshot.observables,
            snapshot.parameters,
            burn_in_step=_burn_in_step(snapshot.parameters, options.burn_in_time),
            block_length=options.block_length,
            assume_stationary=options.assume_stationary,
            **shared,
        )
    else:
        replicas = [
            LJReplicaSource(
                source_id=snapshot.envelope.task_id,
                attempt_id=str(snapshot.envelope.attempt_id),
                status="completed",
                parameters=snapshot.parameters,
                sample_steps=np.asarray(snapshot.observables["sample_steps"], dtype=np.int64),
                series={
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in snapshot.observables["series"].items()
                },
            )
            for snapshot in snapshots
        ]
        if options.kind == "replicas":
            statistics = analyze_lj_replicas(replicas, burn_in_time=options.burn_in_time, **shared)
        else:
            parameter = options.varied_parameter
            assert parameter is not None
            groups: dict[float, list[LJReplicaSource]] = {}
            for replica in replicas:
                groups.setdefault(float(getattr(replica.parameters, parameter)), []).append(replica)
            statistics = analyze_lj_sensitivity(
                list(groups.values()),
                parameter=parameter,
                burn_in_time=options.burn_in_time,
                **shared,
            )
    check_cancelled()
    return LJResearchReport(
        analysis_id=analysis_id,
        analysis_attempt_id=analysis_attempt_id,
        kind=options.kind,
        options=options,
        sources=[
            LJResearchSource(
                identity=VerificationSource.model_validate(snapshot.pin["source"]),
                parameters=checkpoint_parameters(snapshot.parameters),
            )
            for snapshot in snapshots
        ],
        started_at=started,
        completed_at=datetime.now(UTC).isoformat(),
        availability=_availability(statistics, options),
        statistics=statistics,
        implementation=_implementation(),
        limitations=[
            "Intervals are conditional pointwise estimates; diagnostics do not prove equilibrium or stationarity.",
            "Time blocks and whole logical replicas are the uncertainty units; particles, frames and RDF bins are not independent replicas.",
            "Replica intervals estimate the specified finite-window seeded protocol, not a canonical ensemble by default.",
            "A sensitivity contrast measures an observed parameter change, not bias relative to an unknown limiting solution.",
            "Parameter uncertainty, finite-size effects, experimental calibration and physical-model discrepancy are not included in these intervals.",
            "Source selection and burn-in choices are declared by the caller; data-dependent selection may invalidate interval interpretation.",
            "The report is unsigned and describes pinned saved data at analysis time; it does not change the source results or verification certificates.",
        ],
    )


def research_report_bytes(report: LJResearchReport) -> bytes:
    data = canonical_json(report.model_dump(mode="json"))
    if len(data) > MAX_REPORT_BYTES:
        raise ResearchError("RESEARCH_REPORT_LIMIT_EXCEEDED", 413)
    return data


def _validate_source_coverage(report: LJResearchReport) -> None:
    if report.kind == "timeseries":
        return
    expected = {source.identity.task_id: source for source in report.sources}
    groups = [report.statistics] if report.kind == "replicas" else report.statistics["groups"]
    observed_ids = []
    seed_sets = []
    for index, group in enumerate(groups):
        sources = group.get("sources")
        if not isinstance(sources, list) or len(sources) != group["replica_count"]:
            raise ValueError("Replica groups omit their actual independent units")
        seeds, attempts = [], []
        for source in sources:
            saved = expected[source["source_id"]]
            if (
                source["attempt_id"] != saved.identity.attempt_id
                or source["seed"] != saved.parameters["seed"]
            ):
                raise ValueError("Reported replica identities disagree with pinned sources")
            observed_ids.append(source["source_id"])
            seeds.append(source["seed"])
            attempts.append(source["attempt_id"])
            group_parameters = parameters_from_payload(
                {
                    **group["parameters_without_seed"],
                    "model": report.model_version,
                    "seed": saved.parameters["seed"],
                }
            )
            if checkpoint_parameters(group_parameters) != saved.parameters:
                raise ValueError("Replica group mixes different physical protocols")
            if report.kind == "sensitivity":
                parameter = report.options.varied_parameter
                if (
                    parameter is None
                    or saved.parameters[parameter] != report.statistics["parameter_values"][index]
                ):
                    raise ValueError("Reported sensitivity parameter disagrees with source physics")
        if len(set(seeds)) != len(seeds) or len(set(attempts)) != len(attempts):
            raise ValueError("Execution attempts and duplicate seeds are not independent replicas")
        seed_sets.append(sorted(seeds))
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(expected):
        raise ValueError("Report omits or duplicates requested logical replicas")
    if report.kind == "sensitivity" and (
        any(seeds != seed_sets[0] for seeds in seed_sets)
        or report.statistics["matched_replica_count"] != len(seed_sets[0])
        or report.statistics["matched_seeds"] != seed_sets[0]
    ):
        raise ValueError("Sensitivity contrasts must use the complete matched seed set")


def validate_lj_research_report(
    data: bytes, *, expected_sha256: str | None = None
) -> LJResearchReport:
    """Offline structure/hash validation; this does not repeat statistical analysis."""
    if len(data) > MAX_REPORT_BYTES:
        raise ResearchError("RESEARCH_REPORT_LIMIT_EXCEEDED", 413)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ResearchError("RESEARCH_REPORT_INTEGRITY_FAILED")
    try:
        report = LJResearchReport.model_validate_json(data)
        if (
            research_report_bytes(report) != data
            or report.kind != report.options.kind
            or report.options.task_ids != [source.identity.task_id for source in report.sources]
            or any(
                source.identity.model_version != report.model_version for source in report.sources
            )
            or report.availability != _availability(report.statistics, report.options)
            or not VERIFICATION_ID.fullmatch(report.analysis_id)
        ):
            raise ValueError("Research report metadata disagrees")
        for source in report.sources:
            parameters = parameters_from_payload(source.parameters)
            if (
                not isinstance(parameters, LennardJonesParameters)
                or checkpoint_parameters(parameters) != source.parameters
                or checkpoint_parameters_sha256(parameters) != source.identity.parameters_sha256
            ):
                raise ValueError("Reported physical parameters disagree with the pinned digest")
        _validate_source_coverage(report)
        return report
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ResearchError("RESEARCH_REPORT_INVALID") from error
