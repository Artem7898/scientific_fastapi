"""Pin committed LJ occurrences and load bounded, byte-verified observable snapshots.

Admission consults the immutable task archive only. Execution rechecks its exact
anchors against current files, retaining the same JSON bytes that were hashed.
No particle archive is decoded, no dynamics are executed, and no source is written.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.task_store import TaskStore
from ..domain.checkpoint import checkpoint_parameters_sha256
from ..domain.experiments import parameters_from_payload
from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from ..research.lj_analysis import ANALYSIS_VERSION, SERIES_NAMES
from ..schemas.result_envelope import ResultEnvelope
from ..schemas.verification import VerificationSource
from .provenance_evidence import artifact_occurrence_id, build_archive_evidence
from .result_envelope_service import MAX_ARRAY_BYTES, _root, _snapshot, read_result_envelope
from .verification_science import _json_artifact
from .verification_service import (
    IDENTIFIER,
    POLICY_VERSION,
    VerificationError,
    canonical_json,
    content_sha256,
    prepare_verification,
)

MAX_RESEARCH_SOURCES = 50
MAX_RESEARCH_SOURCE_BYTES = 512 * 1024 * 1024
MAX_RESEARCH_PIN_BYTES = 1024 * 1024
MAX_OBSERVABLES_BYTES = 2 * 1024 * 1024
_ARTIFACT_FILES = {
    "manifest": "manifest.json",
    "final_state": "final_state.npz",
    "trajectory": "trajectory.npz",
    "observables": "observables.json",
}


class ResearchSourceError(ValueError):
    """Safe machine-readable admission or source-integrity refusal."""

    def __init__(self, code: str, status_code: int = 409):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class LJResearchSourceSnapshot:
    pin: dict[str, Any]
    envelope: ResultEnvelope
    parameters: LennardJonesParameters
    observables: dict[str, Any]


def _validate_pin(pin: dict[str, Any]) -> tuple[VerificationSource, LennardJonesParameters, int]:
    if not isinstance(pin, dict) or set(pin) != {
        "source_id",
        "source",
        "record",
        "parameters",
        "evidence",
        "options",
        "policy_version",
    }:
        raise ValueError("Invalid pinned source fields")
    source = VerificationSource.model_validate(pin["source"])
    record, evidence = pin["record"], pin["evidence"]
    if (
        not isinstance(record, dict)
        or not isinstance(evidence, dict)
        or pin["source_id"] != source.task_id
        or not IDENTIFIER.fullmatch(source.task_id)
        or not IDENTIFIER.fullmatch(source.attempt_id)
        or source.task_id != record.get("task_id")
        or source.attempt_id != record.get("attempt_id")
        or record.get("status") != "completed"
        or source.model_version != LJ_MODEL_VERSION
        or record.get("model_version") != LJ_MODEL_VERSION
        or evidence.get("model_version") != LJ_MODEL_VERSION
        or type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
        or pin["policy_version"] != POLICY_VERSION
        or not isinstance(pin["options"], dict)
        or set(pin["options"]) != {"rerun"}
        or pin["options"]["rerun"] is not False
        or source.evidence_sha256 != content_sha256(evidence)
        or source.manifest_sha256 != evidence.get("manifest_sha256")
        or source.model_contract_sha256 != evidence.get("model_contract_sha256")
    ):
        raise ValueError("Pinned source identity disagrees")
    parameters = parameters_from_payload(pin["parameters"])
    if (
        not isinstance(parameters, LennardJonesParameters)
        or checkpoint_parameters_sha256(parameters) != source.parameters_sha256
    ):
        raise ValueError("Pinned physical parameters disagree")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(_ARTIFACT_FILES):
        raise ValueError("Invalid LJ artifact set")
    by_name = {}
    total_bytes = 0
    prefix = f"{source.task_id}/{source.attempt_id}"
    for descriptor in artifacts:
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid artifact descriptor")
        name = descriptor.get("name")
        if not isinstance(name, str) or name not in _ARTIFACT_FILES or name in by_name:
            raise ValueError("Invalid LJ artifact name")
        maximum = (
            MAX_ARRAY_BYTES if name in {"final_state", "trajectory"} else MAX_OBSERVABLES_BYTES
        )
        if (
            type(descriptor.get("bytes")) is not int
            or not 0 <= descriptor["bytes"] <= maximum
            or descriptor.get("path") != f"{prefix}/{_ARTIFACT_FILES[name]}"
            or descriptor.get("artifact_id")
            != artifact_occurrence_id(source.task_id, source.attempt_id, None, name)
            or not isinstance(descriptor.get("sha256"), str)
            or len(descriptor["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in descriptor["sha256"])
        ):
            raise ValueError("Invalid LJ artifact anchor")
        total_bytes += descriptor["bytes"]
        by_name[name] = descriptor
    if (
        evidence.get("manifest") != by_name["manifest"]
        or by_name["manifest"]["sha256"] != source.manifest_sha256
        or record.get("artifacts") != {name: item["path"] for name, item in by_name.items()}
    ):
        raise ValueError("Manifest and result artifact anchors disagree")
    return source, parameters, total_bytes


def prepare_lj_research_sources(backend: TaskStore, task_ids: list[str]) -> list[dict[str, Any]]:
    """Copy archive-only source pins; absent or corrupt present-day files are allowed."""
    if not isinstance(task_ids, list) or not 1 <= len(task_ids) <= MAX_RESEARCH_SOURCES:
        raise ResearchSourceError("RESEARCH_INPUT_LIMIT_EXCEEDED", 413)
    if any(not isinstance(task, str) or not IDENTIFIER.fullmatch(task) for task in task_ids):
        raise ResearchSourceError("RESEARCH_SOURCE_NOT_FOUND", 404)
    if len(set(task_ids)) != len(task_ids):
        raise ResearchSourceError("RESEARCH_SOURCE_DUPLICATE")
    pins: list[dict[str, Any]] = []
    total_bytes, pin_bytes = 0, 2
    for task_id in task_ids:
        try:
            archive = backend.get_task_archive(task_id)
        except Exception as error:
            raise ResearchSourceError("RESEARCH_STORAGE_UNAVAILABLE", 503) from error
        if archive is None:
            raise ResearchSourceError("RESEARCH_SOURCE_NOT_FOUND", 404)
        if not isinstance(archive, dict):
            raise ResearchSourceError("RESEARCH_SOURCE_EVIDENCE_INVALID")
        record = archive.get("record", {})
        if not isinstance(record, dict) or record.get("status") != "completed":
            raise ResearchSourceError("RESEARCH_SOURCE_NOT_COMPLETED")
        if not isinstance(archive.get("evidence"), dict):
            raise ResearchSourceError("RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE")
        if record.get("model_version") != LJ_MODEL_VERSION:
            raise ResearchSourceError("RESEARCH_SOURCE_MODEL_UNSUPPORTED")
        try:
            pin = prepare_verification(archive, rerun=False)
            source, _, byte_count = _validate_pin(pin)
            if source.task_id != task_id:
                raise ValueError("Archive identity disagrees with requested task")
            encoded = canonical_json(pin)
        except VerificationError as error:
            code = {
                "VERIFICATION_SOURCE_EVIDENCE_UNAVAILABLE": "RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE",
                "VERIFICATION_SOURCE_LIMIT_EXCEEDED": "RESEARCH_INPUT_LIMIT_EXCEEDED",
            }.get(error.code, "RESEARCH_SOURCE_EVIDENCE_INVALID")
            raise ResearchSourceError(
                code, 413 if code == "RESEARCH_INPUT_LIMIT_EXCEEDED" else 409
            ) from error
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ResearchSourceError("RESEARCH_SOURCE_EVIDENCE_INVALID") from error
        total_bytes += byte_count
        pin_bytes += len(encoded) + (1 if pins else 0)
        if total_bytes > MAX_RESEARCH_SOURCE_BYTES or pin_bytes > MAX_RESEARCH_PIN_BYTES:
            raise ResearchSourceError("RESEARCH_INPUT_LIMIT_EXCEEDED", 413)
        pins.append(pin)
    return pins


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _validate_observables(observables: dict[str, Any], parameters: LennardJonesParameters) -> None:
    schedule = list(parameters.sampling_steps())
    frames = len(schedule)
    steps, times, series = (observables.get(name) for name in ("sample_steps", "times", "series"))
    if (
        not 2 <= frames <= 1001
        or observables.get("analysis_version") != ANALYSIS_VERSION
        or observables.get("model_version") != LJ_MODEL_VERSION
        or type(observables.get("dimensions")) is not int
        or observables["dimensions"] != 2
        or type(observables.get("sample_size")) is not int
        or observables["sample_size"] != parameters.num_particles
        or observables.get("ensemble") != parameters.ensemble
        or not isinstance(steps, list)
        or len(steps) != frames
        or any(type(step) is not int for step in steps)
        or steps != schedule
        or not isinstance(times, list)
        or len(times) != frames
        or any(
            not _finite_number(value)
            or not math.isclose(value, step * parameters.time_step, rel_tol=1e-12, abs_tol=1e-12)
            for step, value in zip(schedule, times, strict=True)
        )
        or not isinstance(series, dict)
        or set(series) != set(SERIES_NAMES)
    ):
        raise ValueError("Saved LJ observable metadata is invalid")
    for values in series.values():
        if (
            not isinstance(values, list)
            or len(values) != frames
            or any(not _finite_number(value) for value in values)
        ):
            raise ValueError("Saved LJ series must be finite bounded sampled vectors")
    production = observables.get("production")
    selected = [step for step in schedule if step >= parameters.equilibration_steps]
    if not isinstance(production, dict) or any(
        type(production.get(name)) is not int or production[name] != expected
        for name, expected in (
            ("requested_equilibration_steps", parameters.equilibration_steps),
            ("first_sample_step", selected[0]),
            ("last_sample_step", selected[-1]),
            ("sample_count", len(selected)),
        )
    ):
        raise ValueError("Saved LJ production sampling metadata disagrees")


def load_lj_research_source(
    pin: dict[str, Any],
    *,
    artifacts_dir: Path,
    check_cancelled: Callable[[], None],
) -> LJResearchSourceSnapshot:
    """Read the exact pinned JSON bytes without decoding NPZ or rerunning a model."""
    check_cancelled()
    try:
        encoded = canonical_json(pin)
        if len(encoded) > MAX_RESEARCH_PIN_BYTES:
            raise ValueError("Pinned source exceeds the request limit")
        detached = json.loads(encoded)
        source, parameters, _ = _validate_pin(detached)
        envelope = read_result_envelope(
            detached["record"], artifacts_dir=artifacts_dir, require_present=True
        )
        check_cancelled()
        if (
            envelope.kind != "simulation"
            or envelope.analysis_id is not None
            or envelope.task_id != source.task_id
            or envelope.attempt_id != source.attempt_id
            or envelope.manifest is None
            or envelope.manifest.sha256 != source.manifest_sha256
            or envelope.model_contract is None
            or envelope.model_contract.model_version != LJ_MODEL_VERSION
            or envelope.model_contract.sha256 != source.model_contract_sha256
            or build_archive_evidence(envelope) != detached["evidence"]
        ):
            raise ValueError("Current files disagree with the archived source occurrence")
        current_parameters = parameters_from_payload(
            {**envelope.parameters, "model": LJ_MODEL_VERSION}
        )
        if checkpoint_parameters_sha256(current_parameters) != source.parameters_sha256:
            raise ValueError("Current parameters disagree with the archived source")
        descriptors = [item for item in envelope.artifacts if item.name == "observables"]
        if len(descriptors) != 1:
            raise ValueError("A source must have exactly one saved observable file")
        descriptor = descriptors[0]
        check_cancelled()
        digest, size, raw = _snapshot(
            _root(artifacts_dir), descriptor.path, MAX_OBSERVABLES_BYTES, retain=True
        )
        if raw is None or digest != descriptor.sha256 or size != descriptor.bytes:
            raise ValueError("Saved observables changed after envelope verification")
        observables = _json_artifact(raw)
        _validate_observables(observables, parameters)
        check_cancelled()
        return LJResearchSourceSnapshot(detached, envelope, parameters, observables)
    except (OSError, KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ResearchSourceError("RESEARCH_SOURCE_INTEGRITY_FAILED") from error
