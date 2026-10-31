"""Build bounded lineage from the durable journal and rechecked archived evidence.

The archive is the authority for committed occurrences. Matching files alone
cannot create history. Integrity here is relative to unsigned saved metadata;
it does not identify a human or independently verify a scientific result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.task_store import TaskStore
from ..schemas.provenance import (
    ProvenanceAnalysis,
    ProvenanceArtifact,
    ProvenanceAttempt,
    ProvenanceEvent,
    ProvenanceGraph,
    ProvenanceTask,
    ProvenanceTimeline,
)
from ..schemas.result_envelope import ResultEnvelope
from .provenance_evidence import build_archive_evidence
from .result_envelope_service import ResultEnvelopeError, read_result_envelope

MAX_GRAPH_NODES = 4096
MAX_ANALYSES = 256
MAX_METADATA_BYTES = 16 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
DIGEST = re.compile(r"[0-9a-f]{64}")
ANALYSIS_ID = re.compile(r"[0-9a-f]{32}")
CURSOR = re.compile(r"[0-9]+-[0-9]+")
URN = "urn:scientific-api:"


class ProvenanceError(ValueError):
    def __init__(self, code: str, status_code: int = 409):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class AnalysisSnapshot:
    archive: dict[str, Any]
    envelope: ResultEnvelope | None


@dataclass(frozen=True)
class ProvenanceSnapshot:
    archive: dict[str, Any]
    envelope: ResultEnvelope | None
    analyses: tuple[AnalysisSnapshot, ...]
    task: ProvenanceTask
    graph: ProvenanceGraph
    timeline: tuple[ProvenanceEvent, ...]


def _bounded(value: Any) -> None:
    if len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode()) > MAX_METADATA_BYTES:
        raise ProvenanceError("PROVENANCE_LIMIT_EXCEEDED", 413)


def _valid_identifier(value: Any, pattern: re.Pattern[str] = IDENTIFIER) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _task_id(task_id: str) -> None:
    if not _valid_identifier(task_id):
        raise ProvenanceError("PROVENANCE_NOT_FOUND", 404)


def _json_literal(value: Any) -> dict[str, Any]:
    return {"@value": value, "@type": "@json"}


def _reference(identifier: str) -> dict[str, str]:
    return {"@id": identifier}


def _content_id(kind: str, value: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return f"{URN}{kind}:{digest}"


def _time(value: float | None) -> str | None:
    try:
        return datetime.fromtimestamp(value, UTC).isoformat() if value is not None else None
    except (OverflowError, OSError, ValueError) as error:
        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error


def _context() -> dict[str, Any]:
    return {
        "@version": 1.1,
        "prov": "http://www.w3.org/ns/prov#",
        "sf": "https://github.com/Artem7898/scientific_fastapi/ns#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "schema_version": "sf:schemaVersion",
        "task_id": "sf:taskId",
        "coverage": "sf:archiveCoverage",
        "storage": "sf:archiveStorage",
        "evidence_status": "sf:evidenceStatus",
        "limitations": "sf:limitation",
    }


class ProvenanceService:
    def __init__(self, backend: TaskStore, artifacts_dir: Path):
        self.backend = backend
        self.artifacts_dir = Path(artifacts_dir)

    def _read_backend(self, method: str, *args, **kwargs):
        try:
            return getattr(self.backend, method)(*args, **kwargs)
        except Exception as error:
            raise ProvenanceError("PROVENANCE_STORAGE_UNAVAILABLE", 503) from error

    def _verify(
        self,
        record: dict[str, Any],
        evidence: dict[str, Any] | None,
        *,
        task_id: str,
        analysis_id: str | None = None,
    ) -> ResultEnvelope | None:
        if evidence is None:
            return None
        if record.get("status") != "completed" or record.get("task_id") != task_id:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        envelope = read_result_envelope(
            record,
            artifacts_dir=self.artifacts_dir,
            analysis_id=analysis_id,
            require_present=True,
        )
        if (
            envelope.task_id != task_id
            or envelope.analysis_id != analysis_id
            or build_archive_evidence(envelope) != evidence
        ):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        return envelope

    @staticmethod
    def _artifacts(
        archive: dict[str, Any], envelope: ResultEnvelope | None
    ) -> list[ProvenanceArtifact]:
        if envelope is None:
            return []
        assert envelope.attempt_id is not None
        return [
            ProvenanceArtifact(
                **record,
                task_id=envelope.task_id,
                attempt_id=envelope.attempt_id,
                analysis_id=envelope.analysis_id,
            )
            for record in archive["evidence"]["artifacts"]
        ]

    def _timeline_page(self, task_id: str, after_id: str | None, limit: int) -> ProvenanceTimeline:
        page = self._read_backend("timeline", task_id, after_id=after_id, limit=limit)
        if page is None:
            raise ProvenanceError("PROVENANCE_NOT_FOUND", 404)
        result = ProvenanceTimeline.model_validate(page)
        if result.task_id != task_id or len(result.events) > limit:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        cursor = tuple(map(int, (after_id or "0-0").split("-")))
        for event in result.events:
            next_cursor = tuple(map(int, event.event_id.split("-")))
            if next_cursor <= cursor:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            cursor = next_cursor
        if result.next_cursor != (result.events[-1].event_id if result.events else None):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        if result.has_more and not result.events:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        return result

    def timeline(
        self, task_id: str, after_id: str | None = None, limit: int = 100
    ) -> ProvenanceTimeline:
        _task_id(task_id)
        if (
            type(limit) is not int
            or not 1 <= limit <= 256
            or (
                after_id is not None
                and (
                    not isinstance(after_id, str)
                    or len(after_id) > 41
                    or not CURSOR.fullmatch(after_id)
                    or any(int(part) > 2**64 - 1 for part in after_id.split("-"))
                )
            )
        ):
            raise ProvenanceError("INVALID_TIMELINE_QUERY", 422)
        try:
            result = self._timeline_page(task_id, after_id, limit)
            _bounded(result.model_dump(mode="json"))
            return result
        except ProvenanceError:
            raise
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error

    def _all_events(self, task_id: str, until_id: str | None) -> tuple[ProvenanceEvent, ...]:
        # The archive and cutoff are captured atomically by the store. Later
        # claims, completion or analysis registrations belong to a new snapshot.
        if until_id is None:
            return ()
        if not isinstance(until_id, str) or not CURSOR.fullmatch(until_id):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        cutoff = tuple(map(int, until_id.split("-")))
        events = []
        cursor = None
        while True:
            page = self._timeline_page(task_id, cursor, 256)
            for event in page.events:
                identity = tuple(map(int, event.event_id.split("-")))
                if identity > cutoff:
                    raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
                events.append(event)
                if len(events) > MAX_GRAPH_NODES:
                    raise ProvenanceError("PROVENANCE_LIMIT_EXCEEDED", 413)
                if identity == cutoff:
                    return tuple(events)
            if not page.has_more:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            cursor = page.next_cursor

    def export_snapshot(self, task_id: str) -> ProvenanceSnapshot:
        """Return one validated archive/evidence/graph snapshot for an export.

        No operational status lookup, filesystem scan or legacy-artifact import
        is performed. A historical record without evidence remains unavailable.
        """
        _task_id(task_id)
        try:
            archive = self._read_backend("get_task_archive", task_id)
            if archive is None:
                raise ProvenanceError("PROVENANCE_NOT_FOUND", 404)
            _bounded(archive)
            archive = copy.deepcopy(archive)
            if archive.get("task_id") != task_id:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            records = archive.get("analyses", [])
            if len(records) > MAX_ANALYSES:
                raise ProvenanceError("PROVENANCE_LIMIT_EXCEEDED", 413)
            record = archive.get("record", {})
            envelope = self._verify(record, archive.get("evidence"), task_id=task_id)
            if envelope is not None and envelope.attempt_id is None:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            analyses = []
            seen = set()
            for item in records:
                analysis_id = item.get("analysis_id")
                if (
                    not _valid_identifier(analysis_id, ANALYSIS_ID)
                    or analysis_id in seen
                    or item.get("task_id") != task_id
                    or not _valid_identifier(item.get("attempt_id"))
                    or item["record"].get("analysis_id") != analysis_id
                    or item["record"].get("attempt_id") != item["attempt_id"]
                ):
                    raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
                seen.add(analysis_id)
                analysis_envelope = self._verify(
                    record, item.get("evidence"), task_id=task_id, analysis_id=analysis_id
                )
                if analysis_envelope is not None and (
                    record.get("status") != "completed"
                    or analysis_envelope.attempt_id != record.get("attempt_id")
                    or analysis_envelope.result.model_dump(mode="json") != item["record"]
                ):
                    raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
                analyses.append(AnalysisSnapshot(item, analysis_envelope))
            events = self._all_events(task_id, archive.get("last_event_id"))
            task = self._task(archive, envelope, tuple(analyses))
            graph = self._graph(task, envelope, tuple(analyses), events)
            _bounded(task.model_dump(mode="json"))
            _bounded(graph.model_dump(mode="json", by_alias=True))
            return ProvenanceSnapshot(archive, envelope, tuple(analyses), task, graph, events)
        except ProvenanceError:
            raise
        except (
            ResultEnvelopeError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            RecursionError,
        ) as error:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error

    def task(self, task_id: str) -> ProvenanceTask:
        return self.export_snapshot(task_id).task

    def graph(self, task_id: str) -> ProvenanceGraph:
        return self.export_snapshot(task_id).graph

    def artifact(self, artifact_id: str) -> ProvenanceArtifact:
        if not _valid_identifier(artifact_id, DIGEST):
            raise ProvenanceError("ARTIFACT_NOT_FOUND", 404)
        locator = self._read_backend("get_artifact_archive", artifact_id)
        if locator is None:
            raise ProvenanceError("ARTIFACT_NOT_FOUND", 404)
        try:
            task = self.task(locator["task_id"])
            records = [*task.artifacts, *(item for a in task.analyses for item in a.artifacts)]
            for record in records:
                if record.artifact_id == artifact_id:
                    expected = {
                        key: value
                        for key, value in record.model_dump(mode="json").items()
                        if key not in {"task_id", "attempt_id", "analysis_id", "evidence_status"}
                    }
                    if (
                        locator.get("artifact_id") != artifact_id
                        or locator.get("attempt_id") != record.attempt_id
                        or locator.get("analysis_id") != record.analysis_id
                        or locator.get("descriptor") != expected
                    ):
                        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
                    return record
            raise ProvenanceError("ARTIFACT_NOT_FOUND", 404)
        except ProvenanceError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error

    def analysis(self, analysis_id: str) -> ProvenanceAnalysis:
        if not _valid_identifier(analysis_id, ANALYSIS_ID):
            raise ProvenanceError("ANALYSIS_NOT_FOUND", 404)
        locator = self._read_backend("get_analysis_archive", analysis_id)
        if locator is None:
            raise ProvenanceError("ANALYSIS_NOT_FOUND", 404)
        try:
            snapshot = self.export_snapshot(locator["task_id"])
            for index, record in enumerate(snapshot.task.analyses):
                if record.analysis_id == analysis_id:
                    if snapshot.analyses[index].archive != locator:
                        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
                    return record
            raise ProvenanceError("ANALYSIS_NOT_FOUND", 404)
        except ProvenanceError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error

    def _task(
        self,
        archive: dict[str, Any],
        envelope: ResultEnvelope | None,
        analyses: tuple[AnalysisSnapshot, ...],
    ) -> ProvenanceTask:
        record = archive.get("record", {})
        payload = archive.get("payload", {})
        version = record.get("model_version") or payload.get("model")
        if (
            version is None
            and payload.get("initial_temperature") is not None
            and "num_particles" in payload
        ):
            version = "ideal-gas-ou-v1"
        analysis_records = []
        for item in analyses:
            data, evidence = item.archive, item.envelope
            analysis_records.append(
                ProvenanceAnalysis(
                    task_id=archive["task_id"],
                    attempt_id=data["attempt_id"],
                    analysis_id=data["analysis_id"],
                    record=data["record"],
                    evidence_status="verified" if evidence else "unavailable",
                    manifest_sha256=evidence.manifest.sha256
                    if evidence and evidence.manifest
                    else None,
                    model_contract_sha256=evidence.model_contract.sha256
                    if evidence and evidence.model_contract
                    else None,
                    source=evidence.source if evidence else None,
                    parameters=evidence.parameters if evidence else {},
                    environment=evidence.environment if evidence else {},
                    artifacts=self._artifacts(data, evidence),
                )
            )
        attempts = [ProvenanceAttempt.model_validate(item) for item in archive.get("attempts", [])]
        if len({item.attempt_id for item in attempts}) != len(attempts):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        if any(not _valid_identifier(item.attempt_id) for item in attempts):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        if envelope and envelope.attempt_id not in {item.attempt_id for item in attempts}:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        limitations = [
            "Unsigned saved metadata and bytes are checked against the committed archive; no author authentication or independent scientific verification is performed",
            "Journal events and attempt claims do not imply checkpoints, deterministic continuation or recorded human identity",
        ]
        if archive["coverage"] == "partial":
            limitations.append(
                "History observed after its beginning is partial; missing historical events are not reconstructed"
            )
        if archive["storage"] == "memory":
            limitations.append(
                "Memory archives survive operational TTL only within this process; restart loses the archive"
            )
        if envelope is None:
            limitations.append(
                "Committed artifact evidence is unavailable; files are not guessed or imported from storage"
            )
        return ProvenanceTask(
            task_id=archive["task_id"],
            coverage=archive["coverage"],
            storage=archive["storage"],
            submitted_at=archive.get("submitted_at"),
            ensemble_id=archive.get("ensemble_id"),
            status=record.get("status", "unknown"),
            model_version=envelope.model_contract.model_version
            if envelope and envelope.model_contract
            else version,
            parameters=envelope.parameters
            if envelope
            else {k: v for k, v in payload.items() if k != "model"},
            record=record,
            evidence_status="verified" if envelope else "unavailable",
            manifest_sha256=envelope.manifest.sha256 if envelope and envelope.manifest else None,
            model_contract_sha256=envelope.model_contract.sha256
            if envelope and envelope.model_contract
            else None,
            execution=envelope.execution if envelope else {},
            rng=envelope.rng if envelope else {},
            environment=envelope.environment if envelope else {},
            diagnostics=envelope.diagnostics if envelope else {},
            units=envelope.units if envelope else {},
            uncertainty=envelope.uncertainty.model_dump(mode="json") if envelope else None,
            attempts=attempts,
            artifacts=self._artifacts(archive, envelope),
            analyses=analysis_records,
            limitations=limitations,
        )

    @staticmethod
    def _graph(
        task: ProvenanceTask,
        envelope: ResultEnvelope | None,
        analyses: tuple[AnalysisSnapshot, ...],
        events: tuple[ProvenanceEvent, ...],
    ) -> ProvenanceGraph:
        nodes: dict[str, dict[str, Any]] = {}

        def add(node: dict[str, Any]) -> str:
            identifier = node["@id"]
            if identifier in nodes and nodes[identifier] != node:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            nodes[identifier] = node
            if len(nodes) > MAX_GRAPH_NODES:
                raise ProvenanceError("PROVENANCE_LIMIT_EXCEEDED", 413)
            return identifier

        def model_contract_node(contract) -> str:
            return add(
                {
                    "@id": f"{URN}model-contract:{contract.sha256}",
                    "@type": "prov:Entity",
                    "sf:modelVersion": contract.model_version,
                    "sf:sha256": contract.sha256,
                    "prov:value": _json_literal(contract.snapshot.model_dump(mode="json")),
                }
            )

        task_identifier = f"{URN}task:{task.task_id}"
        parameter_identifier = add(
            {
                "@id": f"{URN}parameters:{task.task_id}",
                "@type": "prov:Entity",
                "prov:value": _json_literal(task.parameters),
            }
        )
        add(
            {
                "@id": task_identifier,
                "@type": "prov:Entity",
                "sf:status": task.status,
                "sf:parameters": _reference(parameter_identifier),
                "sf:modelVersion": task.model_version,
                "sf:evidenceStatus": task.evidence_status,
            }
        )
        model_identifier = None
        if envelope and envelope.model_contract:
            model_identifier = model_contract_node(envelope.model_contract)
        environment_identifier = backend_identifier = rng_identifier = software_identifier = None
        if envelope:
            for kind, value in (
                ("environment", envelope.environment),
                ("backend", envelope.execution),
                ("rng", envelope.rng),
            ):
                identifier = add(
                    {
                        "@id": _content_id(kind, value),
                        "@type": "prov:Entity",
                        "prov:value": _json_literal(value),
                    }
                )
                if kind == "environment":
                    environment_identifier = identifier
                elif kind == "backend":
                    backend_identifier = identifier
                else:
                    rng_identifier = identifier
            source_hash = envelope.environment.get("source_sha256")
            if isinstance(source_hash, str) and DIGEST.fullmatch(source_hash):
                software_identifier = add(
                    {
                        "@id": f"{URN}software:{source_hash}",
                        "@type": "prov:SoftwareAgent",
                        "sf:sourceSha256": source_hash,
                        "sf:sourceCommit": envelope.environment.get("source_commit"),
                        "sf:packageVersion": envelope.environment.get("package_version"),
                    }
                )
        attempt_identifiers = {}
        for attempt in task.attempts:
            identifier = f"{URN}attempt:{task.task_id}:{attempt.attempt_id}"
            attempt_identifiers[attempt.attempt_id] = identifier
            used = [_reference(parameter_identifier)]
            # Only the committed attempt's environment is attested by its envelope.
            committed = envelope is not None and attempt.attempt_id == envelope.attempt_id
            if committed:
                used.extend(
                    _reference(value)
                    for value in (
                        model_identifier,
                        environment_identifier,
                        backend_identifier,
                        rng_identifier,
                    )
                    if value is not None
                )
            node = {
                "@id": identifier,
                "@type": "prov:Activity",
                "sf:attemptId": attempt.attempt_id,
                "sf:attemptNumber": attempt.attempt,
                "prov:used": used,
                "sf:logicalComputation": _reference(task_identifier),
            }
            if attempt.claimed_at is not None:
                node["prov:startedAtTime"] = {
                    "@value": _time(attempt.claimed_at),
                    "@type": "xsd:dateTime",
                }
            if committed and software_identifier:
                node["prov:wasAssociatedWith"] = _reference(software_identifier)
            add(node)

        def artifact_node(
            record: ProvenanceArtifact, activity: str, derived: list[str] | None = None
        ):
            node = {
                "@id": f"{URN}artifact:{record.artifact_id}",
                "@type": "prov:Entity",
                "sf:name": record.name,
                "sf:path": record.path,
                "sf:sha256": record.sha256,
                "sf:byteSize": record.bytes,
                "sf:mediaType": record.media_type,
                "prov:wasGeneratedBy": _reference(activity),
            }
            if derived:
                node["prov:wasDerivedFrom"] = [_reference(value) for value in derived]
            add(node)

        for artifact in task.artifacts:
            artifact_node(artifact, attempt_identifiers[artifact.attempt_id])
        if envelope is not None and envelope.attempt_id is not None:
            add(
                {
                    "@id": f"{URN}result:{task.task_id}:{envelope.attempt_id}",
                    "@type": "prov:Entity",
                    "prov:value": _json_literal(envelope.result.model_dump(mode="json")),
                    "prov:wasGeneratedBy": _reference(attempt_identifiers[envelope.attempt_id]),
                    "prov:wasDerivedFrom": [
                        _reference(f"{URN}artifact:{item.artifact_id}") for item in task.artifacts
                    ],
                }
            )
        source_artifacts = [
            f"{URN}artifact:{item.artifact_id}"
            for item in task.artifacts
            if item.name in {"manifest", "final_state"}
        ]
        for item, summary in zip(analyses, task.analyses, strict=True):
            identifier = f"{URN}analysis:{summary.analysis_id}"
            node = {
                "@id": identifier,
                "@type": "prov:Activity",
                "sf:analysisId": summary.analysis_id,
                "sf:evidenceStatus": summary.evidence_status,
                "sf:attempt": _reference(attempt_identifiers[summary.attempt_id])
                if summary.attempt_id in attempt_identifiers
                else None,
            }
            if item.envelope:
                if item.envelope.model_contract:
                    node["sf:modelContract"] = _reference(
                        model_contract_node(item.envelope.model_contract)
                    )
                analysis_sources = list(source_artifacts)
                if not analysis_sources and item.envelope.source:
                    # The new analysis may have a strong archive anchor while
                    # its pre-journal parent has only partial history. Preserve
                    # explicit source references, without inventing indexed
                    # parent artifact occurrences or historical verification.
                    for name in ("state_sha256", "manifest_sha256"):
                        digest = item.envelope.source.get(name)
                        if isinstance(digest, str) and DIGEST.fullmatch(digest):
                            source = add(
                                {
                                    "@id": f"{URN}analysis-source:{summary.analysis_id}:{name}",
                                    "@type": "prov:Entity",
                                    "sf:sha256": digest,
                                    "sf:sourceKind": name,
                                    "sf:evidenceStatus": "referenced_by_verified_analysis",
                                    "sf:historicalParentAnchor": "unavailable",
                                }
                            )
                            analysis_sources.append(source)
                node["prov:used"] = [_reference(value) for value in analysis_sources]
                node["sf:parameters"] = _json_literal(item.envelope.parameters)
                analysis_hash = item.envelope.environment.get("analysis_source_sha256")
                if isinstance(analysis_hash, str) and DIGEST.fullmatch(analysis_hash):
                    software = add(
                        {
                            "@id": f"{URN}analysis-software:{analysis_hash}",
                            "@type": "prov:SoftwareAgent",
                            "sf:sourceSha256": analysis_hash,
                        }
                    )
                    node["prov:wasAssociatedWith"] = _reference(software)
                environment = add(
                    {
                        "@id": _content_id("analysis-environment", item.envelope.environment),
                        "@type": "prov:Entity",
                        "prov:value": _json_literal(item.envelope.environment),
                    }
                )
                node["prov:used"].append(_reference(environment))
            add(node)
            for artifact in summary.artifacts:
                artifact_node(
                    artifact,
                    identifier,
                    analysis_sources if item.envelope else source_artifacts,
                )
        for event in events:
            node = {
                "@id": f"{URN}event:{task.task_id}:{event.event_id}",
                "@type": "prov:Entity",
                "sf:eventType": event.event_type,
                "sf:eventId": event.event_id,
                "sf:status": event.status,
                "sf:details": _json_literal(event.details),
            }
            if event.occurred_at is not None:
                node["prov:generatedAtTime"] = {
                    "@value": _time(event.occurred_at),
                    "@type": "xsd:dateTime",
                }
            if event.attempt_id in attempt_identifiers:
                node["sf:attempt"] = _reference(attempt_identifiers[event.attempt_id])
            add(node)
        return ProvenanceGraph.model_validate(
            {
                "task_id": task.task_id,
                "coverage": task.coverage,
                "storage": task.storage,
                "evidence_status": task.evidence_status,
                "@context": _context(),
                "@graph": list(nodes.values()),
                "limitations": task.limitations,
            }
        )
