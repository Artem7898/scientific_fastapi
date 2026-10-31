"""Bounded generic RO-Crate 1.1 exports of an archived scientific snapshot.

The package includes the exact manifest/scientific bytes anchored by the
archive. Envelopes are reserialized from verified objects, so packaging is not
an authorship attestation or a claim of bitwise scientific reproducibility.
"""

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas.result_envelope import ResultEnvelope
from .provenance_service import ProvenanceError, ProvenanceService
from .result_envelope_service import _json, _open, _root, _snapshot

MAX_EXPORT_BYTES = 128 * 1024 * 1024
MAX_EXPORT_FILES = 128
MAX_METADATA_BYTES = 2 * 1024 * 1024
RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.1"
RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
PROJECT_URL = "https://github.com/Artem7898/scientific_fastapi"
SCI_NAMESPACE = PROJECT_URL + "/ns/provenance#"
_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class _Entry:
    path: str
    media_type: str
    sha256: str
    size: int
    data: bytes | None = None
    artifact_id: str | None = None
    activity_id: str | None = None


def _json_bytes(value: Any, *, compact: bool = False) -> bytes:
    if compact:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _safe_path(path: str) -> str:
    """Restrict even verified metadata to portable, relative ZIP member names."""
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or ":" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(
            part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in _DEVICE_NAMES
            for part in path.split("/")
        )
        or any(ord(char) < 32 for char in path)
    ):
        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
    return path


def _generated(
    path: str, media_type: str, data: bytes, *, activity_id: str | None = None
) -> _Entry:
    if len(data) > MAX_METADATA_BYTES:
        raise ProvenanceError("EXPORT_LIMIT_EXCEEDED", 413)
    return _Entry(
        _safe_path(path),
        media_type,
        hashlib.sha256(data).hexdigest(),
        len(data),
        data,
        activity_id=activity_id,
    )


def _add_envelope(entries: dict[str, _Entry], envelope: ResultEnvelope, activity: str) -> None:
    if envelope.integrity.status != "verified" or envelope.manifest is None:
        raise ProvenanceError("EXPORT_RESULT_UNAVAILABLE", 409)
    for record in envelope.artifacts:
        entry = _Entry(
            path=_safe_path(record.path),
            media_type=record.media_type,
            sha256=record.sha256,
            size=record.bytes,
            activity_id=activity,
        )
        previous = entries.get(entry.path)
        if previous is not None and previous != entry:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        entries[entry.path] = entry
    envelope_path = str(Path(envelope.manifest.path).with_name("envelope.json")).replace("\\", "/")
    entry = _generated(
        envelope_path,
        "application/json",
        _json_bytes(envelope.model_dump(mode="json")),
        activity_id=activity,
    )
    if entry.path in entries:
        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
    entries[entry.path] = entry


class _BoundedBuffer(io.BytesIO):
    def write(self, data: Any, /) -> int:
        if self.tell() + len(data) > MAX_EXPORT_BYTES:
            raise ProvenanceError("EXPORT_LIMIT_EXCEEDED", 413)
        return super().write(data)


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_path(path), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _copy_file(root: Path, entry: _Entry, output: Any) -> None:
    """Hash the same opened bytes written to the archive, never reopen by path."""
    descriptor = _open(root, entry.path)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(min(1024 * 1024, entry.size + 1 - size)):
            size += len(chunk)
            if size > entry.size:
                raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
            digest.update(chunk)
            output.write(chunk)
        after = os.fstat(stream.fileno())
    if (
        size != entry.size
        or digest.hexdigest() != entry.sha256
        or (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
        != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
    ):
        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")


def _metadata(
    task_id: str,
    envelope: ResultEnvelope,
    entries: dict[str, _Entry],
    activities: list[dict[str, Any]],
    publication_time: str,
) -> dict[str, Any]:
    files = []
    for path, entry in sorted(entries.items()):
        item: dict[str, Any] = {
            "@id": path,
            "@type": "File",
            "name": path.rsplit("/", 1)[-1],
            "encodingFormat": entry.media_type,
            "contentSize": str(entry.size),
            "sci:sha256": entry.sha256,
        }
        if entry.artifact_id is not None:
            item["sameAs"] = {"@id": f"urn:scientific-api:artifact:{entry.artifact_id}"}
        if entry.activity_id is not None:
            item["prov:wasGeneratedBy"] = {"@id": entry.activity_id}
        files.append(item)
    model_version = envelope.result.model_dump().get("model_version")
    software: dict[str, Any] = {
        "@id": "#scientific-api",
        "@type": ["SoftwareApplication", "prov:SoftwareAgent"],
        "name": "Scientific API",
        "url": PROJECT_URL,
        # This license describes code only; no license for research data is inferred.
        "license": "https://spdx.org/licenses/MIT.html",
    }
    if isinstance(envelope.environment.get("package_version"), str):
        software["softwareVersion"] = envelope.environment["package_version"]
    model: dict[str, Any] = {
        "@id": "#scientific-model",
        "@type": "CreativeWork",
        "name": model_version,
        "identifier": model_version,
        "subjectOf": {"@id": "model-contract.json"},
    }
    if envelope.model_contract is not None:
        model["sci:contractSha256"] = envelope.model_contract.sha256
    return {
        "@context": [
            RO_CRATE_CONTEXT,
            {"prov": "http://www.w3.org/ns/prov#", "sci": SCI_NAMESPACE},
        ],
        "@graph": [
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "conformsTo": {"@id": RO_CRATE_PROFILE},
                "about": {"@id": "./"},
            },
            {
                "@id": "./",
                "@type": "Dataset",
                "name": f"Scientific API result {task_id}",
                "description": (
                    "Archived scientific result and recorded derivations. File integrity is checked "
                    "against archived unsigned metadata; authorship, independent physics validation "
                    "and redistribution rights are not asserted. datePublished records the original "
                    "result manifest creation, not the time of this package reconstruction."
                ),
                "datePublished": publication_time,
                "license": {"@id": "#data-license-unavailable"},
                "hasPart": [{"@id": path} for path in sorted(entries)],
                "mentions": [{"@id": "#scientific-model"}, {"@id": "#scientific-api"}],
                "subjectOf": {"@id": "provenance.jsonld"},
            },
            software,
            model,
            {
                "@id": "#data-license-unavailable",
                "@type": "CreativeWork",
                "name": "Research-data license not supplied",
                "description": "No research-data license or redistribution permission is inferred from the source-code license.",
            },
            *activities,
            *files,
        ],
    }


def export_ro_crate(service: ProvenanceService, task_id: str) -> tuple[bytes, str]:
    """Export one coherent archived snapshot as a generic RO-Crate 1.1 ZIP.

    Limits apply to both total uncompressed bytes and the returned ZIP. No
    network context resolution, numerical execution or optional dependency is
    involved. Unknown historical evidence cannot be exported as verified data.
    """
    if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
        raise ProvenanceError("PROVENANCE_NOT_FOUND", 404)
    try:
        snapshot = service.export_snapshot(task_id)
        envelope = snapshot.envelope
        if envelope is None or envelope.integrity.status != "verified":
            raise ProvenanceError("EXPORT_RESULT_UNAVAILABLE", 409)
        if envelope.result.status != "completed" or envelope.model_contract is None:
            raise ProvenanceError("EXPORT_RESULT_UNAVAILABLE", 409)
        entries: dict[str, _Entry] = {}
        run_activity = "#simulation"
        _add_envelope(entries, envelope, run_activity)
        activities: list[dict[str, Any]] = [
            {
                "@id": run_activity,
                "@type": ["CreateAction", "prov:Activity"],
                "name": "Recorded scientific simulation",
                "actionStatus": {"@id": "http://schema.org/CompletedActionStatus"},
                "instrument": [{"@id": "#scientific-model"}, {"@id": "#scientific-api"}],
                "prov:wasAssociatedWith": {"@id": "#scientific-api"},
            }
        ]
        for analysis in snapshot.analyses:
            if analysis.envelope is None:
                raise ProvenanceError("EXPORT_RESULT_UNAVAILABLE", 409)
            analysis_envelope = analysis.envelope
            activity = f"#analysis-{analysis_envelope.analysis_id}"
            _add_envelope(entries, analysis_envelope, activity)
            activities.append(
                {
                    "@id": activity,
                    "@type": ["CreateAction", "prov:Activity"],
                    "name": "Recorded saved-state analysis",
                    "instrument": {"@id": "#scientific-api"},
                    "prov:wasInformedBy": {"@id": run_activity},
                    "prov:wasAssociatedWith": {"@id": "#scientific-api"},
                }
            )
        for artifact in snapshot.task.artifacts:
            entries[artifact.path] = replace(
                entries[artifact.path], artifact_id=artifact.artifact_id
            )
        for analysis in snapshot.task.analyses:
            for artifact in analysis.artifacts:
                entries[artifact.path] = replace(
                    entries[artifact.path], artifact_id=artifact.artifact_id
                )
        generated = (
            _generated(
                "provenance.jsonld",
                "application/ld+json",
                _json_bytes(snapshot.graph.model_dump(mode="json", by_alias=True)),
            ),
            _generated(
                "task.json", "application/json", _json_bytes(snapshot.task.model_dump(mode="json"))
            ),
            _generated(
                "timeline.json",
                "application/json",
                _json_bytes(
                    {
                        "task_id": task_id,
                        "events": [event.model_dump(mode="json") for event in snapshot.timeline],
                        "coverage": snapshot.task.coverage,
                    }
                ),
            ),
            _generated(
                "model-contract.json",
                "application/json",
                _json_bytes(envelope.model_contract.snapshot.model_dump(mode="json"), compact=True),
            ),
        )
        entries.update((entry.path, entry) for entry in generated)
        root = _root(service.artifacts_dir)
        if envelope.manifest is None:
            raise ProvenanceError("EXPORT_RESULT_UNAVAILABLE")
        manifest_digest, _, manifest_bytes = _snapshot(
            root, envelope.manifest.path, MAX_METADATA_BYTES, retain=True
        )
        if manifest_bytes is None or manifest_digest != envelope.manifest.sha256:
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        publication_time = _json(manifest_bytes).get("created_at")
        if (
            not isinstance(publication_time, str)
            or datetime.fromisoformat(publication_time).utcoffset() is None
        ):
            raise ProvenanceError("EXPORT_METADATA_UNAVAILABLE")
        metadata = _generated(
            "ro-crate-metadata.json",
            "application/ld+json",
            _json_bytes(_metadata(task_id, envelope, entries, activities, publication_time)),
        )
        entries[metadata.path] = metadata
        if (
            len(entries) > MAX_EXPORT_FILES
            or sum(entry.size for entry in entries.values()) > MAX_EXPORT_BYTES
        ):
            raise ProvenanceError("EXPORT_LIMIT_EXCEEDED", 413)
        if any(entry.size < 0 or not _DIGEST.fullmatch(entry.sha256) for entry in entries.values()):
            raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED")
        output = _BoundedBuffer()
        with zipfile.ZipFile(
            output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path, entry in sorted(entries.items()):
                with archive.open(_zip_info(path), mode="w") as member:
                    if entry.data is not None:
                        member.write(entry.data)
                    else:
                        _copy_file(root, entry, member)
        return output.getvalue(), f"scientific-api-{task_id}-ro-crate.zip"
    except ProvenanceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise ProvenanceError("PROVENANCE_INTEGRITY_FAILED") from error
