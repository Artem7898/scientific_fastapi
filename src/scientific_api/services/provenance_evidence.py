"""Bind committed queue history to verified scientific artifact occurrences."""

import hashlib
import json
from typing import Any

from ..schemas.result_envelope import ResultEnvelope


def artifact_occurrence_id(
    task_id: str, attempt_id: str | None, analysis_id: str | None, name: str
) -> str:
    """Identify an occurrence, separately from its file-content SHA256.

    Identical bytes in two experiments retain different provenance identities.
    The versioned JSON encoding avoids delimiter ambiguities in identifiers.
    """
    identity = {
        "schema_version": 1,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "analysis_id": analysis_id,
        "name": name,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_archive_evidence(envelope: ResultEnvelope) -> dict[str, Any]:
    """Return small JSON-safe evidence after verification, before fenced commit.

    The store must index this only if the actual committed state is completed.
    This function neither commits a task nor authenticates the artifact author.
    """
    if (
        envelope.integrity.status != "verified"
        or envelope.manifest is None
        or envelope.model_contract is None
    ):
        raise ValueError("Archival evidence requires a verified result envelope")
    descriptors = []
    for record in envelope.artifacts:
        descriptors.append(
            {
                "artifact_id": artifact_occurrence_id(
                    envelope.task_id, envelope.attempt_id, envelope.analysis_id, record.name
                ),
                **record.model_dump(mode="json"),
            }
        )
    manifest = next(item for item in descriptors if item["name"] == "manifest")
    if manifest["sha256"] != envelope.manifest.sha256:
        raise ValueError("The envelope manifest must match its artifact descriptor")
    return {
        "schema_version": 1,
        "model_version": envelope.model_contract.model_version,
        "model_contract_sha256": envelope.model_contract.sha256,
        "manifest_sha256": envelope.manifest.sha256,
        "manifest": manifest,
        "artifacts": descriptors,
    }
