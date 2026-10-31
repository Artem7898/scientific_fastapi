"""Archived lineage and PROV-O JSON-LD, without identity or validity claims."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

ArchiveCoverage = Literal["complete", "partial"]
ArchiveStorage = Literal["memory", "redis"]
EvidenceStatus = Literal["verified", "unavailable"]


class _ProvenanceModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False, populate_by_name=True
    )


class ProvenanceArtifact(_ProvenanceModel):
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str
    attempt_id: str
    analysis_id: str | None = None
    name: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: StrictInt = Field(ge=0)
    media_type: str
    evidence_status: Literal["verified"] = "verified"


class ProvenanceAttempt(_ProvenanceModel):
    attempt_id: str
    attempt: StrictInt | None = Field(default=None, ge=1)
    claimed_at: FiniteFloat | None = None


class ProvenanceAnalysis(_ProvenanceModel):
    task_id: str
    attempt_id: str
    analysis_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    record: dict[str, Any]
    evidence_status: EvidenceStatus
    manifest_sha256: str | None = None
    model_contract_sha256: str | None = None
    source: dict[str, Any] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ProvenanceArtifact] = Field(default_factory=list)


class ProvenanceTask(_ProvenanceModel):
    schema_version: Literal[1] = 1
    task_id: str
    coverage: ArchiveCoverage
    storage: ArchiveStorage
    submitted_at: FiniteFloat | None
    ensemble_id: str | None = None
    status: Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]
    model_version: str | None
    parameters: dict[str, Any]
    record: dict[str, Any]
    evidence_status: EvidenceStatus
    manifest_sha256: str | None = None
    model_contract_sha256: str | None = None
    execution: dict[str, Any] = Field(default_factory=dict)
    rng: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    uncertainty: dict[str, Any] | None = None
    attempts: list[ProvenanceAttempt] = Field(default_factory=list)
    artifacts: list[ProvenanceArtifact] = Field(default_factory=list)
    analyses: list[ProvenanceAnalysis] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ProvenanceEvent(_ProvenanceModel):
    event_id: str = Field(pattern=r"^[0-9]+-[0-9]+$")
    event_type: str
    occurred_at: FiniteFloat | None
    attempt_id: str | None = None
    attempt: StrictInt | None = Field(default=None, ge=0)
    status: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceTimeline(_ProvenanceModel):
    task_id: str
    events: list[ProvenanceEvent] = Field(max_length=256)
    next_cursor: str | None
    has_more: bool


class ProvenanceGraph(_ProvenanceModel):
    schema_version: Literal[1] = 1
    task_id: str
    coverage: ArchiveCoverage
    storage: ArchiveStorage
    evidence_status: EvidenceStatus
    context: dict[str, Any] = Field(alias="@context")
    nodes: list[dict[str, Any]] = Field(alias="@graph", max_length=4096)
    limitations: list[str] = Field(default_factory=list)
