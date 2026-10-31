"""Independent verification records, separate from immutable scientific results."""

from typing import Any, Literal

from pydantic import Field, StrictBool, field_validator

from .simulation import StrictModel

VerificationVerdict = Literal["passed", "failed", "inconclusive", "unsupported"]
VerificationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class VerificationRequest(StrictModel):
    rerun: StrictBool = False


class VerificationCheck(StrictModel):
    name: str
    status: VerificationVerdict
    reason: str
    expected: Any = None
    observed: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationSource(StrictModel):
    task_id: str
    attempt_id: str
    model_version: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerificationCertificate(StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["result-verification-v1"] = "result-verification-v1"
    verification_id: str
    verifier_attempt_id: str
    source: VerificationSource
    options: VerificationRequest
    started_at: str
    completed_at: str
    verdict: VerificationVerdict
    checks: list[VerificationCheck]
    verifier: dict[str, Any]
    authenticity: Literal["unsigned"] = "unsigned"
    scope: Literal["recorded_artifacts_at_verification_time"] = (
        "recorded_artifacts_at_verification_time"
    )
    limitations: list[str]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class VerificationAccepted(StrictModel):
    verification_id: str
    status: Literal["pending"] = "pending"
    source_task_id: str
    source_attempt_id: str
    check_url: str


class VerificationRecord(StrictModel):
    verification_id: str
    status: VerificationStatus
    source_task_id: str
    source_attempt_id: str
    submitted_at: float | None
    attempts: int
    verdict: VerificationVerdict | None = None
    certificate_sha256: str | None = None
    error_code: str | None = None
    error: str | None = None


class VerificationList(StrictModel):
    task_id: str
    verifications: list[VerificationRecord]
