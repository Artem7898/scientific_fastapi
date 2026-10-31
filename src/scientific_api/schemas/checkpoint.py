"""Strict public references to immutable, registered computation checkpoints."""

from typing import Any, Literal

from pydantic import Field, StrictInt, field_validator

from .simulation import StrictModel


class CheckpointFile(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: StrictInt = Field(ge=0)


class CheckpointDescriptor(StrictModel):
    schema_version: Literal[1] = 1
    checkpoint_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    completed_steps: StrictInt = Field(ge=0, le=10_000)
    total_steps: StrictInt = Field(gt=0, le=10_000)
    model_version: Literal["ideal-gas-ou-v1", "langevin-free-v1", "lennard-jones-2d-v1"]
    parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: CheckpointFile
    state: CheckpointFile
    created_at: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class CheckpointList(StrictModel):
    task_id: str
    checkpoints: list[CheckpointDescriptor] = Field(max_length=32)
    integrity_status: Literal["recorded_not_rechecked"] = "recorded_not_rechecked"


class CheckpointDetails(StrictModel):
    descriptor: CheckpointDescriptor
    metadata: dict[str, Any]
    integrity_status: Literal["verified"] = "verified"
    fingerprint_status: Literal["not_checked"] = "not_checked"
    authenticity: Literal["unsigned"] = "unsigned"


class CheckpointResumeRequest(StrictModel):
    checkpoint_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class CheckpointResumeAccepted(StrictModel):
    task_id: str
    status: Literal["pending", "running"]
    checkpoint_id: str
    deduplicated: bool
    check_url: str
