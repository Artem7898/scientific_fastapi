"""Additive evidence records; integrity is relative to saved, unsigned metadata."""

import json
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, field_validator

from .analysis import VelocityAnalysisResponse
from .langevin import LangevinResponse
from .lennard_jones import LennardJonesResponse
from .model_contract import ModelContract
from .simulation import SimulationResponse, StrictModel


class RecordedSimulationResponse(SimulationResponse):
    # Standalone Python/CLI runs historically have no queue attempt identifier.
    # The existing HTTP response deliberately retains its required string.
    attempt_id: str | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


class RecordedLangevinResponse(LangevinResponse):
    attempt_id: str | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


class RecordedLennardJonesResponse(LennardJonesResponse):
    attempt_id: str | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


RecordedSimulation = Annotated[
    RecordedSimulationResponse | RecordedLangevinResponse | RecordedLennardJonesResponse,
    Field(discriminator="model_version"),
]


class EnvelopeIntegrity(StrictModel):
    status: Literal["verified", "not_verified"]
    reason: str | None = None
    scope: Literal["saved_metadata_and_bytes"] = "saved_metadata_and_bytes"


class EnvelopeArtifact(StrictModel):
    name: str
    path: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: StrictInt = Field(ge=0)


class EnvelopeModelContract(StrictModel):
    model_version: str
    schema_version: StrictInt = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: ModelContract

    @field_validator("snapshot", mode="before")
    @classmethod
    def json_contract_snapshot(cls, value: Any) -> Any:
        # Strict contract tuples represent JSON arrays; accept their documented
        # JSON representation without weakening the contract's scalar types.
        if isinstance(value, dict):
            return ModelContract.model_validate_json(json.dumps(value, allow_nan=False))
        return value


class EnvelopeUncertainty(StrictModel):
    availability: Literal["available", "partial", "unavailable"]
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class ResultEnvelope(StrictModel):
    """No signature, independent rerun or scientific-validity verdict is implied."""

    schema_version: Literal[1] = 1
    kind: Literal["simulation", "velocity_analysis"]
    task_id: str
    attempt_id: str | None
    analysis_id: str | None = None
    result: RecordedSimulation | VelocityAnalysisResponse
    model_contract: EnvelopeModelContract | None = None
    integrity: EnvelopeIntegrity
    authenticity: Literal["unsigned"] = "unsigned"
    scientific_verification: Literal["not_performed"] = "not_performed"
    manifest: EnvelopeArtifact | None = None
    artifacts: list[EnvelopeArtifact] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    rng: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    uncertainty: EnvelopeUncertainty
    source: dict[str, Any] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: Any) -> Any:
        # Python's bool is an int subclass, and Literal[1] alone accepts True.
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value
