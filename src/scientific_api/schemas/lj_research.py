"""Explicit choices and immutable reports for conditional LJ uncertainty."""

from typing import Annotated, Any, Literal

from pydantic import Field, FiniteFloat, StrictBool, StrictInt, field_validator, model_validator

from .simulation import StrictModel
from .verification import VerificationSource

ResearchKind = Literal["timeseries", "replicas", "sensitivity"]
ResearchStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
ResearchAvailability = Literal["available", "partial", "unavailable"]
LJObservable = Literal[
    "thermal_temperature",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "pressure",
    "virial",
]
SourceIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
OBSERVABLES = (
    "thermal_temperature",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "pressure",
    "virial",
)


class LJResearchRequest(StrictModel):
    kind: ResearchKind = "timeseries"
    task_ids: list[SourceIdentifier] = Field(min_length=1, max_length=50)
    observables: list[LJObservable] = Field(
        default_factory=lambda: list(OBSERVABLES), min_length=1, max_length=6
    )
    burn_in_time: FiniteFloat | None = Field(default=None, ge=0, le=1e10)
    bootstrap_seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)
    bootstrap_samples: StrictInt = Field(default=1000, ge=200, le=5000)
    confidence_level: FiniteFloat = Field(
        default=0.95, json_schema_extra={"enum": [0.9, 0.95, 0.99]}
    )
    block_length: StrictInt | None = Field(default=None, ge=1, le=1001)
    assume_stationary: StrictBool = False
    varied_parameter: Literal["time_step", "cutoff_radius"] | None = None

    @field_validator("confidence_level")
    @classmethod
    def supported_confidence(cls, value: float) -> float:
        if value not in {0.9, 0.95, 0.99}:
            raise ValueError("confidence_level must be 0.9, 0.95 or 0.99")
        return value

    @model_validator(mode="after")
    def validate_design(self) -> "LJResearchRequest":
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must identify distinct logical runs")
        if len(set(self.observables)) != len(self.observables):
            raise ValueError("observables must be unique")
        if self.kind == "timeseries" and len(self.task_ids) != 1:
            raise ValueError("timeseries requires exactly one source run")
        if self.kind == "replicas" and len(self.task_ids) < 2:
            raise ValueError("replicas requires at least two source runs")
        if self.kind == "sensitivity":
            if len(self.task_ids) < 4 or self.varied_parameter is None:
                raise ValueError("sensitivity requires at least four runs and varied_parameter")
        elif self.varied_parameter is not None:
            raise ValueError("varied_parameter belongs to sensitivity analyses")
        if self.kind != "timeseries" and (self.block_length is not None or self.assume_stationary):
            raise ValueError("block_length and assume_stationary belong to timeseries analyses")
        return self


class LJResearchAccepted(StrictModel):
    analysis_id: str
    status: Literal["pending"] = "pending"
    kind: ResearchKind
    source_task_ids: list[str]
    check_url: str


class LJResearchRecord(StrictModel):
    analysis_id: str
    status: ResearchStatus
    kind: ResearchKind
    source_task_ids: list[str]
    submitted_at: FiniteFloat | None
    attempts: StrictInt = Field(ge=0)
    availability: ResearchAvailability | None = None
    report_sha256: str | None = None
    error_code: str | None = None
    error: str | None = None


class LJResearchSource(StrictModel):
    identity: VerificationSource
    parameters: dict[str, Any]


class LJResearchReport(StrictModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["lj-uncertainty-report-v1"] = "lj-uncertainty-report-v1"
    analysis_id: str
    analysis_attempt_id: str
    kind: ResearchKind
    model_version: Literal["lennard-jones-2d-v1"] = "lennard-jones-2d-v1"
    options: LJResearchRequest
    sources: list[LJResearchSource]
    started_at: str
    completed_at: str
    availability: ResearchAvailability
    statistics: dict[str, Any]
    implementation: dict[str, Any]
    source_integrity: Literal["verified_against_committed_archive"] = (
        "verified_against_committed_archive"
    )
    authenticity: Literal["unsigned"] = "unsigned"
    limitations: list[str]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class LJResearchList(StrictModel):
    task_id: str
    analyses: list[LJResearchRecord]
