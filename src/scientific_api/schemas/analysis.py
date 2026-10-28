"""Strict contracts for analysis of an already committed scientific run."""

from typing import Literal

from pydantic import Field, StrictInt, model_validator

from .simulation import StrictModel

AnalysisArtifact = Literal["analysis", "manifest", "png", "pdf"]


class VelocityAnalysisRequest(StrictModel):
    bins: StrictInt = Field(default=32, ge=2, le=256)
    formats: list[Literal["png", "pdf"]] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def unique_formats(self) -> "VelocityAnalysisRequest":
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must not contain duplicates")
        return self


class VelocityAnalysisResponse(StrictModel):
    analysis_id: str
    task_id: str
    attempt_id: str
    status: Literal["completed"] = "completed"
    analysis_version: str
    artifacts: dict[AnalysisArtifact, str]
