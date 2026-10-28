"""Strict inputs and explicit statistical outputs for independent replicas."""

from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, model_validator

from .simulation import StrictModel


class EnsembleParameters(StrictModel):
    """One physical experiment; seeds belong to the ensemble, not this input."""

    initial_temperature: FiniteFloat = Field(gt=0, le=1e6)
    num_particles: StrictInt = Field(gt=0, le=1_000_000)
    time_step: FiniteFloat = Field(default=0.001, gt=0, le=1000)
    num_steps: StrictInt = Field(default=100, gt=0, le=10_000)


class EnsembleRequest(StrictModel):
    model: Literal["ideal-gas-ou-v1"] = "ideal-gas-ou-v1"
    parameters: EnsembleParameters
    replicas: StrictInt = Field(default=50, ge=1, le=50)
    base_seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)
    failure_policy: Literal["require_all"] = "require_all"

    @model_validator(mode="after")
    def bound_work(self) -> "EnsembleRequest":
        work = self.replicas * self.parameters.num_particles * self.parameters.num_steps
        if work > 100_000_000:
            raise ValueError("replicas * num_particles * num_steps must not exceed 100000000")
        return self


class EnsembleAccepted(StrictModel):
    ensemble_id: str
    status: Literal["pending"] = "pending"
    check_url: str


class EnsembleStatistic(StrictModel):
    n: StrictInt = Field(ge=1)
    mean: FiniteFloat
    sample_standard_deviation: FiniteFloat | None = Field(ge=0)
    standard_error: FiniteFloat | None = Field(ge=0)


class EnsembleCounts(StrictModel):
    requested: StrictInt = Field(ge=1)
    pending: StrictInt = Field(ge=0)
    running: StrictInt = Field(ge=0)
    completed: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)
    cancelled: StrictInt = Field(ge=0)


class EnsembleChild(StrictModel):
    replica_index: StrictInt = Field(ge=0)
    task_id: str
    seed: StrictInt = Field(ge=0, le=2**64 - 1)
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    result: dict[str, Any] | None


class EnsembleStatus(StrictModel):
    ensemble_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    cancel_requested: bool
    model: Literal["ideal-gas-ou-v1"]
    failure_policy: Literal["require_all"]
    counts: EnsembleCounts
    children: list[EnsembleChild]
    statistics: dict[Literal["final_temperature", "total_energy"], EnsembleStatistic] | None
    theoretical_temperature_standard_error: FiniteFloat | None = Field(ge=0)


class EnsembleCancellation(StrictModel):
    ensemble_id: str
    status: Literal["cancel_requested"] = "cancel_requested"
