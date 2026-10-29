"""Strict, finite transport contracts for simulations in reduced units."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator

from ..domain.simulation import ComputeBackend, SimulationParameters


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class SimulationRequest(StrictModel):
    initial_temperature: FiniteFloat = Field(
        gt=0.0,
        le=1e6,
        description="Temperature in reduced units (m = k_B = 1)",
        examples=[300.0],
    )
    num_particles: StrictInt = Field(gt=0, le=1_000_000)
    time_step: FiniteFloat = Field(
        default=1e-3, gt=0.0, le=1000.0, description="Reduced time; thermostat rate is 1"
    )
    num_steps: StrictInt = Field(default=100, gt=0, le=10_000)
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)
    backend: ComputeBackend = "numpy"

    @model_validator(mode="after")
    def bound_work(self) -> "SimulationRequest":
        if self.num_particles * self.num_steps > 100_000_000:
            raise ValueError("num_particles * num_steps must not exceed 100000000")
        return self

    def to_parameters(self) -> SimulationParameters:
        return SimulationParameters(**self.model_dump())


class TaskAcceptedResponse(StrictModel):
    task_id: str
    status: Literal["pending"] = "pending"
    check_url: str


class TaskInProgress(StrictModel):
    task_id: str
    status: Literal["pending", "running"]


class SimulationResponse(StrictModel):
    task_id: str
    attempt_id: str
    status: Literal["completed"]
    final_temperature: FiniteFloat = Field(ge=0)
    total_energy: FiniteFloat = Field(ge=0)
    temperature_standard_error: FiniteFloat = Field(ge=0)
    model_version: str
    seed: StrictInt
    num_steps: StrictInt
    units: dict[str, str]
    artifacts: dict[Literal["manifest", "final_state"], str]


class TaskFailed(StrictModel):
    task_id: str
    status: Literal["failed", "cancelled"]
    error: str


TaskStatusResponse = Annotated[
    TaskInProgress | SimulationResponse | TaskFailed, Field(discriminator="status")
]
