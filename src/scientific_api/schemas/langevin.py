"""Strict transport contracts for joint position/velocity Langevin dynamics."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from ..domain.langevin import LangevinParameters
from ..domain.simulation import ComputeBackend


class _StrictLangevinModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class LangevinEnsembleParameters(_StrictLangevinModel):
    """Physical inputs shared by independent replicas; seeds are assigned centrally."""

    initial_temperature: FiniteFloat = Field(ge=1e-6, le=1e6)
    bath_temperature: FiniteFloat = Field(ge=1e-6, le=1e6)
    num_particles: StrictInt = Field(ge=3, le=1_000_000)
    mass: FiniteFloat = Field(default=1.0, ge=1e-3, le=1e3)
    friction: FiniteFloat = Field(default=1.0, ge=1e-3, le=1e3)
    time_step: FiniteFloat = Field(default=0.01, ge=1e-6, le=10.0)
    num_steps: StrictInt = Field(default=1000, ge=1, le=10_000)
    backend: ComputeBackend = "numpy"
    initial_mean_velocity: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    initial_position: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    acceleration: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    boundary: Literal["unbounded", "periodic"] = "unbounded"
    box_size: tuple[FiniteFloat, FiniteFloat] | None = None
    sample_stride: StrictInt = Field(default=10, ge=1, le=10_000)
    diffusion_fit_start_step: StrictInt | None = Field(default=None, ge=0, le=10_000)
    diffusion_fit_end_step: StrictInt | None = Field(default=None, ge=0, le=10_000)

    @field_validator(
        "initial_mean_velocity", "initial_position", "acceleration", "box_size", mode="before"
    )
    @classmethod
    def normalize_json_pairs(cls, value: Any) -> Any:
        # FastAPI validates the decoded JSON object in Python strict mode;
        # arrays represent a fixed pair, not an implicit numeric conversion.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_physics_and_resources(self) -> "LangevinEnsembleParameters":
        LangevinParameters(**self.model_dump(exclude={"seed"}), seed=0)
        return self


class LangevinRequest(LangevinEnsembleParameters):
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)

    def to_parameters(self) -> LangevinParameters:
        return LangevinParameters(**self.model_dump())


class DiffusionResult(_StrictLangevinModel):
    status: Literal["available", "unavailable"]
    estimate: FiniteFloat | None
    standard_error: FiniteFloat | None = Field(ge=0)
    theoretical: FiniteFloat = Field(ge=0)
    method: str
    reason: str | None
    requested_fit_start_step: StrictInt | None
    requested_fit_end_step: StrictInt | None
    fit_start_step: StrictInt | None
    fit_end_step: StrictInt | None
    sample_count: StrictInt = Field(ge=0)
    intercept: FiniteFloat | None
    positive_estimate: bool | None
    minimum_relaxation_times: FiniteFloat = Field(ge=0)
    maximum_relative_finite_window_bias: FiniteFloat = Field(ge=0)
    expected_finite_window_estimate: FiniteFloat | None
    relative_finite_window_bias: FiniteFloat | None
    independent_units: str
    uncertainty_scope: str


class LangevinResponse(_StrictLangevinModel):
    task_id: str
    attempt_id: str
    status: Literal["completed"]
    model_version: Literal["langevin-free-v1"]
    seed: StrictInt
    num_steps: StrictInt
    thermal_temperature: FiniteFloat = Field(ge=0)
    thermal_temperature_standard_error: FiniteFloat = Field(ge=0)
    kinetic_energy: FiniteFloat = Field(ge=0)
    mean_velocity: list[FiniteFloat] = Field(min_length=2, max_length=2)
    mean_displacement: list[FiniteFloat] = Field(min_length=2, max_length=2)
    final_msd: FiniteFloat = Field(ge=0)
    final_centered_msd: FiniteFloat = Field(ge=0)
    diffusion: DiffusionResult
    units: dict[str, str]
    artifacts: dict[Literal["manifest", "final_state", "trajectory", "transport"], str]
