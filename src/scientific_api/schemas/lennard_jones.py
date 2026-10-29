"""Strict model-specific contracts for periodic 2D Lennard–Jones experiments."""

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

from ..domain.lennard_jones import LennardJonesParameters
from ..domain.simulation import ComputeBackend


class _StrictLennardJonesModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class LennardJonesEnsembleParameters(_StrictLennardJonesModel):
    """Shared physical inputs; independent replica seeds are assigned centrally."""

    num_particles: StrictInt = Field(default=64, ge=3, le=4096)
    box_size: tuple[FiniteFloat, FiniteFloat] = (12.0, 12.0)
    initial_temperature: FiniteFloat = Field(default=1.0, ge=1e-4)
    epsilon: FiniteFloat = Field(default=1.0, ge=1e-3, le=1e3)
    sigma: FiniteFloat = Field(default=1.0, ge=0.1, le=10.0)
    mass: FiniteFloat = Field(default=1.0, ge=0.1, le=10.0)
    time_step: FiniteFloat = Field(default=0.002, gt=0)
    num_steps: StrictInt = Field(default=1000, ge=1, le=10_000)
    ensemble: Literal["nve", "nvt"] = "nve"
    bath_temperature: FiniteFloat | None = Field(default=None, ge=1e-4)
    friction: FiniteFloat | None = Field(default=None, ge=1e-3, le=1e3)
    cutoff_radius: FiniteFloat = Field(default=2.5, gt=0)
    neighbor_skin: FiniteFloat = Field(default=0.3, gt=0)
    interaction_backend: Literal["direct", "verlet"] = "verlet"
    backend: ComputeBackend = "numpy"
    sample_stride: StrictInt = Field(default=10, ge=1, le=10_000)
    equilibration_steps: StrictInt = Field(default=0, ge=0, lt=10_000)
    rdf_bins: StrictInt = Field(default=64, ge=8, le=256)

    @field_validator("box_size", mode="before")
    @classmethod
    def normalize_json_box(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_physics_and_resources(self) -> "LennardJonesEnsembleParameters":
        parameters = LennardJonesParameters(**self.model_dump(exclude={"seed"}), seed=0)
        # A complete accepted payload records the actual thermostat settings,
        # including the NVT default rate, rather than depending on a worker default.
        self.friction = parameters.friction
        return self


class LennardJonesRequest(LennardJonesEnsembleParameters):
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)

    def to_parameters(self) -> LennardJonesParameters:
        return LennardJonesParameters(**self.model_dump())


class LennardJonesResponse(_StrictLennardJonesModel):
    task_id: str
    attempt_id: str
    status: Literal["completed"]
    model_version: Literal["lennard-jones-2d-v1"]
    seed: StrictInt
    num_steps: StrictInt
    ensemble: Literal["nve", "nvt"]
    thermal_temperature: FiniteFloat = Field(ge=0)
    kinetic_energy: FiniteFloat = Field(ge=0)
    potential_energy: FiniteFloat
    total_energy: FiniteFloat
    pressure: FiniteFloat
    diagnostics: dict[str, Any]
    units: dict[str, str]
    artifacts: dict[Literal["manifest", "final_state", "trajectory", "observables"], str]
