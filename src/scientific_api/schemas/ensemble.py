"""Strict inputs and explicit statistical outputs for independent replicas."""

from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, model_validator

from ..domain.langevin import LangevinParameters
from ..domain.lennard_jones import (
    MAX_LJ_PAIR_EVALUATIONS,
    MAX_LJ_PARTICLE_STEPS,
    LennardJonesParameters,
)
from ..domain.simulation import ComputeBackend
from .langevin import LangevinEnsembleParameters
from .lennard_jones import LennardJonesEnsembleParameters
from .simulation import StrictModel

MAX_ENSEMBLE_TRAJECTORY_BYTES = 512 * 1024 * 1024


class EnsembleParameters(StrictModel):
    """One physical experiment; seeds belong to the ensemble, not this input."""

    initial_temperature: FiniteFloat = Field(gt=0, le=1e6)
    num_particles: StrictInt = Field(gt=0, le=1_000_000)
    time_step: FiniteFloat = Field(default=0.001, gt=0, le=1000)
    num_steps: StrictInt = Field(default=100, gt=0, le=10_000)
    backend: ComputeBackend = "numpy"


class EnsembleRequest(StrictModel):
    model: Literal["ideal-gas-ou-v1", "langevin-free-v1", "lennard-jones-2d-v1"] = "ideal-gas-ou-v1"
    parameters: EnsembleParameters | LangevinEnsembleParameters | LennardJonesEnsembleParameters
    replicas: StrictInt = Field(default=50, ge=1, le=50)
    base_seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)
    failure_policy: Literal["require_all"] = "require_all"

    @model_validator(mode="before")
    @classmethod
    def select_parameter_contract(cls, value: Any) -> Any:
        # LJ defaults overlap the original model's fields. Select the strict
        # parameter class from the explicit model before validating the union.
        if isinstance(value, dict) and isinstance(value.get("parameters"), dict):
            model = value.get("model", "ideal-gas-ou-v1")
            contract = (
                {
                    "ideal-gas-ou-v1": EnsembleParameters,
                    "langevin-free-v1": LangevinEnsembleParameters,
                    "lennard-jones-2d-v1": LennardJonesEnsembleParameters,
                }.get(model)
                if isinstance(model, str)
                else None
            )
            if contract is not None:
                return {**value, "parameters": contract.model_validate(value["parameters"])}
        return value

    @model_validator(mode="after")
    def bound_work(self) -> "EnsembleRequest":
        expected = {
            "ideal-gas-ou-v1": EnsembleParameters,
            "langevin-free-v1": LangevinEnsembleParameters,
            "lennard-jones-2d-v1": LennardJonesEnsembleParameters,
        }[self.model]
        if not isinstance(self.parameters, expected):
            raise ValueError("parameters do not match the requested model")
        work = self.replicas * self.parameters.num_particles * self.parameters.num_steps
        if work > 100_000_000:
            raise ValueError("replicas * num_particles * num_steps must not exceed 100000000")
        if isinstance(
            self.parameters, (LangevinEnsembleParameters, LennardJonesEnsembleParameters)
        ):
            # The ensemble parameter contract deliberately has no seed. Its
            # validated sampling schedule is independent of the eventual seed.
            parameter_class = (
                LennardJonesParameters
                if isinstance(self.parameters, LennardJonesEnsembleParameters)
                else LangevinParameters
            )
            physics = parameter_class(**self.parameters.model_dump(), seed=0)
            trajectory_bytes = len(physics.sampling_steps()) * physics.num_particles * 2 * 8
            if self.replicas * trajectory_bytes > MAX_ENSEMBLE_TRAJECTORY_BYTES:
                raise ValueError("ensemble sampled position trajectories must not exceed 512 MiB")
            if isinstance(physics, LennardJonesParameters):
                if work > MAX_LJ_PARTICLE_STEPS:
                    raise ValueError("LJ ensemble particle-step budget must not exceed 20000000")
                if (
                    physics.interaction_backend == "direct"
                    and self.replicas * physics.direct_pair_evaluations() > MAX_LJ_PAIR_EVALUATIONS
                ):
                    raise ValueError(
                        "LJ ensemble direct pair evaluations must not exceed 100000000"
                    )
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
    model: Literal["ideal-gas-ou-v1", "langevin-free-v1", "lennard-jones-2d-v1"]
    failure_policy: Literal["require_all"]
    counts: EnsembleCounts
    children: list[EnsembleChild]
    statistics: (
        dict[
            Literal[
                "final_temperature",
                "total_energy",
                "thermal_temperature",
                "kinetic_energy",
                "final_centered_msd",
                "diffusion_estimate",
                "potential_energy",
                "pressure",
            ],
            EnsembleStatistic,
        ]
        | None
    )
    theoretical_temperature_standard_error: FiniteFloat | None = Field(ge=0)


class EnsembleCancellation(StrictModel):
    ensemble_id: str
    status: Literal["cancel_requested"] = "cancel_requested"
