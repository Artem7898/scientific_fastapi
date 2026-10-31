"""Dispatch persisted task payloads through a fixed, versioned model registry."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from .lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from .simulation import MODEL_VERSION, SimulationParameters

ExperimentParameters = SimulationParameters | LangevinParameters | LennardJonesParameters
MODEL_PARAMETER_TYPES: Mapping[str, type[ExperimentParameters]] = MappingProxyType(
    {
        MODEL_VERSION: SimulationParameters,
        LANGEVIN_MODEL_VERSION: LangevinParameters,
        LJ_MODEL_VERSION: LennardJonesParameters,
    }
)


def parameters_from_payload(payload: dict[str, Any]) -> ExperimentParameters:
    """Legacy payloads without a model continue to use the original OU model."""
    if not isinstance(payload, dict):
        raise TypeError("Task payload must be an object")
    values = dict(payload)
    model = values.pop("model", MODEL_VERSION)
    if not isinstance(model, str) or model not in MODEL_PARAMETER_TYPES:
        raise ValueError("Unknown scientific model")
    return MODEL_PARAMETER_TYPES[model](**values)
