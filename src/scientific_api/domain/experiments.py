"""Dispatch persisted task payloads through a fixed, versioned model registry."""

from typing import Any

from .langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from .lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from .simulation import MODEL_VERSION, SimulationParameters

ExperimentParameters = SimulationParameters | LangevinParameters | LennardJonesParameters


def parameters_from_payload(payload: dict[str, Any]) -> ExperimentParameters:
    """Legacy payloads without a model continue to use the original OU model."""
    if not isinstance(payload, dict):
        raise TypeError("Task payload must be an object")
    values = dict(payload)
    model = values.pop("model", MODEL_VERSION)
    if model == MODEL_VERSION:
        return SimulationParameters(**values)
    if model == LANGEVIN_MODEL_VERSION:
        return LangevinParameters(**values)
    if model == LJ_MODEL_VERSION:
        return LennardJonesParameters(**values)
    raise ValueError("Unknown scientific model")
