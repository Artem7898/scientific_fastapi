"""Shared checkpoint identities and bounds, independent of storage and HTTP."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .experiments import ExperimentParameters
from .langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from .lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from .simulation import MODEL_VERSION

MAX_CHECKPOINTS_PER_TASK = 32
MAX_CHECKPOINT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_CHECKPOINT_ARRAY_BYTES = 128 * 1024 * 1024
MAX_CHECKPOINT_ARCHIVE_BYTES = 160 * 1024 * 1024
MAX_CHECKPOINT_METADATA_BYTES = 2 * 1024 * 1024


def checkpoint_model_version(parameters: ExperimentParameters) -> str:
    if isinstance(parameters, LennardJonesParameters):
        return LJ_MODEL_VERSION
    if isinstance(parameters, LangevinParameters):
        return LANGEVIN_MODEL_VERSION
    return MODEL_VERSION


def checkpoint_parameters(parameters: ExperimentParameters) -> dict[str, Any]:
    """Normalize dataclass defaults and JSON tuples without changing physics."""
    return json.loads(
        json.dumps(
            {"model": checkpoint_model_version(parameters), **asdict(parameters)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def checkpoint_parameters_sha256(parameters: ExperimentParameters) -> str:
    payload = json.dumps(
        checkpoint_parameters(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
