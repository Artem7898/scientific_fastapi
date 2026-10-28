"""Submit bounded independent replicas and inspect their durable contributions."""

import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from ....core.queue_manager import QueueManager
from ....core.task_store import QueueFull
from ....schemas.ensemble import (
    EnsembleAccepted,
    EnsembleCancellation,
    EnsembleRequest,
    EnsembleStatus,
)
from ....services.ensemble_service import (
    TERMINAL_STATUSES,
    aggregate_ensemble,
    ensemble_manifest,
    prepare_ensemble,
    valid_ensemble_id,
)
from ...dependencies import get_queue_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ensembles", tags=["Ensembles"])
Manager = Annotated[QueueManager, Depends(get_queue_manager)]
IdempotencyKey = Annotated[
    str | None, Header(alias="Idempotency-Key", min_length=1, max_length=128)
]


@router.post("", response_model=EnsembleAccepted, status_code=202)
async def submit_ensemble(
    payload: EnsembleRequest,
    response: Response,
    manager: Manager,
    idempotency_key: IdempotencyKey = None,
) -> EnsembleAccepted:
    parameters = payload.parameters
    if parameters.num_steps > manager.settings.max_simulation_iterations:
        raise HTTPException(422, "num_steps exceeds the configured iteration limit")
    if (
        payload.replicas * parameters.num_particles * parameters.num_steps
        > manager.settings.max_particle_steps
    ):
        raise HTTPException(422, "Ensemble exceeds the configured total particle-step limit")
    ensemble_id = uuid.uuid4().hex
    try:
        with manager.telemetry.span(
            "ensemble.enqueue",
            attributes={"ensemble.id": ensemble_id, "ensemble.replicas": payload.replicas},
            kind="producer",
        ):
            children, metadata = prepare_ensemble(payload, ensemble_id=ensemble_id)
            ensemble_id = await asyncio.to_thread(
                manager.backend.submit_ensemble,
                ensemble_id,
                children,
                manager.telemetry.inject(),
                metadata,
                idempotency_key=idempotency_key,
            )
    except QueueFull:
        raise HTTPException(
            429, "Task capacity cannot accept this ensemble", headers={"Retry-After": "1"}
        ) from None
    except ValueError:
        raise HTTPException(
            409, "Idempotency-Key was already used for a different request"
        ) from None
    except Exception:
        logger.exception("Could not persist an ensemble")
        raise HTTPException(503, "Task storage is unavailable") from None
    check_url = f"/api/v1/ensembles/{ensemble_id}"
    response.headers["Location"] = check_url
    response.headers["Retry-After"] = "1"
    return EnsembleAccepted(ensemble_id=ensemble_id, check_url=check_url)


async def _snapshot(ensemble_id: str, manager: QueueManager) -> dict:
    if not valid_ensemble_id(ensemble_id):
        raise HTTPException(404, "Ensemble not found")
    try:
        snapshot = await asyncio.to_thread(manager.backend.get_ensemble, ensemble_id)
    except Exception:
        logger.exception("Could not read ensemble %s", ensemble_id)
        raise HTTPException(503, "Task storage is unavailable") from None
    if snapshot is None:
        raise HTTPException(404, "Ensemble not found or expired")
    return snapshot


def _aggregate(snapshot: dict) -> dict:
    try:
        return aggregate_ensemble(snapshot)
    except (ValueError, KeyError, TypeError, OverflowError):
        logger.exception("Ensemble records have inconsistent contributions")
        raise HTTPException(503, "Ensemble records are unavailable") from None


@router.get("/{ensemble_id}", response_model=EnsembleStatus)
async def get_ensemble_status(ensemble_id: str, manager: Manager):
    snapshot = await _snapshot(ensemble_id, manager)
    return _aggregate(snapshot)


@router.post("/{ensemble_id}/cancel", response_model=EnsembleCancellation, status_code=202)
async def cancel_ensemble(ensemble_id: str, manager: Manager) -> EnsembleCancellation:
    snapshot = await _snapshot(ensemble_id, manager)
    if _aggregate(snapshot)["status"] in TERMINAL_STATUSES:
        raise HTTPException(409, "Ensemble has already finished")
    try:
        accepted = await asyncio.to_thread(manager.backend.cancel_ensemble, ensemble_id)
    except Exception:
        logger.exception("Could not request ensemble cancellation")
        raise HTTPException(503, "Task storage is unavailable") from None
    if not accepted:
        raise HTTPException(409, "Ensemble has already finished or expired")
    return EnsembleCancellation(ensemble_id=ensemble_id)


@router.get("/{ensemble_id}/manifest")
async def get_ensemble_manifest(ensemble_id: str, manager: Manager, response: Response):
    snapshot = await _snapshot(ensemble_id, manager)
    if _aggregate(snapshot)["status"] not in TERMINAL_STATUSES:
        raise HTTPException(409, "Manifest is available only after all replicas finish")
    response.headers["Content-Disposition"] = f'attachment; filename="ensemble_{ensemble_id}.json"'
    return ensemble_manifest(snapshot)
