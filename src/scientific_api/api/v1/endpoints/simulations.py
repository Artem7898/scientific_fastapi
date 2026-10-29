"""Submit, inspect and download reproducible simulation tasks."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from ....core.queue_manager import QueueManager
from ....core.task_store import QueueFull
from ....models.computation import BackendUnavailable, require_backend
from ....schemas.langevin import LangevinRequest
from ....schemas.lennard_jones import LennardJonesRequest
from ....schemas.simulation import (
    SimulationRequest,
    TaskAcceptedResponse,
    TaskStatusResponse,
)
from ...dependencies import get_queue_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulations", tags=["Computational Physics"])
Manager = Annotated[QueueManager, Depends(get_queue_manager)]


@router.post(
    "/monte-carlo",
    response_model=TaskAcceptedResponse,
    status_code=202,
    summary="Submit a seeded ideal-gas thermostat trajectory",
)
async def submit_simulation(
    payload: SimulationRequest,
    response: Response,
    manager: Manager,
) -> TaskAcceptedResponse:
    return await _admit_simulation(payload, response, manager)


@router.post(
    "/langevin",
    response_model=TaskAcceptedResponse,
    status_code=202,
    summary="Submit exact 2D Langevin motion, drift and transport diagnostics",
)
async def submit_langevin(
    payload: LangevinRequest,
    response: Response,
    manager: Manager,
) -> TaskAcceptedResponse:
    return await _admit_simulation(payload, response, manager)


@router.post(
    "/lennard-jones",
    response_model=TaskAcceptedResponse,
    status_code=202,
    summary="Submit periodic 2D Lennard–Jones NVE or NVT dynamics",
)
async def submit_lennard_jones(
    payload: LennardJonesRequest,
    response: Response,
    manager: Manager,
) -> TaskAcceptedResponse:
    return await _admit_simulation(payload, response, manager)


async def _admit_simulation(
    payload: SimulationRequest | LangevinRequest | LennardJonesRequest,
    response: Response,
    manager: QueueManager,
) -> TaskAcceptedResponse:
    if payload.num_steps > manager.settings.max_simulation_iterations:
        raise HTTPException(422, "num_steps exceeds the configured iteration limit")
    if payload.num_particles * payload.num_steps > manager.settings.max_particle_steps:
        raise HTTPException(422, "Requested workload exceeds the configured particle-step limit")
    try:
        require_backend(payload.backend)
    except BackendUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    task_id = uuid.uuid4().hex
    try:
        task_payload = payload.model_dump()
        if isinstance(payload, LangevinRequest):
            task_payload = {"model": "langevin-free-v1", **task_payload}
        elif isinstance(payload, LennardJonesRequest):
            task_payload = {"model": "lennard-jones-2d-v1", **task_payload}
        await manager.submit(task_id, task_payload)
    except QueueFull:
        raise HTTPException(
            429, "Task capacity is exhausted; retry later", headers={"Retry-After": "1"}
        ) from None
    except Exception:
        logger.exception("Could not persist a submitted simulation task")
        raise HTTPException(503, "Task storage is unavailable") from None
    check_url = f"/api/v1/simulations/status/{task_id}"
    response.headers["Location"] = check_url
    response.headers["Retry-After"] = "1"
    return TaskAcceptedResponse(task_id=task_id, check_url=check_url)


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Read the state of an accepted simulation",
)
async def get_simulation_status(task_id: str, manager: Manager):
    if not _valid_task_id(task_id):
        raise HTTPException(404, "Task not found")
    try:
        result = await asyncio.to_thread(manager.backend.get, task_id)
    except Exception:
        logger.exception("Could not read simulation status")
        raise HTTPException(503, "Task storage is unavailable") from None
    if result is None:
        raise HTTPException(404, "Task not found or expired")
    return result


def _valid_task_id(task_id: str) -> bool:
    return (
        len(task_id) <= 128
        and bool(task_id)
        and all(
            character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in task_id
        )
    )


@router.get(
    "/{task_id}/artifacts/{artifact_name}",
    summary="Download a committed model-specific run artifact",
)
async def get_artifact(
    task_id: str,
    artifact_name: Literal["manifest", "final_state", "trajectory", "transport", "observables"],
    request: Request,
    manager: Manager,
) -> FileResponse:
    result = await get_simulation_status(task_id, manager)
    if result["status"] != "completed":
        raise HTTPException(409, "Artifacts are available only for completed tasks")
    root = Path(request.app.state.settings.artifacts_dir).resolve()
    attempt_id = result.get("attempt_id", "")
    if not _valid_task_id(attempt_id):
        raise HTTPException(404, "Artifact not found")
    filenames = {"manifest": "manifest.json", "final_state": "final_state.npz"}
    if result.get("model_version") == "langevin-free-v1":
        filenames.update(trajectory="trajectory.npz", transport="transport.json")
    elif result.get("model_version") == "lennard-jones-2d-v1":
        filenames.update(trajectory="trajectory.npz", observables="observables.json")
    if artifact_name not in filenames:
        raise HTTPException(404, "Artifact not found")
    expected = f"{task_id}/{attempt_id}/{filenames[artifact_name]}"
    if result.get("artifacts", {}).get(artifact_name) != expected:
        raise HTTPException(404, "Artifact not found")
    try:
        path = (root / expected).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError
    except (OSError, RuntimeError):
        raise HTTPException(404, "Artifact not found") from None
    return FileResponse(
        path,
        media_type="application/json"
        if artifact_name in {"manifest", "transport", "observables"}
        else "application/octet-stream",
        filename=path.name,
    )
