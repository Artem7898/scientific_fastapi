from typing import Any
import uuid
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ....schemas.simulation import SimulationRequest, SimulationResponse
from ....core.queue_manager import queue_manager
from ....core.task_store import task_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulations", tags=["Computational Physics"])


class TaskAcceptedResponse(BaseModel):
    """Response returned when a task is successfully queued."""
    task_id: str = Field(..., description="Unique identifier for the computation task")
    status: str = Field(default="pending", description="Current status of the task")
    check_url: str = Field(..., description="URL to poll for task results")


@router.post(
    "/monte-carlo",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,  # 202 Accepted - стандарт для асинхронных задач!
    summary="Submit a stochastic dynamics simulation",
)

async def submit_simulation(payload: SimulationRequest):
    """
    Submits a simulation to the background queue.
    Does not wait for the computation to finish.
    """
    task_id = uuid.uuid4().hex

    try:
        await queue_manager.add_task(task_id, payload)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue is currently unavailable."
        )

    return TaskAcceptedResponse(
        task_id=task_id,
        check_url=f"/api/v1/simulations/status/{task_id}"
    )


@router.get(
    "/status/{task_id}",
    response_model=SimulationResponse | dict[str, Any],
    summary="Get simulation results by task ID"
)
async def get_simulation_status(task_id: str):
    """
    Polls the status of a previously submitted simulation.
    Returns 404 if task is not found, or the final result/error if completed.
    """
    result = task_store.get(task_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found or expired."
        )

    # Если задача еще в работе (worker не успел обновить статус)
    if not result:
        return {"task_id": task_id, "status": "processing"}

    return result