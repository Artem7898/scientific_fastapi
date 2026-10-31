"""Discover, inspect and continue registered numerical checkpoints."""

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response

from ....core.task_store import CheckpointStoreError, QueueFull
from ....domain.experiments import parameters_from_payload
from ....schemas.checkpoint import (
    CheckpointDescriptor,
    CheckpointDetails,
    CheckpointList,
    CheckpointResumeAccepted,
    CheckpointResumeRequest,
)
from ....services.checkpoint_service import (
    CheckpointError,
    inspect_checkpoint,
    load_checkpoint,
    read_checkpoint_artifact,
)
from .simulations import Manager, _valid_task_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulations", tags=["Checkpoints and continuation"])


def _error(code: str, status_code: int) -> HTTPException:
    messages = {
        404: "Task or checkpoint not found",
        409: "The checkpoint cannot be used for this continuation",
        429: "Task capacity is exhausted; retry later",
        503: "Checkpoint storage or execution preparation is unavailable",
    }
    headers = {"Cache-Control": "no-store"}
    if status_code == 429:
        headers["Retry-After"] = "1"
    message = messages.get(status_code, "Could not read checkpoint")
    if code == "CHECKPOINT_PREFLIGHT_CAPACITY_EXHAUSTED":
        message = "Checkpoint read capacity is exhausted in this API process; retry later"
    return HTTPException(
        status_code,
        {"code": code, "message": message},
        headers=headers,
    )


async def _read(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await _result(asyncio.to_thread(operation, *args, **kwargs))


async def _result(computation: Any) -> Any:
    try:
        return await computation
    except CheckpointError as error:
        raise _error(error.code, error.status_code) from None
    except CheckpointStoreError as error:
        status = 404 if error.code in {"TASK_NOT_FOUND", "CHECKPOINT_NOT_FOUND"} else 409
        raise _error(error.code, status) from None
    except QueueFull:
        raise _error("TASK_CAPACITY_EXHAUSTED", 429) from None
    except Exception:
        logger.exception("Could not read or prepare a scientific checkpoint")
        raise _error("CHECKPOINT_STORAGE_UNAVAILABLE", 503) from None


async def _bounded_read(
    request: Request, operation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    # Allocate state arrays for at most two concurrent preflights/downloads in
    # this application process. Inspecting metadata only uses streaming hashes.
    capacity = getattr(request.app.state, "_checkpoint_read_capacity", None)
    if capacity is None:
        capacity = BoundedSemaphore(2)
        request.app.state._checkpoint_read_capacity = capacity
    if not capacity.acquire(blocking=False):
        raise _error("CHECKPOINT_PREFLIGHT_CAPACITY_EXHAUSTED", 429)

    def execute() -> Any:
        try:
            return operation(*args, **kwargs)
        finally:
            # Cancelling an HTTP coroutine cannot stop an already submitted
            # OS-thread job. Retain admission until the actual work finishes.
            capacity.release()

    try:
        future = asyncio.get_running_loop().run_in_executor(None, execute)
    except Exception:
        capacity.release()
        logger.exception("Could not submit a scientific checkpoint read")
        raise _error("CHECKPOINT_STORAGE_UNAVAILABLE", 503) from None
    except BaseException:
        capacity.release()
        raise

    def finished(done: asyncio.Future[Any]) -> None:
        if not done.cancelled():
            done.exception()

    future.add_done_callback(finished)
    # Shield queued work as well as running work: cancelling its wrapper must
    # not remove the queued execute() call and bypass its finally/release.
    return await _result(asyncio.shield(future))


def _validate_ids(task_id: str, checkpoint_id: str | None = None) -> None:
    if not _valid_task_id(task_id):
        raise _error("TASK_NOT_FOUND", 404)
    if checkpoint_id is not None and not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
        raise _error("CHECKPOINT_NOT_FOUND", 404)


async def _descriptor(manager: Manager, task_id: str, checkpoint_id: str) -> dict:
    _validate_ids(task_id, checkpoint_id)
    descriptor = await _read(manager.backend.get_checkpoint, task_id, checkpoint_id)
    if descriptor is None:
        raise _error("CHECKPOINT_NOT_FOUND", 404)
    return descriptor


@router.get(
    "/{task_id}/checkpoints",
    response_model=CheckpointList,
    summary="List registered checkpoint metadata without reading numerical state",
)
async def list_checkpoints(task_id: str, response: Response, manager: Manager) -> CheckpointList:
    _validate_ids(task_id)
    descriptors = await _read(manager.backend.list_checkpoints, task_id)
    if descriptors is None:
        raise _error("TASK_NOT_FOUND", 404)
    response.headers["Cache-Control"] = "no-store"
    return CheckpointList(
        task_id=task_id,
        checkpoints=[CheckpointDescriptor.model_validate(item) for item in descriptors],
    )


@router.get(
    "/{task_id}/checkpoints/{checkpoint_id}",
    response_model=CheckpointDetails,
    summary="Verify checkpoint files and inspect metadata without checking CPU compatibility",
)
async def get_checkpoint(
    task_id: str,
    checkpoint_id: str,
    request: Request,
    response: Response,
    manager: Manager,
) -> CheckpointDetails:
    descriptor = await _descriptor(manager, task_id, checkpoint_id)
    metadata = await _read(
        inspect_checkpoint,
        descriptor,
        artifacts_dir=Path(request.app.state.settings.artifacts_dir),
    )
    response.headers["Cache-Control"] = "no-store"
    return CheckpointDetails(
        descriptor=CheckpointDescriptor.model_validate(descriptor), metadata=metadata
    )


@router.get(
    "/{task_id}/checkpoints/{checkpoint_id}/artifacts/{artifact_name}",
    response_class=Response,
    summary="Download a checksum-verified checkpoint manifest or numerical state",
)
async def download_checkpoint_artifact(
    task_id: str,
    checkpoint_id: str,
    artifact_name: Literal["manifest", "state"],
    request: Request,
    manager: Manager,
) -> Response:
    descriptor = await _descriptor(manager, task_id, checkpoint_id)
    payload, filename, media_type = await _bounded_read(
        request,
        read_checkpoint_artifact,
        descriptor,
        artifacts_dir=Path(request.app.state.settings.artifacts_dir),
        name=artifact_name,
    )
    return Response(
        payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.post(
    "/{task_id}/resume",
    response_model=CheckpointResumeAccepted,
    status_code=202,
    summary="Continue a failed or cancelled standalone task from its latest checkpoint",
)
async def resume_simulation(
    task_id: str,
    payload: CheckpointResumeRequest,
    request: Request,
    response: Response,
    manager: Manager,
) -> CheckpointResumeAccepted:
    _validate_ids(task_id)
    archive = await _read(manager.backend.get_task_archive, task_id)
    if archive is None:
        raise _error("TASK_NOT_FOUND", 404)
    if archive.get("ensemble_id") is not None:
        raise _error("RESUME_ENSEMBLE_UNSUPPORTED", 409)
    status = archive.get("record", {}).get("status")
    if status not in {"failed", "cancelled", "pending", "running"}:
        raise _error("RESUME_STATE_CONFLICT", 409)
    if payload.checkpoint_id is None:
        selected = archive.get("selected_checkpoint")
        if status in {"pending", "running"} and isinstance(selected, dict):
            # An idempotent retry of {} keeps the checkpoint already admitted
            # for this continuation, even after its worker saves a newer one.
            descriptor = selected
        else:
            descriptor = await _read(manager.backend.get_latest_checkpoint, task_id)
        if descriptor is None:
            raise _error("CHECKPOINT_NOT_FOUND", 404)
    else:
        descriptor = await _descriptor(manager, task_id, payload.checkpoint_id)
    checkpoint_id = descriptor["checkpoint_id"]
    if status in {"pending", "running"}:
        selected = archive.get("selected_checkpoint")
        if not isinstance(selected, dict) or selected.get("checkpoint_id") != checkpoint_id:
            raise _error("RESUME_STATE_CONFLICT", 409)
        # Repeating an already admitted continuation does not rehydrate a
        # running worker or require its original files to be loaded again.
    else:
        latest = await _read(manager.backend.get_latest_checkpoint, task_id)
        if latest is None or latest.get("checkpoint_id") != checkpoint_id:
            raise _error("CHECKPOINT_NOT_LATEST", 409)

        def prepare() -> None:
            try:
                parameters = parameters_from_payload(archive["payload"])
            except (ValueError, KeyError, TypeError):
                raise CheckpointStoreError("CHECKPOINT_PARAMETERS_MISMATCH") from None
            load_checkpoint(
                descriptor,
                parameters=parameters,
                artifacts_dir=Path(request.app.state.settings.artifacts_dir),
                validate_fingerprint=manager.settings.execution_mode != "api",
            )

        await _bounded_read(request, prepare)
    with manager.telemetry.span(
        "simulation.resume", attributes={"simulation.task_id": task_id}, kind="producer"
    ):
        accepted = await _read(
            manager.backend.request_resume, task_id, checkpoint_id, manager.telemetry.inject()
        )
    check_url = f"/api/v1/simulations/status/{task_id}"
    response.headers.update(
        {"Location": check_url, "Retry-After": "1", "Cache-Control": "no-store"}
    )
    return CheckpointResumeAccepted(**accepted, check_url=check_url)
