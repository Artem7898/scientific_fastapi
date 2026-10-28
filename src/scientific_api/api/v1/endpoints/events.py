"""Bounded, resumable task progress streams and cooperative cancellation."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from .simulations import Manager, _valid_task_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulations", tags=["Task progress"])
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_HEARTBEAT_SECONDS = 15.0


class _ConnectionLimiter:
    """Nonwaiting admission on the application's event loop."""

    def __init__(self, capacity: int) -> None:
        self._semaphore = asyncio.BoundedSemaphore(capacity)

    async def acquire(self) -> bool:
        # A successful semaphore acquisition does not suspend when capacity is
        # available. No connection waits or allocates a dedicated worker.
        if self._semaphore.locked():
            return False
        await self._semaphore.acquire()
        return True

    def release(self) -> None:
        self._semaphore.release()


class _ProgressResponse(StreamingResponse):
    """Release admission even when disconnect happens before iteration starts."""

    def __init__(self, content: AsyncIterator[str], limiter: _ConnectionLimiter) -> None:
        self._limiter = limiter
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._limiter.release()


def _frame(event: str, data: dict[str, Any], event_id: str | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    return f"{prefix}event: {event}\ndata: {payload}\n\n"


async def _progress(
    manager: Manager, task_id: str, initial: dict[str, Any], cursor: str | None
) -> AsyncIterator[str]:
    page = initial
    last_output = time.monotonic()
    while True:
        snapshot = page["snapshot"]
        latest_id = page["last_event_id"]
        terminal_sent = False
        if page["reset"]:
            cursor = latest_id
            yield _frame("reset", snapshot, cursor)
            last_output = time.monotonic()
            terminal_sent = snapshot["status"] in _TERMINAL
        else:
            for event in page["events"]:
                cursor = event["event_id"]
                yield _frame("progress", event["data"], cursor)
                last_output = time.monotonic()
                terminal_sent = event["data"]["status"] in _TERMINAL

        # A page's snapshot can already be terminal while its first retained
        # events are not. Drain all pages before closing. A client reconnecting
        # at the terminal ID still receives the terminal snapshot once.
        caught_up = cursor == latest_id
        if snapshot["status"] in _TERMINAL and caught_up:
            if not terminal_sent:
                yield _frame("progress", snapshot, latest_id)
            return
        if not caught_up and page["events"]:
            pass  # Fetch the next page immediately, without adding poll latency.
        else:
            if time.monotonic() - last_output >= _HEARTBEAT_SECONDS:
                yield ": keep-alive\n\n"
                last_output = time.monotonic()
            await asyncio.sleep(manager.settings.event_poll_interval_seconds)
        try:
            page = await asyncio.to_thread(manager.backend.events, task_id, cursor)
        except Exception:
            logger.exception("Could not read task progress during an event stream")
            yield _frame("error", {"task_id": task_id, "error": "Task storage is unavailable"})
            return
        if page is None:
            yield _frame("expired", {"task_id": task_id, "status": "expired"})
            return


@router.get(
    "/{task_id}/events",
    response_class=StreamingResponse,
    summary="Stream resumable task progress without rerunning the simulation",
)
async def task_events(
    task_id: str,
    request: Request,
    manager: Manager,
    last_event_id: Annotated[
        str | None,
        Header(alias="Last-Event-ID", max_length=64, pattern=r"^[0-9]+-[0-9]+$"),
    ] = None,
) -> StreamingResponse:
    if not _valid_task_id(task_id):
        raise HTTPException(404, "Task not found")
    try:
        page = await asyncio.to_thread(manager.backend.events, task_id, last_event_id)
    except Exception:
        logger.exception("Could not open task progress stream")
        raise HTTPException(503, "Task storage is unavailable") from None
    if page is None:
        raise HTTPException(404, "Task not found or expired")

    limiter = getattr(request.app.state, "_sse_connections", None)
    if limiter is None:
        limiter = _ConnectionLimiter(manager.settings.sse_max_connections)
        request.app.state._sse_connections = limiter
    if not await limiter.acquire():
        raise HTTPException(
            429, "Progress stream capacity is exhausted; retry later", headers={"Retry-After": "1"}
        )
    try:
        return _ProgressResponse(_progress(manager, task_id, page, last_event_id), limiter)
    except BaseException:
        limiter.release()
        raise


@router.post(
    "/{task_id}/cancel", status_code=202, summary="Request cooperative cancellation of a task"
)
async def cancel_task(task_id: str, response: Response, manager: Manager) -> dict[str, str]:
    if not _valid_task_id(task_id):
        raise HTTPException(404, "Task not found")
    try:
        accepted = await asyncio.to_thread(manager.backend.request_cancel, task_id)
        record = await asyncio.to_thread(manager.backend.get, task_id) if not accepted else None
    except Exception:
        logger.exception("Could not request task cancellation")
        raise HTTPException(503, "Task storage is unavailable") from None
    if not accepted:
        if record is None:
            raise HTTPException(404, "Task not found or expired")
        if record["status"] in {"completed", "failed"}:
            raise HTTPException(409, "A completed or failed task cannot be cancelled")
        if record["status"] != "cancelled":
            raise HTTPException(503, "Task cancellation is temporarily unavailable")
    response.headers["Location"] = f"/api/v1/simulations/status/{task_id}"
    return {"task_id": task_id, "status": "cancellation_requested"}
