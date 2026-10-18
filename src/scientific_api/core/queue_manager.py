"""Bounded asynchronous workers around a synchronous scientific use case."""

import asyncio
import contextvars
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any

from ..domain.simulation import SimulationCancelled, SimulationParameters
from ..services.simulation_service import run_simulation
from .config import Settings
from .experiment_tracker import ExperimentTracker
from .task_store import Job, TaskStore
from .telemetry import Telemetry

logger = logging.getLogger(__name__)


class QueueManager:
    def __init__(
        self,
        backend: TaskStore,
        settings: Settings,
        tracker: ExperimentTracker,
        telemetry: Telemetry,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.tracker = tracker
        self.telemetry = telemetry
        self._workers: list[asyncio.Task[None]] = []
        self._events: dict[str, Event] = {}
        self._stopping = False
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_count, thread_name_prefix="ScientificWorker"
        )

    @property
    def ready(self) -> bool:
        return (
            bool(self._workers)
            and not self._stopping
            and all(not worker.done() for worker in self._workers)
        )

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("Queue manager has already started")
        self._workers = [
            asyncio.create_task(self._worker(), name=f"scientific-worker-{number}")
            for number in range(self.settings.worker_count)
        ]

    async def submit(self, task_id: str, payload: dict[str, Any]) -> None:
        if self._stopping:
            raise RuntimeError("Server is shutting down")
        with self.telemetry.span(
            "simulation.enqueue", attributes={"simulation.task_id": task_id}, kind="producer"
        ):
            await asyncio.to_thread(self.backend.submit, task_id, payload, self.telemetry.inject())

    async def _worker(self) -> None:
        while not self._stopping:
            try:
                job = await asyncio.to_thread(self.backend.claim)
            except Exception:
                logger.exception("Could not reserve a simulation task")
                await asyncio.sleep(0.5)
                continue
            if job is None:
                await asyncio.sleep(0.05)
                continue
            await self._execute(job)

    async def _heartbeat(self, job: Job, cancelled: Event) -> None:
        while True:
            await asyncio.sleep(max(1.0, self.settings.task_lease_seconds / 3))
            try:
                alive = await asyncio.to_thread(self.backend.heartbeat, job)
            except Exception:
                logger.exception("Worker lost contact with task storage")
                cancelled.set()
                return
            if not alive:
                logger.warning("Worker lost its task lease for %s", job.task_id)
                cancelled.set()
                return

    async def _release(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self.backend.release, job)
        except Exception:
            # Redis will recover the expired lease after reconnecting.
            logger.exception("Could not release interrupted task %s", job.task_id)

    async def _execute(self, job: Job) -> None:
        cancelled = Event()
        self._events[job.owner] = cancelled
        heartbeat = asyncio.create_task(self._heartbeat(job, cancelled))
        started = time.perf_counter()
        computation: asyncio.Future[dict[str, Any]] | None = None
        try:
            with self.telemetry.span(
                "simulation.execute",
                carrier=job.carrier,
                attributes={"simulation.task_id": job.task_id},
                kind="consumer",
            ):
                self.telemetry.observe_queue_wait(time.time() - job.submitted_at)
                parameters = SimulationParameters(**job.payload)
                context = contextvars.copy_context()
                function = partial(
                    run_simulation,
                    parameters,
                    task_id=job.task_id,
                    attempt_id=job.owner,
                    artifacts_dir=Path(self.settings.artifacts_dir),
                    tracker=self.tracker,
                    cancel_event=cancelled,
                )
                computation = asyncio.get_running_loop().run_in_executor(
                    self._executor, context.run, function
                )
                result = await asyncio.shield(computation)
                for name in ("final_temperature", "total_energy", "temperature_standard_error"):
                    if not math.isfinite(result[name]):
                        raise FloatingPointError("Simulation produced a nonfinite result")
                if result.get("task_id") != job.task_id or result.get("status") != "completed":
                    raise ValueError("Scientific result violates the task contract")
                committed = await asyncio.to_thread(self.backend.finish, job, result)
                if committed:
                    self.telemetry.record_task("completed")
                else:
                    logger.warning("Discarded result from an obsolete task lease")
        except SimulationCancelled:
            await self._release(job)
            self.telemetry.record_task("cancelled")
        except asyncio.CancelledError:
            cancelled.set()
            # An OS thread cannot be killed by cancelling its asyncio wrapper.
            # Release only if the computation has actually stopped. Otherwise
            # leave the lease to expire, preventing an immediate duplicate.
            if computation is None or computation.done():
                if computation is not None and not computation.cancelled():
                    computation.exception()
                await self._release(job)
            elif computation is not None:
                computation.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
            raise
        except Exception as exc:
            logger.exception("Simulation %s failed", job.task_id)
            message = (
                "Simulation produced nonfinite metrics"
                if isinstance(exc, FloatingPointError)
                else "Simulation execution failed; inspect server logs"
            )
            try:
                committed = await asyncio.to_thread(
                    self.backend.finish,
                    job,
                    {"task_id": job.task_id, "status": "failed", "error": message},
                )
                if committed:
                    self.telemetry.record_task("failed")
            except Exception:
                logger.exception("Could not persist failed task; lease will expire")
        finally:
            self.telemetry.observe_execution(time.perf_counter() - started)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._events.pop(job.owner, None)

    async def stop(self) -> None:
        self._stopping = True
        for event in self._events.values():
            event.set()
        if self._workers:
            _, pending = await asyncio.wait(
                self._workers, timeout=self.settings.shutdown_timeout_seconds
            )
            for worker in pending:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            if pending:
                logger.warning(
                    "Shutdown deadline reached; active threads received cooperative cancellation"
                )
        self._executor.shutdown(wait=False, cancel_futures=True)
