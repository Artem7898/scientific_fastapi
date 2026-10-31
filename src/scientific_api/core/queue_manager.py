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

from ..domain.experiments import ExperimentParameters, parameters_from_payload
from ..domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from ..domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from ..domain.simulation import SimulationCancelled
from ..models.lj_forces import NeighborCapacityError, PairBudgetExceeded, ParticleOverlapError
from ..services.checkpoint_service import CheckpointControl, CheckpointError, load_checkpoint
from ..services.provenance_evidence import build_archive_evidence
from ..services.result_envelope_service import read_result_envelope
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
            try:
                if await asyncio.to_thread(self.backend.is_cancel_requested, job):
                    cancelled.set()
                    return
            except Exception:
                logger.exception("Could not check task cancellation")
                cancelled.set()
                return

    async def _release(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self.backend.release, job)
        except Exception:
            # Redis will recover the expired lease after reconnecting.
            logger.exception("Could not release interrupted task %s", job.task_id)

    async def _finish(
        self, job: Job, result: dict[str, Any], *, evidence: dict[str, Any] | None = None
    ) -> bool:
        if evidence is None:
            committed = await asyncio.to_thread(self.backend.finish, job, result)
        else:
            committed = await asyncio.to_thread(self.backend.finish, job, result, evidence=evidence)
        if committed:
            # A concurrent persisted cancellation can atomically replace the
            # requested outcome. Count the actual terminal state.
            record = await asyncio.to_thread(self.backend.get, job.task_id)
            if record is not None:
                self.telemetry.record_task(record["status"])
        return committed

    async def _cancel_or_release(self, job: Job) -> None:
        try:
            if await asyncio.to_thread(self.backend.is_cancel_requested, job):
                await self._finish(
                    job, {"status": "cancelled", "error": "Cancellation requested by client"}
                )
            else:
                await self._release(job)
        except Exception:
            logger.exception("Could not persist interrupted task state for %s", job.task_id)

    def _report_progress(
        self, job: Job, cancelled: Event, completed: int, total: int, phase: str
    ) -> None:
        # Runs in the scientific thread, never waits for the HTTP event loop.
        if cancelled.is_set() or self._stopping or self.backend.is_cancel_requested(job):
            cancelled.set()
            raise SimulationCancelled("Simulation interrupted")
        if not self.backend.report_progress(job, completed, total, phase):
            cancelled.set()
            raise SimulationCancelled("Task lease is no longer current")

    def _register_checkpoint(self, job: Job, cancelled: Event, descriptor: dict[str, Any]) -> None:
        if self._stopping or cancelled.is_set() or self.backend.is_cancel_requested(job):
            cancelled.set()
            raise SimulationCancelled("Checkpoint publication interrupted")
        try:
            current = self.backend.register_checkpoint(job, descriptor)
        except Exception as error:
            # The remote write may have succeeded before its response was lost.
            # Its complete immutable files must remain available to the archive.
            code = getattr(error, "code", "CHECKPOINT_REGISTRATION_UNCONFIRMED")
            raise CheckpointError(code) from error
        if not current:
            cancelled.set()
            raise SimulationCancelled("Checkpoint lease is no longer current")

    def _compute(
        self, parameters: ExperimentParameters, job: Job, cancelled: Event
    ) -> dict[str, Any]:
        # Run compatibility checks on the actual scientific worker thread. API
        # process hardware is not evidence of a remote worker's compatibility.
        options: dict[str, Any] = {
            "task_id": job.task_id,
            "attempt_id": job.owner,
            "artifacts_dir": Path(self.settings.artifacts_dir),
            "tracker": self.tracker,
            "cancel_event": cancelled,
            "progress_callback": partial(self._report_progress, job, cancelled),
        }
        if job.checkpoint is not None or self.settings.checkpoint_interval_steps:
            loaded = (
                load_checkpoint(
                    job.checkpoint,
                    parameters=parameters,
                    artifacts_dir=Path(self.settings.artifacts_dir),
                )
                if job.checkpoint is not None
                else None
            )
            options["checkpoint_control"] = CheckpointControl(
                parameters,
                task_id=job.task_id,
                attempt_id=job.owner,
                artifacts_dir=Path(self.settings.artifacts_dir),
                interval_steps=self.settings.checkpoint_interval_steps,
                loaded=loaded,
                register=partial(self._register_checkpoint, job, cancelled),
            )
        return run_simulation(parameters, **options)

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
                parameters = parameters_from_payload(job.payload)
                context = contextvars.copy_context()
                function = partial(
                    self._compute,
                    parameters,
                    job,
                    cancelled,
                )
                computation = asyncio.get_running_loop().run_in_executor(
                    self._executor, context.run, function
                )
                result = await asyncio.shield(computation)
                if isinstance(parameters, LennardJonesParameters):
                    metrics = (
                        "thermal_temperature",
                        "kinetic_energy",
                        "potential_energy",
                        "total_energy",
                        "pressure",
                    )
                elif isinstance(parameters, LangevinParameters):
                    metrics = (
                        "thermal_temperature",
                        "kinetic_energy",
                        "thermal_temperature_standard_error",
                    )
                else:
                    metrics = ("final_temperature", "total_energy", "temperature_standard_error")
                for name in metrics:
                    if not math.isfinite(result[name]):
                        raise FloatingPointError("Simulation produced a nonfinite result")
                if (
                    isinstance(parameters, LangevinParameters)
                    and result.get("model_version") != LANGEVIN_MODEL_VERSION
                ) or (
                    isinstance(parameters, LennardJonesParameters)
                    and result.get("model_version") != LJ_MODEL_VERSION
                ):
                    raise ValueError("Scientific result has the wrong model version")
                if (
                    result.get("task_id") != job.task_id
                    or result.get("attempt_id") != job.owner
                    or result.get("status") != "completed"
                ):
                    raise ValueError("Scientific result violates the task contract")
                # Admission of a successful new result requires the exact
                # manifest and artifacts verified after computation/tracking.
                # The task store still fences this commit with the lease owner.
                envelope = await asyncio.to_thread(
                    read_result_envelope,
                    result,
                    artifacts_dir=Path(self.settings.artifacts_dir),
                    require_present=True,
                )
                committed = await self._finish(
                    job, result, evidence=build_archive_evidence(envelope)
                )
                if not committed:
                    logger.warning("Discarded result from an obsolete task lease")
        except SimulationCancelled:
            await self._cancel_or_release(job)
        except asyncio.CancelledError:
            cancelled.set()
            # An OS thread cannot be killed by cancelling its asyncio wrapper.
            # Release only if the computation has actually stopped. Otherwise
            # leave the lease to expire, preventing an immediate duplicate.
            if computation is None or computation.done():
                if computation is not None and not computation.cancelled():
                    computation.exception()
                await self._cancel_or_release(job)
            elif computation is not None:
                computation.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
            raise
        except Exception as exc:
            logger.exception("Simulation %s failed", job.task_id)
            if isinstance(exc, CheckpointError):
                message = "Checkpoint continuation failed; inspect the machine-readable error code"
            elif isinstance(exc, ParticleOverlapError):
                message = "Particles approached within 0.5 sigma; reduce timestep or density"
            elif isinstance(exc, PairBudgetExceeded):
                message = "LJ pair-work budget exhausted; reduce particles or steps"
            elif isinstance(exc, NeighborCapacityError):
                message = "LJ neighbor capacity exhausted; reduce particles or density"
            elif isinstance(exc, FloatingPointError):
                message = "Simulation produced nonfinite metrics"
            else:
                message = "Simulation execution failed; inspect server logs"
            try:
                failure = {"task_id": job.task_id, "status": "failed", "error": message}
                if isinstance(exc, CheckpointError):
                    failure["error_code"] = exc.code
                await self._finish(job, failure)
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
