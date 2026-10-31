"""Bounded leased workers for immutable, source-pinned scientific reports."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

from ..domain.simulation import SimulationCancelled
from ..schemas.lj_research import LJResearchRecord, LJResearchReport, LJResearchRequest
from ..services.lj_research_service import (
    ResearchError,
    compute_lj_research,
    prepare_lj_research,
    research_report_bytes,
    validate_lj_research_report,
)
from ..services.verification_service import IDENTIFIER, VERIFICATION_ID
from .config import Settings
from .research_store import ResearchCapacityError, ResearchStore
from .task_store import Job, TaskStore
from .telemetry import Telemetry

logger = logging.getLogger(__name__)


class ResearchManager:
    def __init__(
        self,
        backend: ResearchStore,
        source_store: TaskStore,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.backend = backend
        self.source_store = source_store
        self.settings = settings
        self.telemetry = telemetry
        self._workers: list[asyncio.Task[None]] = []
        self._events: dict[str, Event] = {}
        self._stopping = False
        self._executor = ThreadPoolExecutor(
            max_workers=settings.research_worker_count, thread_name_prefix="ResearchWorker"
        )

    @property
    def ready(self) -> bool:
        return (
            bool(self._workers) and not self._stopping and all(not w.done() for w in self._workers)
        )

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("Research workers have already started")
        self._workers = [
            asyncio.create_task(self._worker(), name=f"research-worker-{number}")
            for number in range(self.settings.research_worker_count)
        ]

    async def submit(self, request: LJResearchRequest) -> dict[str, Any]:
        if self._stopping:
            raise ResearchError("RESEARCH_STORAGE_UNAVAILABLE", 503)
        try:
            payload = await asyncio.to_thread(prepare_lj_research, self.source_store, request)
            identifier = uuid4().hex
            with self.telemetry.span(
                "research.enqueue", attributes={"research.id": identifier}, kind="producer"
            ):
                await asyncio.to_thread(
                    self.backend.submit, identifier, payload, self.telemetry.inject()
                )
        except ResearchError:
            raise
        except ResearchCapacityError as error:
            raise ResearchError(error.code, 429) from error
        except Exception as error:
            logger.exception("Could not admit an LJ research analysis")
            raise ResearchError("RESEARCH_STORAGE_UNAVAILABLE", 503) from error
        return {
            "analysis_id": identifier,
            "status": "pending",
            "kind": request.kind,
            "source_task_ids": request.task_ids,
            "check_url": f"/api/v1/research/analyses/{identifier}",
        }

    def _record(self, identifier: str) -> dict[str, Any]:
        if not VERIFICATION_ID.fullmatch(identifier):
            raise ResearchError("RESEARCH_NOT_FOUND", 404)
        try:
            record = self.backend.get(identifier)
        except Exception as error:
            raise ResearchError("RESEARCH_STORAGE_UNAVAILABLE", 503) from error
        if record is None:
            raise ResearchError("RESEARCH_NOT_FOUND", 404)
        return record

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        options = record["request"]["options"]
        return LJResearchRecord(
            analysis_id=record["analysis_id"],
            status=record["status"],
            kind=options["kind"],
            source_task_ids=options["task_ids"],
            submitted_at=record["submitted_at"],
            attempts=record["attempts"],
            availability=record.get("availability"),
            report_sha256=record.get("report_sha256"),
            error_code=record.get("error_code"),
            error=record.get("error"),
        ).model_dump(mode="json")

    def get(self, identifier: str) -> dict[str, Any]:
        return self._public(self._record(identifier))

    def list(self, task_id: str) -> dict[str, Any]:
        if not IDENTIFIER.fullmatch(task_id):
            raise ResearchError("RESEARCH_SOURCE_NOT_FOUND", 404)
        try:
            if self.source_store.get_task_archive(task_id) is None:
                raise ResearchError("RESEARCH_SOURCE_NOT_FOUND", 404)
            records = self.backend.list_for_source(task_id)
        except ResearchError:
            raise
        except Exception as error:
            raise ResearchError("RESEARCH_STORAGE_UNAVAILABLE", 503) from error
        return {"task_id": task_id, "analyses": [self._public(record) for record in records]}

    def report(self, identifier: str) -> tuple[bytes, str]:
        record = self._record(identifier)
        if record["status"] != "completed":
            raise ResearchError("RESEARCH_REPORT_NOT_READY")
        try:
            data = research_report_bytes(LJResearchReport.model_validate(record["report"]))
            report = validate_lj_research_report(data, expected_sha256=record["report_sha256"])
            if (
                report.analysis_id != identifier
                or report.options.model_dump(mode="json") != record["request"]["options"]
                or [source.identity.model_dump(mode="json") for source in report.sources]
                != [pin["source"] for pin in record["request"]["sources"]]
                or report.availability != record["availability"]
            ):
                raise ValueError("Research report index mismatch")
        except (KeyError, ValueError, TypeError) as error:
            raise ResearchError("RESEARCH_REPORT_INTEGRITY_FAILED") from error
        return data, record["report_sha256"]

    def _compute(self, job: Job, cancelled: Event) -> dict[str, Any]:
        def check_cancelled() -> None:
            if cancelled.is_set() or self._stopping:
                raise SimulationCancelled("Research analysis interrupted")

        report = compute_lj_research(
            job.payload,
            analysis_id=job.task_id,
            analysis_attempt_id=job.owner,
            artifacts_dir=Path(self.settings.artifacts_dir),
            check_cancelled=check_cancelled,
        )
        check_cancelled()
        data = research_report_bytes(report)
        validate_lj_research_report(data)
        return {
            "analysis_id": job.task_id,
            "status": "completed",
            "availability": report.availability,
            "report": report.model_dump(mode="json"),
            "report_sha256": hashlib.sha256(data).hexdigest(),
        }

    async def _worker(self) -> None:
        while not self._stopping:
            try:
                job = await asyncio.to_thread(self.backend.claim)
            except Exception:
                logger.exception("Could not reserve a result research")
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
                if not await asyncio.to_thread(self.backend.heartbeat, job):
                    cancelled.set()
                    return
                if await asyncio.to_thread(self.backend.is_cancel_requested, job):
                    cancelled.set()
                    return
            except Exception:
                logger.exception("Research worker lost contact with task storage")
                cancelled.set()
                return

    async def _release(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self.backend.release, job)
        except Exception:
            logger.exception("Could not release research; its lease will expire")

    async def _execute(self, job: Job) -> None:
        cancelled = Event()
        self._events[job.owner] = cancelled
        heartbeat = asyncio.create_task(self._heartbeat(job, cancelled))
        computation: asyncio.Future[dict[str, Any]] | None = None
        try:
            with self.telemetry.span(
                "research.execute",
                carrier=job.carrier,
                attributes={"research.id": job.task_id},
                kind="consumer",
            ):
                context = contextvars.copy_context()
                computation = asyncio.get_running_loop().run_in_executor(
                    self._executor, context.run, partial(self._compute, job, cancelled)
                )
                result = await asyncio.shield(computation)
                if cancelled.is_set() or self._stopping:
                    raise SimulationCancelled("Research lease interrupted")
                # An ambiguous write is not replaced with an execution-failure
                # record: the terminal commit may have succeeded. Redis leases
                # recover an uncommitted attempt; a committed record is final.
                try:
                    await asyncio.to_thread(self.backend.finish, job, result)
                except Exception:
                    logger.exception("Research commit was not confirmed; retain stored outcome")
        except SimulationCancelled:
            await self._release(job)
        except asyncio.CancelledError:
            cancelled.set()
            if computation is None or computation.done():
                if computation is not None and not computation.cancelled():
                    computation.exception()
                await self._release(job)
            else:
                computation.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
            raise
        except Exception as error:
            logger.exception("Research analysis execution failed")
            try:
                await asyncio.to_thread(
                    self.backend.finish,
                    job,
                    {
                        "analysis_id": job.task_id,
                        "status": "failed",
                        "error_code": (
                            error.code
                            if isinstance(error, ResearchError)
                            else "RESEARCH_ANALYSIS_INPUT_INVALID"
                            if isinstance(error, (ValueError, TypeError))
                            else "RESEARCH_EXECUTION_FAILED"
                        ),
                        "error": "Research execution failed; inspect worker logs",
                    },
                )
            except Exception:
                logger.exception("Could not persist research failure; its lease will expire")
        finally:
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
        self._executor.shutdown(wait=False, cancel_futures=True)
