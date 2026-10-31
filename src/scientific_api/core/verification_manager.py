"""Separate leased verification consumers; scientific tasks remain read-only."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any
from uuid import uuid4

from ..domain.simulation import SimulationCancelled
from ..schemas.verification import VerificationRecord
from ..services.verification_service import (
    IDENTIFIER,
    VERIFICATION_ID,
    VerificationError,
    certificate_bytes,
    prepare_verification,
    validate_verification_certificate,
    verify_result,
)
from .config import Settings
from .task_store import Job, TaskStore
from .telemetry import Telemetry
from .verification_store import VerificationCapacityError, VerificationStore

logger = logging.getLogger(__name__)


class VerificationManager:
    def __init__(
        self,
        backend: VerificationStore,
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
            max_workers=settings.verification_worker_count, thread_name_prefix="VerificationWorker"
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
            raise RuntimeError("Verification workers have already started")
        self._workers = [
            asyncio.create_task(self._worker(), name=f"verification-worker-{number}")
            for number in range(self.settings.verification_worker_count)
        ]

    async def submit(self, task_id: str, *, rerun: bool = False) -> dict[str, Any]:
        if self._stopping:
            raise VerificationError("VERIFICATION_STORAGE_UNAVAILABLE", 503)
        if not IDENTIFIER.fullmatch(task_id):
            raise VerificationError("VERIFICATION_SOURCE_NOT_FOUND", 404)
        try:
            archive = await asyncio.to_thread(self.source_store.get_task_archive, task_id)
            if archive is None:
                raise VerificationError("VERIFICATION_SOURCE_NOT_FOUND", 404)
            payload = prepare_verification(archive, rerun=rerun)
            identifier = uuid4().hex
            with self.telemetry.span(
                "verification.enqueue", attributes={"verification.id": identifier}, kind="producer"
            ):
                await asyncio.to_thread(
                    self.backend.submit, identifier, payload, self.telemetry.inject()
                )
        except VerificationError:
            raise
        except VerificationCapacityError as error:
            raise VerificationError(error.code, 429) from error
        except Exception as error:
            logger.exception("Could not admit a result verification")
            raise VerificationError("VERIFICATION_STORAGE_UNAVAILABLE", 503) from error
        return {
            "verification_id": identifier,
            "status": "pending",
            "source_task_id": task_id,
            "source_attempt_id": payload["source"]["attempt_id"],
            "check_url": f"/api/v1/verifications/{identifier}",
        }

    def _record(self, identifier: str) -> dict[str, Any]:
        if not VERIFICATION_ID.fullmatch(identifier):
            raise VerificationError("VERIFICATION_NOT_FOUND", 404)
        try:
            record = self.backend.get(identifier)
        except Exception as error:
            raise VerificationError("VERIFICATION_STORAGE_UNAVAILABLE", 503) from error
        if record is None:
            raise VerificationError("VERIFICATION_NOT_FOUND", 404)
        return record

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        source = record["request"]["source"]
        return VerificationRecord(
            verification_id=record["verification_id"],
            status=record["status"],
            source_task_id=source["task_id"],
            source_attempt_id=source["attempt_id"],
            submitted_at=record["submitted_at"],
            attempts=record["attempts"],
            verdict=record.get("verdict"),
            certificate_sha256=record.get("certificate_sha256"),
            error_code=record.get("error_code"),
            error=record.get("error"),
        ).model_dump(mode="json")

    def get(self, identifier: str) -> dict[str, Any]:
        return self._public(self._record(identifier))

    def list(self, task_id: str) -> dict[str, Any]:
        if not IDENTIFIER.fullmatch(task_id):
            raise VerificationError("VERIFICATION_SOURCE_NOT_FOUND", 404)
        try:
            if self.source_store.get_task_archive(task_id) is None:
                raise VerificationError("VERIFICATION_SOURCE_NOT_FOUND", 404)
            records = self.backend.list_for_source(task_id)
        except VerificationError:
            raise
        except Exception as error:
            raise VerificationError("VERIFICATION_STORAGE_UNAVAILABLE", 503) from error
        return {"task_id": task_id, "verifications": [self._public(record) for record in records]}

    def certificate(self, identifier: str) -> tuple[bytes, str]:
        record = self._record(identifier)
        if record["status"] != "completed":
            raise VerificationError("VERIFICATION_CERTIFICATE_NOT_READY")
        try:
            from ..schemas.verification import VerificationCertificate

            data = certificate_bytes(VerificationCertificate.model_validate(record["certificate"]))
            certificate = validate_verification_certificate(
                data, expected_sha256=record["certificate_sha256"]
            )
            if (
                certificate.verification_id != identifier
                or certificate.source.model_dump(mode="json") != record["request"]["source"]
                or certificate.verdict != record["verdict"]
            ):
                raise ValueError("Certificate index mismatch")
        except (KeyError, ValueError, TypeError) as error:
            raise VerificationError("VERIFICATION_CERTIFICATE_INTEGRITY_FAILED") from error
        return data, record["certificate_sha256"]

    async def _worker(self) -> None:
        while not self._stopping:
            try:
                job = await asyncio.to_thread(self.backend.claim)
            except Exception:
                logger.exception("Could not reserve a result verification")
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
                logger.exception("Verification worker lost contact with task storage")
                cancelled.set()
                return

    def _compute(self, job: Job, cancelled: Event) -> dict[str, Any]:
        def check_cancelled() -> None:
            if cancelled.is_set() or self._stopping:
                raise SimulationCancelled("Result verification interrupted")

        # A fresh verifier-owned directory contains optional rerun files only.
        # It is never a child of the pinned source attempt.
        with TemporaryDirectory(prefix="scientific-verification-") as directory:
            certificate = verify_result(
                job.payload,
                verification_id=job.task_id,
                verifier_attempt_id=job.owner,
                artifacts_dir=Path(self.settings.artifacts_dir),
                temporary_root=Path(directory),
                check_cancelled=check_cancelled,
                max_particle_steps=self.settings.verification_rerun_max_particle_steps,
                max_lj_pair_checks=self.settings.verification_rerun_max_lj_pair_checks,
            )
            check_cancelled()
            data = certificate_bytes(certificate)
            validate_verification_certificate(data)
            return {
                "verification_id": job.task_id,
                "status": "completed",
                "verdict": certificate.verdict,
                "certificate": certificate.model_dump(mode="json"),
                "certificate_sha256": hashlib.sha256(data).hexdigest(),
            }

    async def _release(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self.backend.release, job)
        except Exception:
            logger.exception("Could not release verification; its lease will expire")

    async def _execute(self, job: Job) -> None:
        cancelled = Event()
        self._events[job.owner] = cancelled
        heartbeat = asyncio.create_task(self._heartbeat(job, cancelled))
        computation: asyncio.Future[dict[str, Any]] | None = None
        try:
            with self.telemetry.span(
                "verification.execute",
                carrier=job.carrier,
                attributes={"verification.id": job.task_id},
                kind="consumer",
            ):
                context = contextvars.copy_context()
                computation = asyncio.get_running_loop().run_in_executor(
                    self._executor, context.run, partial(self._compute, job, cancelled)
                )
                result = await asyncio.shield(computation)
                if cancelled.is_set() or self._stopping:
                    raise SimulationCancelled("Verification lease interrupted")
                # An ambiguous write is not replaced with an execution-failure
                # record: the terminal commit may have succeeded. Redis leases
                # recover an uncommitted attempt; a committed record is final.
                try:
                    await asyncio.to_thread(self.backend.finish, job, result)
                except Exception:
                    logger.exception("Verification commit was not confirmed; retain stored outcome")
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
        except Exception:
            logger.exception("Result verification execution failed")
            try:
                await asyncio.to_thread(
                    self.backend.finish,
                    job,
                    {
                        "verification_id": job.task_id,
                        "status": "failed",
                        "error_code": "VERIFICATION_EXECUTION_FAILED",
                        "error": "Verification execution failed; inspect worker logs",
                    },
                )
            except Exception:
                logger.exception("Could not persist verification failure; its lease will expire")
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
