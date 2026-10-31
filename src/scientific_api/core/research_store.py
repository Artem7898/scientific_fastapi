"""Independent durable research deliveries using the tested verification leases.

Only queue terminology is adapted. Research requests, report data and uint64
seeds remain opaque JSON; source simulations, verification records and their
provenance are never modified by this store.
"""

from __future__ import annotations

import copy
from typing import Protocol

from .task_store import Job, QueueFull
from .verification_store import (
    MAX_VERIFICATIONS_PER_SOURCE,
    MemoryVerificationStore,
    RedisVerificationStore,
    VerificationCapacityError,
    VerificationConflictError,
    VerificationStore,
)

MAX_RESEARCH_PER_SOURCE = MAX_VERIFICATIONS_PER_SOURCE
DEFAULT_RESEARCH_QUEUE_SIZE = 16


class ResearchCapacityError(QueueFull):
    """Admission failed without allocating a delivery or source slot."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ResearchConflictError(ValueError):
    """An analysis identifier already belongs to an immutable request."""

    code = "RESEARCH_ALREADY_EXISTS"

    def __init__(self):
        super().__init__(self.code)


class ResearchStore(Protocol):
    def submit(
        self, analysis_id: str, payload: dict, carrier: dict[str, str] | None = None
    ) -> None: ...
    def get(self, analysis_id: str) -> dict | None: ...
    def claim(self) -> Job | None: ...
    def heartbeat(self, job: Job) -> bool: ...
    def finish(self, job: Job, record: dict) -> bool: ...
    def release(self, job: Job) -> bool: ...
    def request_cancel(self, analysis_id: str) -> bool: ...
    def is_cancel_requested(self, job: Job) -> bool: ...
    def list_for_source(self, source_id: str) -> list[dict]: ...
    def health(self) -> bool: ...
    def close(self) -> None: ...


def _research_record(record: dict) -> dict:
    public = copy.deepcopy(record)
    identifier = public.pop("verification_id")
    if public.get("analysis_id", identifier) != identifier:
        raise ValueError("Research record identity disagrees with its delivery")
    public["analysis_id"] = identifier
    if public.get("status") == "cancelled" and public.get("error") in {
        "Verification cancelled by request",
        "Verification cancelled during shutdown",
    }:
        public["error"] = public["error"].replace("Verification", "Research analysis", 1)
    return public


class _ResearchQueue[StoreType: VerificationStore]:
    _store: StoreType

    def submit(
        self, analysis_id: str, payload: dict, carrier: dict[str, str] | None = None
    ) -> None:
        try:
            self._store.submit(analysis_id, payload, carrier)
        except VerificationCapacityError as error:
            codes = {
                "VERIFICATION_SOURCE_LIMIT_EXCEEDED": "RESEARCH_SOURCE_LIMIT_EXCEEDED",
                "VERIFICATION_QUEUE_FULL": "RESEARCH_QUEUE_FULL",
            }
            raise ResearchCapacityError(codes[error.code]) from None
        except VerificationConflictError:
            raise ResearchConflictError() from None

    def get(self, analysis_id: str) -> dict | None:
        record = self._store.get(analysis_id)
        return _research_record(record) if record is not None else None

    def list_for_source(self, source_id: str) -> list[dict]:
        return [_research_record(record) for record in self._store.list_for_source(source_id)]

    def claim(self) -> Job | None:
        return self._store.claim()

    def heartbeat(self, job: Job) -> bool:
        return self._store.heartbeat(job)

    def finish(self, job: Job, record: dict) -> bool:
        if not isinstance(record, dict):
            raise ValueError("Research terminal data must be a JSON object")
        stored = copy.deepcopy(record)
        if (
            stored.get("analysis_id", job.task_id) != job.task_id
            or stored.get("verification_id", job.task_id) != job.task_id
        ):
            raise ValueError("The terminal research identity must match its delivery")
        stored.pop("analysis_id", None)
        stored["verification_id"] = job.task_id
        return self._store.finish(job, stored)

    def release(self, job: Job) -> bool:
        return self._store.release(job)

    def request_cancel(self, analysis_id: str) -> bool:
        return self._store.request_cancel(analysis_id)

    def is_cancel_requested(self, job: Job) -> bool:
        return self._store.is_cancel_requested(job)

    def health(self) -> bool:
        return self._store.health()

    def close(self) -> None:
        self._store.close()


class MemoryResearchStore(_ResearchQueue[MemoryVerificationStore]):
    """Process-local research queue; reports are volatile across restarts."""

    def __init__(
        self,
        max_queue_size: int = DEFAULT_RESEARCH_QUEUE_SIZE,
        lease_seconds: float = 30,
        max_attempts: int = 3,
        max_per_source: int = MAX_RESEARCH_PER_SOURCE,
        ttl_seconds: float = 3600,
    ) -> None:
        self._store = MemoryVerificationStore(
            max_queue_size=max_queue_size,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            max_per_source=max_per_source,
            ttl_seconds=ttl_seconds,
        )


class RedisResearchStore(_ResearchQueue[RedisVerificationStore]):
    """Shared research deliveries isolated from simulation and verification queues."""

    def __init__(
        self,
        uri: str,
        namespace: str = "scientific-api",
        max_queue_size: int = DEFAULT_RESEARCH_QUEUE_SIZE,
        lease_seconds: float = 30,
        max_attempts: int = 3,
        max_per_source: int = MAX_RESEARCH_PER_SOURCE,
        ttl_seconds: float = 3600,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("Invalid research Redis namespace")
        self._store = RedisVerificationStore(
            uri,
            namespace=namespace + ":research",
            max_queue_size=max_queue_size,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            max_per_source=max_per_source,
            ttl_seconds=ttl_seconds,
        )
