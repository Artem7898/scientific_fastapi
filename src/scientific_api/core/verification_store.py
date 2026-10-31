"""Separate leased verification deliveries with durable, immutable terminal records.

The queue implementation is reused in a dedicated namespace. Verification
requests remain opaque JSON: no simulator, checkpoint or source provenance store
is consulted or modified here. Redis payloads and results are never round-tripped
through Lua numbers, preserving uint64 scientific seeds.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from typing import Protocol

from . import task_store
from .task_store import Job, MemoryTaskStore, QueueFull, RedisTaskStore, TaskStore

IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
MAX_VERIFICATIONS_PER_SOURCE = 32
MAX_JSON_BYTES = 2 * 1024 * 1024


class VerificationCapacityError(QueueFull):
    """Admission was refused before allocating any delivery or source slot."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class VerificationConflictError(ValueError):
    """A verification identifier already denotes an immutable request."""

    code = "VERIFICATION_ALREADY_EXISTS"

    def __init__(self):
        super().__init__(self.code)


class VerificationStore(Protocol):
    def submit(
        self, verification_id: str, payload: dict, carrier: dict[str, str] | None = None
    ) -> None: ...
    def get(self, verification_id: str) -> dict | None: ...
    def claim(self) -> Job | None: ...
    def heartbeat(self, job: Job) -> bool: ...
    def finish(self, job: Job, record: dict) -> bool: ...
    def release(self, job: Job) -> bool: ...
    def request_cancel(self, verification_id: str) -> bool: ...
    def is_cancel_requested(self, job: Job) -> bool: ...
    def list_for_source(self, source_id: str) -> list[dict]: ...
    def health(self) -> bool: ...
    def close(self) -> None: ...


def _identity(value: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid verification identifier")


def _json(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValueError("Verification data must be a JSON object")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Verification data exceeds its size limit")
    return raw


def _admission(verification_id: str, payload: dict, carrier: dict | None) -> tuple[str, str, str]:
    _identity(verification_id)
    payload_raw = _json(payload)
    source_id = payload.get("source_id")
    if not isinstance(source_id, str):
        raise ValueError("Verification requests must identify their source")
    _identity(source_id)
    carrier = {} if carrier is None else carrier
    if not isinstance(carrier, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in carrier.items()
    ):
        raise ValueError("Verification tracing headers must be strings")
    return source_id, payload_raw, _json(carrier)


def _options(max_queue_size: int, max_attempts: int, max_per_source: int) -> None:
    if type(max_queue_size) is not int or not 1 <= max_queue_size <= 100_000:
        raise ValueError("Verification queue capacity must be in 1..100000")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
        raise ValueError("Verification maximum attempts must be in 1..100")
    if type(max_per_source) is not int or not 1 <= max_per_source <= MAX_VERIFICATIONS_PER_SOURCE:
        raise ValueError("Verification source capacity must be in 1..32")


def _view(archive: dict) -> dict:
    record = copy.deepcopy(archive["record"])
    record.pop("task_id", None)
    # The reused queue has historical simulation wording for cooperative cancel.
    # Only its exact built-in errors are translated; verifier failures are opaque.
    if record.get("status") == "cancelled" and record.get("error") in {
        "Simulation cancelled by request",
        "Simulation cancelled during shutdown",
    }:
        record["error"] = record["error"].replace("Simulation", "Verification", 1)
    return {
        **record,
        "verification_id": archive["task_id"],
        "request": copy.deepcopy(archive["payload"]),
        "submitted_at": archive["submitted_at"],
        "attempts": len(archive["attempts"]),
    }


class _VerificationQueue[QueueType: TaskStore]:
    _queue: QueueType

    def get(self, verification_id: str) -> dict | None:
        _identity(verification_id)
        archive = self._queue.get_task_archive(verification_id)
        return _view(archive) if archive is not None else None

    def claim(self) -> Job | None:
        return self._queue.claim()

    def heartbeat(self, job: Job) -> bool:
        return self._queue.heartbeat(job)

    def finish(self, job: Job, record: dict) -> bool:
        _identity(job.task_id)
        raw = _json(record)
        stored = json.loads(raw)
        if stored.get("verification_id", job.task_id) != job.task_id:
            raise ValueError("The terminal verification identity must match its delivery")
        for field in ("request", "submitted_at", "attempts"):
            if field in stored:
                raise ValueError("Terminal records cannot replace admission metadata")
        stored["verification_id"] = job.task_id
        return self._queue.finish(job, stored)

    def release(self, job: Job) -> bool:
        return self._queue.release(job)

    def request_cancel(self, verification_id: str) -> bool:
        _identity(verification_id)
        return self._queue.request_cancel(verification_id)

    def is_cancel_requested(self, job: Job) -> bool:
        return self._queue.is_cancel_requested(job)

    def health(self) -> bool:
        return self._queue.health()

    def close(self) -> None:
        self._queue.close()


class MemoryVerificationStore(_VerificationQueue[MemoryTaskStore]):
    """One-process verification queue; all requests and certificates are volatile."""

    def __init__(
        self,
        max_queue_size: int = 100,
        lease_seconds: float = 30,
        max_attempts: int = 3,
        max_per_source: int = MAX_VERIFICATIONS_PER_SOURCE,
        ttl_seconds: float = 3600,
    ) -> None:
        _options(max_queue_size, max_attempts, max_per_source)
        self._queue = MemoryTaskStore(
            max_queue_size=max_queue_size,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            ttl_seconds=ttl_seconds,
        )
        self._sources: dict[str, list[str]] = {}
        self._max_per_source = max_per_source
        self._admission_lock = threading.Lock()

    def submit(
        self, verification_id: str, payload: dict, carrier: dict[str, str] | None = None
    ) -> None:
        source_id, payload_raw, carrier_raw = _admission(verification_id, payload, carrier)
        with self._admission_lock:
            if self.get(verification_id) is not None:
                raise VerificationConflictError()
            previous = self._sources.get(source_id, [])
            if len(previous) >= self._max_per_source:
                raise VerificationCapacityError("VERIFICATION_SOURCE_LIMIT_EXCEEDED")
            try:
                self._queue.submit(
                    verification_id, json.loads(payload_raw), json.loads(carrier_raw)
                )
            except QueueFull as error:
                raise VerificationCapacityError("VERIFICATION_QUEUE_FULL") from error
            self._sources.setdefault(source_id, []).append(verification_id)

    def list_for_source(self, source_id: str) -> list[dict]:
        _identity(source_id)
        with self._admission_lock:
            self._queue._ensure_open()
            # Preserve admission order and do not prune completed source slots.
            identifiers = list(self._sources.get(source_id, []))
            records = [self.get(verification_id) for verification_id in identifiers]
            if any(record is None for record in records):
                raise RuntimeError("Verification source index is inconsistent")
            return [record for record in records if record is not None]

    def release(self, job: Job) -> bool:
        # MemoryTaskStore's public release preserves its older cancellation
        # contract. Verification shutdown instead relinquishes the delivery,
        # matching the shared Redis queue without changing simulation behavior.
        queue = self._queue
        with queue._lock:
            queue._ensure_open()
            now = task_store._now()
            entry = queue._owned(job, now)
            if entry is None:
                return False
            queue._observe(job.task_id, entry, now)
            queue._journal(job.task_id, entry, "released", now, {"reason": "worker_shutdown"})
            if entry.cancel_requested:
                queue._complete(job.task_id, entry, task_store._cancelled(job.task_id), now)
                return True
            entry.record = {"task_id": job.task_id, "status": "pending"}
            entry.owner = None
            entry.lease_until = 0
            entry.attempts -= 1
            entry.completed_steps = 0
            queue._pending.append(job.task_id)
            queue._journal(job.task_id, entry, "retry", now, {"reason": "worker_shutdown"})
            queue._emit(job.task_id, entry, "retry")
            return True


_SUBMIT = (
    task_store._LUA
    + r"""
local id=ARGV[3]
local source=prefix..'verification-source:'..ARGV[7]
if redis.call('EXISTS',taskkey(id))==1 or redis.call('EXISTS',archivekey(id))==1 then return -2 end
if redis.call('LLEN',source)>=tonumber(ARGV[8]) then return -3 end
if redis.call('SCARD',active)>=tonumber(ARGV[6]) then return -1 end
-- Validated request/result JSON stays opaque, including uint64 seeds.
add(id,ARGV[4],ARGV[5],0)
redis.call('RPUSH',source,id)
return 1
"""
)


class RedisVerificationStore(_VerificationQueue[RedisTaskStore]):
    """Durable verification delivery namespace, separate from simulation queues."""

    def __init__(
        self,
        uri: str,
        namespace: str = "scientific-api",
        max_queue_size: int = 100,
        lease_seconds: float = 30,
        max_attempts: int = 3,
        max_per_source: int = MAX_VERIFICATIONS_PER_SOURCE,
        ttl_seconds: float = 3600,
    ) -> None:
        _options(max_queue_size, max_attempts, max_per_source)
        if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9:_-]+", namespace):
            raise ValueError("Invalid verification Redis namespace")
        self._queue = RedisTaskStore(
            uri,
            namespace=namespace + ":verifications",
            max_queue_size=max_queue_size,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            ttl_seconds=ttl_seconds,
        )
        self._max_per_source = max_per_source
        self._submit = self._queue._redis.register_script(_SUBMIT)

    def submit(
        self, verification_id: str, payload: dict, carrier: dict[str, str] | None = None
    ) -> None:
        source_id, payload_raw, carrier_raw = _admission(verification_id, payload, carrier)
        self._queue._ensure_open()
        code = self._submit(
            args=[
                self._queue._prefix,
                self._queue._ttl_ms,
                verification_id,
                payload_raw,
                carrier_raw,
                self._queue._capacity,
                source_id,
                self._max_per_source,
            ]
        )
        if code == -1:
            raise VerificationCapacityError("VERIFICATION_QUEUE_FULL")
        if code == -2:
            raise VerificationConflictError()
        if code == -3:
            raise VerificationCapacityError("VERIFICATION_SOURCE_LIMIT_EXCEEDED")

    def list_for_source(self, source_id: str) -> list[dict]:
        _identity(source_id)
        self._queue._ensure_open()
        identifiers = self._queue._redis.lrange(
            self._queue._prefix + "verification-source:" + source_id,
            0,
            MAX_VERIFICATIONS_PER_SOURCE - 1,
        )
        records = [self.get(verification_id) for verification_id in identifiers]
        if any(record is None for record in records):
            raise RuntimeError("Verification source index is inconsistent")
        return [record for record in records if record is not None]
