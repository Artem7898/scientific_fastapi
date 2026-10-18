"""Task state and leased, at-least-once delivery backends.

Only terminal records expire. A worker must renew its lease and stop computing
before releasing a Redis job. Owner tokens prevent a worker whose lease was
reclaimed from overwriting the current attempt's result. These guarantees do
not make external side effects exactly-once; adapters must be idempotent.

Methods are synchronous: asynchronous callers should use ``asyncio.to_thread``.
Redis is imported only when its backend is selected.
"""

from __future__ import annotations

import copy
import importlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4


class QueueFull(Exception):
    """The backend's pending and running capacity has been reached."""


@dataclass(frozen=True)
class Job:
    task_id: str
    payload: dict[str, Any]
    carrier: dict[str, str]
    submitted_at: float
    owner: str


class TaskStore(Protocol):
    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None: ...

    def get(self, task_id: str) -> dict[str, Any] | None: ...
    def claim(self) -> Job | None: ...
    def heartbeat(self, job: Job) -> bool: ...
    def finish(self, job: Job, result: dict[str, Any]) -> bool: ...
    def release(self, job: Job) -> bool: ...
    def health(self) -> bool: ...
    def close(self) -> None: ...


def _now() -> float:
    return time.time()


def _validate_options(
    max_queue_size: int, ttl_seconds: float, lease_seconds: float, max_attempts: int
) -> None:
    if max_queue_size < 1 or max_attempts < 1:
        raise ValueError("Queue capacity and maximum attempts must be positive")
    if not (0 < ttl_seconds < float("inf")) or not (0 < lease_seconds < float("inf")):
        raise ValueError("Record TTL and lease duration must be finite and positive")


def _terminal_result(task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in {"completed", "failed", "cancelled"}:
        raise ValueError("A finished task must have a terminal status")
    record = copy.deepcopy(result)
    record["task_id"] = task_id
    # Transport metadata is private even if an adapter accidentally includes it.
    for field in ("payload", "carrier", "owner"):
        record.pop(field, None)
    return record


@dataclass
class _Entry:
    payload: dict[str, Any]
    carrier: dict[str, str]
    submitted_at: float
    record: dict[str, Any]
    attempts: int = 0
    owner: str | None = None
    lease_until: float = 0.0
    expires_at: float | None = None


class MemoryTaskStore:
    """Per-application backend; restarting the process loses its task state."""

    def __init__(
        self,
        max_queue_size: int = 100,
        ttl_seconds: float = 3600,
        lease_seconds: float = 30,
        max_attempts: int = 3,
    ) -> None:
        _validate_options(max_queue_size, ttl_seconds, lease_seconds, max_attempts)
        self._capacity = max_queue_size
        self._ttl = ttl_seconds
        self._lease = lease_seconds
        self._max_attempts = max_attempts
        self._entries: dict[str, _Entry] = {}
        self._pending: deque[str] = deque()
        self._active = 0
        self._lock = threading.Lock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Task backend is closed")

    def _cleanup(self, now: float) -> None:
        for task_id, entry in list(self._entries.items()):
            if entry.expires_at is not None and entry.expires_at <= now:
                del self._entries[task_id]

    def _complete(self, entry: _Entry, record: dict[str, Any], now: float) -> None:
        entry.record = record
        entry.payload = {}
        entry.carrier = {}
        entry.owner = None
        entry.lease_until = 0
        entry.expires_at = now + self._ttl
        self._active -= 1

    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None:
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            if task_id in self._entries:
                raise ValueError(f"Task already exists: {task_id}")
            if self._active >= self._capacity:
                raise QueueFull("Pending and running task capacity has been reached")
            self._entries[task_id] = _Entry(
                copy.deepcopy(payload),
                dict(carrier),
                now,
                {"task_id": task_id, "status": "pending"},
            )
            self._pending.append(task_id)
            self._active += 1

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            self._cleanup(_now())
            entry = self._entries.get(task_id)
            return copy.deepcopy(entry.record) if entry else None

    def claim(self) -> Job | None:
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            for task_id, entry in self._entries.items():
                if entry.record["status"] != "running" or entry.lease_until > now:
                    continue
                if entry.attempts >= self._max_attempts:
                    self._complete(
                        entry,
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "error": "Worker lease expired; maximum attempts exhausted",
                        },
                        now,
                    )
                else:
                    entry.record = {"task_id": task_id, "status": "pending"}
                    entry.owner = None
                    self._pending.append(task_id)
            if not self._pending:
                return None
            task_id = self._pending.popleft()
            entry = self._entries[task_id]
            entry.owner = str(uuid4())
            entry.attempts += 1
            entry.lease_until = now + self._lease
            entry.record = {"task_id": task_id, "status": "running"}
            return Job(
                task_id,
                copy.deepcopy(entry.payload),
                dict(entry.carrier),
                entry.submitted_at,
                entry.owner,
            )

    def _owned(self, job: Job, now: float) -> _Entry | None:
        entry = self._entries.get(job.task_id)
        if (
            entry is not None
            and entry.record["status"] == "running"
            and entry.owner == job.owner
            and entry.lease_until > now
        ):
            return entry
        return None

    def heartbeat(self, job: Job) -> bool:
        with self._lock:
            self._ensure_open()
            now = _now()
            entry = self._owned(job, now)
            if entry is None:
                return False
            entry.lease_until = now + self._lease
            return True

    def finish(self, job: Job, result: dict[str, Any]) -> bool:
        record = _terminal_result(job.task_id, result)
        with self._lock:
            self._ensure_open()
            now = _now()
            entry = self._owned(job, now)
            if entry is None:
                return False
            self._complete(entry, record, now)
            return True

    def release(self, job: Job) -> bool:
        # In-memory jobs cannot survive process shutdown, so make loss explicit.
        return self.finish(
            job, {"status": "cancelled", "error": "Simulation cancelled during shutdown"}
        )

    def health(self) -> bool:
        with self._lock:
            return not self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True


# Lua keeps admission, lease reclamation and owner comparisons atomic across
# processes. Redis TIME avoids differences between worker-machine clocks.
_TIME = "local t=redis.call('TIME'); local now=tonumber(t[1])+tonumber(t[2])/1000000; "

_SUBMIT = (
    _TIME
    + """
if redis.call('EXISTS',KEYS[1]) == 1 then return -2 end
if redis.call('SCARD',KEYS[3]) >= tonumber(ARGV[4]) then return -1 end
redis.call('HSET',KEYS[1], 'payload',ARGV[2], 'carrier',ARGV[3],
    'submitted_at',tostring(now), 'attempts',0, 'state','pending',
    'record',cjson.encode({task_id=ARGV[1],status='pending'}))
redis.call('RPUSH',KEYS[2],ARGV[1]); redis.call('SADD',KEYS[3],ARGV[1])
return 1
"""
)

_CLAIM = (
    _TIME
    + """
local expired=redis.call('ZRANGEBYSCORE',KEYS[2],'-inf',now)
for _,id in ipairs(expired) do
    local key=ARGV[1]..id
    if redis.call('HGET',key,'state') == 'running' then
        if tonumber(redis.call('HGET',key,'attempts') or '0') >= tonumber(ARGV[4]) then
            redis.call('HSET',key,'state','failed','record',cjson.encode({
                task_id=id,status='failed',
                error='Worker lease expired; maximum attempts exhausted'}))
            redis.call('HDEL',key,'owner','lease_until','payload','carrier')
            redis.call('PEXPIRE',key,ARGV[5]); redis.call('SREM',KEYS[3],id)
        else
            redis.call('HSET',key,'state','pending','record',
                cjson.encode({task_id=id,status='pending'}))
            redis.call('HDEL',key,'owner','lease_until')
            redis.call('RPUSH',KEYS[1],id)
        end
    elseif redis.call('EXISTS',key) == 0 then
        redis.call('SREM',KEYS[3],id)
    end
    redis.call('ZREM',KEYS[2],id)
end
while true do
    local id=redis.call('LPOP',KEYS[1])
    if not id then return nil end
    local key=ARGV[1]..id
    if redis.call('HGET',key,'state') == 'pending' then
        redis.call('HINCRBY',key,'attempts',1)
        redis.call('HSET',key,'state','running','owner',ARGV[2],
            'lease_until',tostring(now+tonumber(ARGV[3])),
            'record',cjson.encode({task_id=id,status='running'}))
        redis.call('ZADD',KEYS[2],now+tonumber(ARGV[3]),id)
        return {id,redis.call('HGET',key,'payload'),
            redis.call('HGET',key,'carrier'),
            redis.call('HGET',key,'submitted_at'),ARGV[2]}
    elseif redis.call('EXISTS',key) == 0 then
        redis.call('SREM',KEYS[3],id)
    end
end
"""
)

_OWNED = (
    _TIME
    + """
if redis.call('HGET',KEYS[1],'state') ~= 'running'
    or redis.call('HGET',KEYS[1],'owner') ~= ARGV[2]
    or tonumber(redis.call('HGET',KEYS[1],'lease_until') or '0') <= now
then return 0 end
"""
)

_HEARTBEAT = (
    _OWNED
    + """
redis.call('HSET',KEYS[1],'lease_until',tostring(now+tonumber(ARGV[3])))
redis.call('ZADD',KEYS[2],now+tonumber(ARGV[3]),ARGV[1]); return 1
"""
)

_FINISH = (
    _OWNED
    + """
redis.call('HSET',KEYS[1],'state',ARGV[3],'record',ARGV[4])
redis.call('HDEL',KEYS[1],'owner','lease_until','payload','carrier')
redis.call('PEXPIRE',KEYS[1],ARGV[5])
redis.call('ZREM',KEYS[2],ARGV[1]); redis.call('SREM',KEYS[3],ARGV[1]); return 1
"""
)

_RELEASE = (
    _OWNED
    + """
redis.call('HSET',KEYS[1],'state','pending','record',
    cjson.encode({task_id=ARGV[1],status='pending'}))
redis.call('HDEL',KEYS[1],'owner','lease_until')
-- Cooperative shutdown does not consume a failed delivery attempt.
redis.call('HINCRBY',KEYS[1],'attempts',-1)
redis.call('ZREM',KEYS[2],ARGV[1]); redis.call('RPUSH',KEYS[3],ARGV[1]); return 1
"""
)


class RedisTaskStore:
    """Shared reliable queue for a standalone Redis instance.

    Creating or closing a client never resets another worker's jobs. Use the
    same namespace to share a queue; choose different namespaces for isolation.
    """

    def __init__(
        self,
        uri: str,
        max_queue_size: int = 100,
        ttl_seconds: float = 3600,
        lease_seconds: float = 30,
        max_attempts: int = 3,
        namespace: str = "scientific-api",
    ) -> None:
        _validate_options(max_queue_size, ttl_seconds, lease_seconds, max_attempts)
        if not namespace:
            raise ValueError("Redis task namespace must not be empty")
        try:
            redis: Any = importlib.import_module("redis")
            retry: Any = importlib.import_module("redis.retry")
            backoff: Any = importlib.import_module("redis.backoff")
        except ImportError as exc:
            raise RuntimeError(
                "Redis backend requires the 'redis' optional extra: install scientific-api[redis]"
            ) from exc
        self._redis: Any = redis.Redis.from_url(
            uri,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry=retry.Retry(backoff.NoBackoff(), 0),
            health_check_interval=30,
        )
        self._capacity = max_queue_size
        self._ttl_ms = max(1, int(ttl_seconds * 1000))
        self._lease = lease_seconds
        self._max_attempts = max_attempts
        self._task_prefix = f"{namespace}:task:"
        self._pending = f"{namespace}:pending"
        self._running = f"{namespace}:running"
        self._active = f"{namespace}:active"
        self._closed = False
        self._submit = self._redis.register_script(_SUBMIT)
        self._claim = self._redis.register_script(_CLAIM)
        self._heartbeat = self._redis.register_script(_HEARTBEAT)
        self._finish = self._redis.register_script(_FINISH)
        self._release = self._redis.register_script(_RELEASE)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Task backend is closed")

    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None:
        self._ensure_open()
        result = self._submit(
            keys=[self._task_prefix + task_id, self._pending, self._active],
            args=[
                task_id,
                json.dumps(payload, allow_nan=False),
                json.dumps(carrier),
                self._capacity,
            ],
        )
        if result == -1:
            raise QueueFull("Pending and running task capacity has been reached")
        if result == -2:
            raise ValueError(f"Task already exists: {task_id}")

    def get(self, task_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        raw = self._redis.hget(self._task_prefix + task_id, "record")
        return json.loads(raw) if raw is not None else None

    def claim(self) -> Job | None:
        self._ensure_open()
        result = self._claim(
            keys=[self._pending, self._running, self._active],
            args=[self._task_prefix, str(uuid4()), self._lease, self._max_attempts, self._ttl_ms],
        )
        if not result:
            return None
        return Job(
            task_id=result[0],
            payload=json.loads(result[1]),
            carrier=json.loads(result[2]),
            submitted_at=float(result[3]),
            owner=result[4],
        )

    def heartbeat(self, job: Job) -> bool:
        self._ensure_open()
        return bool(
            self._heartbeat(
                keys=[self._task_prefix + job.task_id, self._running],
                args=[job.task_id, job.owner, self._lease],
            )
        )

    def finish(self, job: Job, result: dict[str, Any]) -> bool:
        self._ensure_open()
        record = _terminal_result(job.task_id, result)
        return bool(
            self._finish(
                keys=[self._task_prefix + job.task_id, self._running, self._active],
                args=[
                    job.task_id,
                    job.owner,
                    record["status"],
                    json.dumps(record, allow_nan=False),
                    self._ttl_ms,
                ],
            )
        )

    def release(self, job: Job) -> bool:
        self._ensure_open()
        return bool(
            self._release(
                keys=[self._task_prefix + job.task_id, self._running, self._pending],
                args=[job.task_id, job.owner],
            )
        )

    def health(self) -> bool:
        if self._closed:
            return False
        return bool(self._redis.ping())

    def close(self) -> None:
        if not self._closed:
            self._redis.close()
            self._closed = True
