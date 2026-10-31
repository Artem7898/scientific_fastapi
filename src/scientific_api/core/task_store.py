"""Atomic leased queues, bounded replay and durable ensemble contributions.

Only operational terminal state expires; the separate provenance journal does not.
Delivery is at least once; owner tokens fence
results and progress, not external side effects. Redis supports a standalone
instance, and is imported only when its backend is selected.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from ..domain.checkpoint import (
    MAX_CHECKPOINT_ARCHIVE_BYTES,
    MAX_CHECKPOINT_METADATA_BYTES,
    MAX_CHECKPOINT_TOTAL_BYTES,
    MAX_CHECKPOINTS_PER_TASK,
    checkpoint_model_version,
    checkpoint_parameters_sha256,
)
from ..domain.experiments import parameters_from_payload

EVENT_LIMIT = 256
TERMINAL = {"completed", "failed", "cancelled"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MAX_ARCHIVE_JSON_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 256 * 1024
MAX_ARCHIVED_ANALYSES = 256
MAX_MANUAL_RESUMES = 32
CHECKPOINT_ID = re.compile(r"[0-9a-f]{32}")


class QueueFull(Exception):
    """Pending and running capacity is exhausted."""


class ArchiveCapacityError(Exception):
    """A task has no room for another immutable analysis occurrence."""


class CheckpointStoreError(ValueError):
    """A bounded machine-readable checkpoint or resume refusal."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Job:
    task_id: str
    payload: dict[str, Any]
    carrier: dict[str, str]
    submitted_at: float
    owner: str
    checkpoint: dict | None = None


class TaskStore(Protocol):
    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None: ...
    def get(self, task_id: str) -> dict[str, Any] | None: ...
    def claim(self) -> Job | None: ...
    def heartbeat(self, job: Job) -> bool: ...
    def finish(self, job: Job, result: dict[str, Any], *, evidence: dict | None = None) -> bool: ...
    def get_task_archive(self, task_id: str) -> dict | None: ...
    def timeline(
        self, task_id: str, after_id: str | None = None, limit: int = 100
    ) -> dict | None: ...
    def register_analysis(
        self, task_id: str, attempt_id: str, analysis_id: str, record: dict, *, evidence: dict
    ) -> None: ...
    def get_analysis_archive(self, analysis_id: str) -> dict | None: ...
    def get_artifact_archive(self, artifact_id: str) -> dict | None: ...
    def register_checkpoint(self, job: Job, descriptor: dict) -> bool: ...
    def get_checkpoint(self, task_id: str, checkpoint_id: str) -> dict | None: ...
    def list_checkpoints(self, task_id: str) -> list[dict] | None: ...
    def get_latest_checkpoint(self, task_id: str) -> dict | None: ...
    def request_resume(self, task_id: str, checkpoint_id: str, carrier: dict) -> dict: ...
    def release(self, job: Job) -> bool: ...
    def report_progress(
        self, job: Job, completed_steps: int, total_steps: int, phase: str
    ) -> bool: ...
    def events(
        self, task_id: str, after_id: str | None = None, limit: int = 100
    ) -> dict | None: ...
    def request_cancel(self, task_id: str) -> bool: ...
    def is_cancel_requested(self, job: Job) -> bool: ...
    def submit_ensemble(
        self,
        ensemble_id: str,
        children: list[tuple[str, dict]],
        carrier: dict,
        metadata: dict,
        idempotency_key: str | None = None,
    ) -> str: ...
    def get_ensemble(self, ensemble_id: str) -> dict | None: ...
    def cancel_ensemble(self, ensemble_id: str) -> bool: ...
    def health(self) -> bool: ...
    def close(self) -> None: ...


def _now() -> float:
    return time.time()


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid archive identifier")


def _archive_json(value: dict, maximum: int = MAX_ARCHIVE_JSON_BYTES) -> str:
    if not isinstance(value, dict):
        raise ValueError("Archive records must be JSON objects")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > maximum:
        raise ValueError("Archive record exceeds its size limit")
    return raw


def _checkpoint_identity(task_id: str, checkpoint_id: str) -> None:
    _identifier(task_id)
    if not isinstance(checkpoint_id, str) or not CHECKPOINT_ID.fullmatch(checkpoint_id):
        raise CheckpointStoreError("CHECKPOINT_DESCRIPTOR_INVALID")


def _checkpoint_descriptor(job: Job, descriptor: dict, original_payload: dict) -> tuple[str, int]:
    try:
        raw = _archive_json(descriptor, MAX_CHECKPOINT_METADATA_BYTES)
        if set(descriptor) != {
            "schema_version",
            "checkpoint_id",
            "task_id",
            "attempt_id",
            "completed_steps",
            "total_steps",
            "model_version",
            "parameters_sha256",
            "fingerprint_sha256",
            "manifest",
            "state",
            "created_at",
        }:
            raise ValueError("Invalid descriptor fields")
        _checkpoint_identity(job.task_id, descriptor["checkpoint_id"])
        if (
            type(descriptor["schema_version"]) is not int
            or descriptor["schema_version"] != 1
            or descriptor["task_id"] != job.task_id
            or descriptor["attempt_id"] != job.owner
            or type(descriptor["completed_steps"]) is not int
            or type(descriptor["total_steps"]) is not int
            or not 0 <= descriptor["completed_steps"] <= descriptor["total_steps"] <= 10_000
            or descriptor["total_steps"] <= 0
            or any(
                not isinstance(descriptor[name], str) or not DIGEST.fullmatch(descriptor[name])
                for name in ("parameters_sha256", "fingerprint_sha256")
            )
            or not isinstance(descriptor["created_at"], str)
            or datetime.fromisoformat(descriptor["created_at"]).tzinfo is None
        ):
            raise ValueError("Invalid descriptor values")
        prefix = f"{job.task_id}/{job.owner}/checkpoints/{descriptor['checkpoint_id']}"
        size = 0
        for name, filename, maximum in (
            ("manifest", "checkpoint.json", MAX_CHECKPOINT_METADATA_BYTES),
            ("state", "state.npz", MAX_CHECKPOINT_ARCHIVE_BYTES),
        ):
            item = descriptor[name]
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "sha256", "bytes"}
                or item["path"] != f"{prefix}/{filename}"
                or not isinstance(item["sha256"], str)
                or not DIGEST.fullmatch(item["sha256"])
                or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= maximum
            ):
                raise ValueError("Invalid checkpoint artifact")
            size += item["bytes"]
        parameters = parameters_from_payload(original_payload)
        if (
            checkpoint_parameters_sha256(parameters) != descriptor["parameters_sha256"]
            or checkpoint_parameters_sha256(parameters_from_payload(job.payload))
            != descriptor["parameters_sha256"]
            or checkpoint_model_version(parameters) != descriptor["model_version"]
            or parameters.num_steps != descriptor["total_steps"]
        ):
            raise CheckpointStoreError("CHECKPOINT_PARAMETERS_MISMATCH")
        return raw, size
    except CheckpointStoreError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise CheckpointStoreError("CHECKPOINT_DESCRIPTOR_INVALID") from error


def _timeline_cursor(after_id: str | None, limit: int) -> None:
    _cursor(after_id, limit)
    if after_id is not None and (
        len(after_id) > 41
        or not re.fullmatch(r"[0-9]+-[0-9]+", after_id)
        or any(int(part) > 2**64 - 1 for part in after_id.split("-"))
    ):
        raise ValueError("Invalid archive event cursor")


def _evidence_records(
    task_id: str, attempt_id: str, analysis_id: str | None, evidence: dict | None
) -> tuple[str, list[dict]]:
    if evidence is None:
        return "", []
    raw = _archive_json(evidence, MAX_EVIDENCE_BYTES)
    if (
        type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
        or not isinstance(evidence.get("model_version"), str)
        or not isinstance(evidence.get("model_contract_sha256"), str)
        or not DIGEST.fullmatch(evidence["model_contract_sha256"])
        or not isinstance(evidence.get("manifest_sha256"), str)
        or not DIGEST.fullmatch(evidence["manifest_sha256"])
        or not isinstance(evidence.get("manifest"), dict)
        or not isinstance(evidence.get("artifacts"), list)
        or not 1 <= len(evidence["artifacts"]) <= 32
    ):
        raise ValueError("Invalid archive evidence")
    prefix = f"{task_id}/{attempt_id}/"
    if analysis_id is not None:
        prefix += f"analyses/{analysis_id}/"
    by_id = {}
    for descriptor in [evidence["manifest"], *evidence["artifacts"]]:
        if (
            not isinstance(descriptor, dict)
            or any(
                not isinstance(descriptor.get(name), str)
                for name in ("artifact_id", "name", "path", "sha256", "media_type")
            )
            or not DIGEST.fullmatch(descriptor["artifact_id"])
            or not DIGEST.fullmatch(descriptor["sha256"])
            or type(descriptor.get("bytes")) is not int
            or descriptor["bytes"] < 0
            or not descriptor["path"].startswith(prefix)
            or any(part in {"", ".", ".."} for part in descriptor["path"].split("/"))
            or "\\" in descriptor["path"]
        ):
            raise ValueError("Invalid archive artifact descriptor")
        artifact_id = descriptor["artifact_id"]
        if artifact_id in by_id and by_id[artifact_id] != descriptor:
            raise ValueError("Conflicting archive artifact descriptors")
        by_id[artifact_id] = descriptor
    if evidence["manifest"].get("sha256") != evidence["manifest_sha256"]:
        raise ValueError("Manifest evidence digest disagrees")
    locators = [
        {
            "artifact_id": artifact_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "analysis_id": analysis_id,
            "descriptor": descriptor,
        }
        for artifact_id, descriptor in by_id.items()
    ]
    return raw, locators


def _analysis_record(task_id: str, attempt_id: str, analysis_id: str, record: dict) -> str:
    for value in (task_id, attempt_id, analysis_id):
        _identifier(value)
    raw = _archive_json(record)
    if (
        record.get("task_id") != task_id
        or record.get("attempt_id") != attempt_id
        or record.get("analysis_id") != analysis_id
        or record.get("status") != "completed"
    ):
        raise ValueError("Analysis record does not match its identity")
    return raw


def _validate_options(
    max_queue_size: int, ttl_seconds: float, lease_seconds: float, max_attempts: int
) -> None:
    if max_queue_size < 1 or max_attempts < 1:
        raise ValueError("Queue capacity and maximum attempts must be positive")
    if not 0 < ttl_seconds < float("inf") or not 0 < lease_seconds < float("inf"):
        raise ValueError("Record TTL and lease duration must be finite and positive")


def _terminal_result(task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in TERMINAL:
        raise ValueError("A finished task must have a terminal status")
    record = copy.deepcopy(result)
    record["task_id"] = task_id
    for name in ("payload", "carrier", "owner"):
        record.pop(name, None)
    return record


def _cancelled(task_id: str) -> dict[str, str]:
    return {"task_id": task_id, "status": "cancelled", "error": "Simulation cancelled by request"}


def _total(payload: dict) -> int:
    value = payload.get("num_steps", 0)
    return value if type(value) is int and value >= 0 else 0


def _progress(completed_steps: int, total_steps: int, phase: str) -> None:
    if (
        type(completed_steps) is not int
        or type(total_steps) is not int
        or not 0 <= completed_steps <= total_steps
        or total_steps <= 0
        or not isinstance(phase, str)
        or not 1 <= len(phase) <= 64
    ):
        raise ValueError("Progress requires integer 0 <= completed <= total and a bounded phase")


def _cursor(after_id: str | None, limit: int) -> None:
    if after_id is not None and (
        not isinstance(after_id, str) or not re.fullmatch(r"\d+-\d+", after_id)
    ):
        raise ValueError("Event cursor must be a stream ID")
    if type(limit) is not int or not 1 <= limit <= EVENT_LIMIT:
        raise ValueError("Event limit must be in 1..256")


def _ensemble_input(
    ensemble_id: str, children: list[tuple[str, dict]], metadata: dict, idempotency_key: str | None
) -> tuple[str, str | None]:
    if not isinstance(ensemble_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", ensemble_id):
        raise ValueError("Invalid ensemble identifier")
    if not children or len({item[0] for item in children}) != len(children):
        raise ValueError("An ensemble requires unique nonempty children")
    for task_id, payload in children:
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ValueError("Invalid child identifier")
        if not isinstance(payload, dict):
            raise ValueError("Child payload must be an object")
    canonical = json.dumps(
        {"metadata": metadata, "payloads": [p for _, p in children]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    fingerprint = metadata.get("request_fingerprint")
    if fingerprint is None:
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    elif not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
        raise ValueError("request_fingerprint must be a SHA256 digest")
    token = None
    if idempotency_key is not None:
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("Idempotency key must contain 1..256 characters")
        token = hashlib.sha256(idempotency_key.encode()).hexdigest()
    return fingerprint.lower(), token


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
    cancel_requested: bool = False
    ensemble_id: str | None = None
    completed_steps: int = 0
    total_steps: int = 0
    phase: str = "pending"
    sequence: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=EVENT_LIMIT))
    manual_checkpoint_id: str | None = None


@dataclass
class _Ensemble:
    metadata: dict
    children: list[str]
    results: dict[str, dict] = field(default_factory=dict)
    expires_at: float | None = None
    token: str | None = None
    cancel_requested: bool = False


class MemoryTaskStore:
    """One-process development queue; process restart loses all state."""

    def __init__(
        self,
        max_queue_size: int = 100,
        ttl_seconds: float = 3600,
        lease_seconds: float = 30,
        max_attempts: int = 3,
    ) -> None:
        _validate_options(max_queue_size, ttl_seconds, lease_seconds, max_attempts)
        self._capacity, self._ttl = max_queue_size, ttl_seconds
        self._lease, self._max_attempts = lease_seconds, max_attempts
        self._entries: dict[str, _Entry] = {}
        self._archives: dict[str, dict] = {}
        self._journals: dict[str, list[dict]] = {}
        self._analyses: dict[str, dict] = {}
        self._artifacts: dict[str, dict] = {}
        self._ensemble_archives: dict[str, dict] = {}
        self._checkpoints: dict[str, dict[str, dict]] = {}
        self._sse_sequences: dict[str, int] = {}
        self._ensembles: dict[str, _Ensemble] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
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
        for ensemble_id, parent in list(self._ensembles.items()):
            if parent.expires_at is not None and parent.expires_at <= now:
                del self._ensembles[ensemble_id]
                if parent.token and self._idempotency.get(parent.token, (None,))[0] == ensemble_id:
                    self._idempotency.pop(parent.token, None)

    @staticmethod
    def _snapshot(task_id: str, entry: _Entry) -> dict:
        return {
            "task_id": task_id,
            "status": entry.record["status"],
            "phase": entry.phase,
            "attempt": entry.attempts,
            "completed_steps": entry.completed_steps,
            "total_steps": entry.total_steps,
        }

    def _emit(self, task_id: str, entry: _Entry, phase: str) -> None:
        entry.phase = phase
        entry.sequence += 1
        self._sse_sequences[task_id] = entry.sequence
        entry.history.append(
            {"event_id": f"{entry.sequence}-0", "data": self._snapshot(task_id, entry)}
        )

    def _observe(self, task_id: str, entry: _Entry, now: float) -> dict:
        archive = self._archives.get(task_id)
        if archive is not None:
            return archive
        archive = {
            "task_id": task_id,
            "payload": copy.deepcopy(entry.payload),
            "submitted_at": entry.submitted_at,
            "ensemble_id": entry.ensemble_id,
            "record": copy.deepcopy(entry.record),
            "evidence": None,
            "attempts": [],
            "analyses": [],
            "coverage": "partial",
            "storage": "memory",
        }
        self._archives[task_id] = archive
        self._journals[task_id] = []
        owner = entry.owner or entry.record.get("attempt_id")
        if owner:
            archive["attempts"].append(
                {"attempt_id": owner, "attempt": max(1, entry.attempts), "claimed_at": None}
            )
        self._journal(task_id, entry, "history_observed", now, {"coverage": "partial"})
        return archive

    def _journal(
        self, task_id: str, entry: _Entry, event_type: str, now: float, details: dict | None = None
    ) -> None:
        archive = self._archives[task_id]
        attempts = archive["attempts"]
        attempt = attempts[-1] if attempts else None
        events = self._journals[task_id]
        events.append(
            {
                "event_id": f"{len(events) + 1}-0",
                "event_type": event_type,
                "occurred_at": now,
                "attempt_id": attempt["attempt_id"] if attempt else None,
                "attempt": attempt["attempt"] if attempt else 0,
                "status": entry.record["status"],
                "details": copy.deepcopy(details or {}),
            }
        )
        archive["record"] = copy.deepcopy(entry.record)

    def _check_artifacts(self, locators: list[dict]) -> None:
        for locator in locators:
            previous = self._artifacts.get(locator["artifact_id"])
            if previous is not None and previous != locator:
                raise ValueError("Artifact identifier conflicts with an archived occurrence")

    def _index_artifacts(self, locators: list[dict]) -> None:
        for locator in locators:
            self._artifacts[locator["artifact_id"]] = copy.deepcopy(locator)

    def _add(
        self, task_id: str, payload: dict, carrier: dict, now: float, ensemble_id: str | None = None
    ) -> None:
        entry = _Entry(
            copy.deepcopy(payload),
            dict(carrier),
            now,
            {"task_id": task_id, "status": "pending"},
            ensemble_id=ensemble_id,
            total_steps=_total(payload),
        )
        self._entries[task_id] = entry
        self._pending.append(task_id)
        self._active += 1
        self._emit(task_id, entry, "pending")
        self._archives[task_id] = {
            "task_id": task_id,
            "payload": copy.deepcopy(payload),
            "submitted_at": now,
            "ensemble_id": ensemble_id,
            "record": copy.deepcopy(entry.record),
            "evidence": None,
            "attempts": [],
            "analyses": [],
            "coverage": "complete",
            "storage": "memory",
        }
        self._journals[task_id] = []
        self._journal(task_id, entry, "submitted", now)

    def _complete(
        self, task_id: str, entry: _Entry, record: dict, now: float, evidence: dict | None = None
    ) -> None:
        archive = self._observe(task_id, entry, now)
        entry.record = record
        if record["status"] == "completed":
            entry.completed_steps = entry.total_steps
        self._emit(task_id, entry, record["status"])
        archive["evidence"] = copy.deepcopy(evidence) if record["status"] == "completed" else None
        self._journal(task_id, entry, record["status"], now)
        entry.payload, entry.carrier, entry.owner = {}, {}, None
        entry.lease_until = 0
        entry.cancel_requested = False
        self._active -= 1
        parent = self._ensembles.get(entry.ensemble_id) if entry.ensemble_id else None
        if parent is not None:
            parent.results.setdefault(task_id, copy.deepcopy(record))
            if len(parent.results) == len(parent.children):
                parent.expires_at = now + self._ttl
                for child in parent.children:
                    if child in self._entries:
                        self._entries[child].expires_at = now + self._ttl
        else:
            entry.expires_at = now + self._ttl

    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None:
        _identifier(task_id)
        _archive_json(payload)
        _archive_json(carrier)
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            if task_id in self._entries or task_id in self._archives:
                raise ValueError(f"Task already exists: {task_id}")
            if self._active >= self._capacity:
                raise QueueFull("Pending and running task capacity has been reached")
            self._add(task_id, payload, carrier, now)

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
                if entry.cancel_requested:
                    self._complete(task_id, entry, _cancelled(task_id), now)
                elif entry.attempts >= self._max_attempts:
                    self._complete(
                        task_id,
                        entry,
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "error": "Worker lease expired; maximum attempts exhausted",
                        },
                        now,
                    )
                else:
                    self._observe(task_id, entry, now)
                    entry.record = {"task_id": task_id, "status": "pending"}
                    self._journal(task_id, entry, "retry", now, {"reason": "lease_expired"})
                    entry.owner = None
                    entry.completed_steps = 0
                    self._emit(task_id, entry, "retry")
                    self._pending.append(task_id)
            while self._pending:
                task_id = self._pending.popleft()
                entry = self._entries.get(task_id)
                if entry is None or entry.record["status"] != "pending":
                    continue
                archive = self._observe(task_id, entry, now)
                entry.owner = str(uuid4())
                entry.attempts += 1
                entry.lease_until = now + self._lease
                entry.record = {"task_id": task_id, "status": "running"}
                previous = archive["attempts"][-1]["attempt"] if archive["attempts"] else 0
                archive["attempts"].append(
                    {"attempt_id": entry.owner, "attempt": previous + 1, "claimed_at": now}
                )
                checkpoints = self._checkpoints.get(task_id, {})
                checkpoint = next(reversed(checkpoints.values())) if checkpoints else None
                archive["selected_checkpoint"] = copy.deepcopy(checkpoint)
                if checkpoint is not None:
                    entry.completed_steps = checkpoint["completed_steps"]
                    entry.total_steps = checkpoint["total_steps"]
                self._journal(task_id, entry, "attempt_claimed", now)
                if checkpoint is not None:
                    self._journal(
                        task_id,
                        entry,
                        "attempt_resumed",
                        now,
                        {
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "parent_attempt_id": checkpoint["attempt_id"],
                            "completed_steps": checkpoint["completed_steps"],
                        },
                    )
                self._emit(task_id, entry, "running")
                return Job(
                    task_id,
                    copy.deepcopy(entry.payload),
                    dict(entry.carrier),
                    entry.submitted_at,
                    entry.owner,
                    copy.deepcopy(checkpoint),
                )
            return None

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
            entry = self._owned(job, _now())
            if entry is None:
                return False
            entry.lease_until = _now() + self._lease
            return True

    def finish(self, job: Job, result: dict[str, Any], *, evidence: dict | None = None) -> bool:
        record = _terminal_result(job.task_id, result)
        _archive_json(record)
        _, locators = _evidence_records(job.task_id, job.owner, None, evidence)
        if evidence is not None and record.get("attempt_id") != job.owner:
            raise ValueError("Evidence must belong to the completed attempt")
        with self._lock:
            self._ensure_open()
            now = _now()
            entry = self._owned(job, now)
            if entry is None:
                return False
            accepted = record["status"] == "completed" and not entry.cancel_requested
            if accepted:
                self._check_artifacts(locators)
            self._complete(
                job.task_id,
                entry,
                _cancelled(job.task_id) if entry.cancel_requested else record,
                now,
                evidence if accepted else None,
            )
            if accepted:
                self._index_artifacts(locators)
            return True

    def release(self, job: Job) -> bool:
        with self._lock:
            self._ensure_open()
            now = _now()
            entry = self._owned(job, now)
            if entry is None:
                return False
            self._observe(job.task_id, entry, now)
            self._journal(job.task_id, entry, "released", now, {"reason": "worker_shutdown"})
            record = (
                _cancelled(job.task_id)
                if entry.cancel_requested
                else {
                    "task_id": job.task_id,
                    "status": "cancelled",
                    "error": "Simulation cancelled during shutdown",
                }
            )
            self._complete(job.task_id, entry, record, now)
            return True

    def report_progress(self, job: Job, completed_steps: int, total_steps: int, phase: str) -> bool:
        _progress(completed_steps, total_steps, phase)
        with self._lock:
            self._ensure_open()
            entry = self._owned(job, _now())
            if (
                entry is None
                or entry.cancel_requested
                or completed_steps < entry.completed_steps
                or entry.total_steps not in (0, total_steps)
            ):
                return False
            entry.completed_steps, entry.total_steps = completed_steps, total_steps
            self._emit(job.task_id, entry, phase)
            return True

    def events(self, task_id: str, after_id: str | None = None, limit: int = 100) -> dict | None:
        _cursor(after_id, limit)
        with self._lock:
            self._ensure_open()
            self._cleanup(_now())
            entry = self._entries.get(task_id)
            if entry is None:
                return None
            history = list(entry.history)
            ids = [event["event_id"] for event in history]
            reset = after_id not in (None, "0-0") and after_id not in ids
            start = ids.index(after_id) + 1 if after_id in ids else 0
            return {
                "snapshot": self._snapshot(task_id, entry),
                "events": [] if reset else copy.deepcopy(history[start : start + limit]),
                "reset": reset,
                "last_event_id": ids[-1],
            }

    def _cancel(self, task_id: str, now: float) -> bool:
        entry = self._entries.get(task_id)
        if entry is None or entry.record["status"] in {"completed", "failed"}:
            return False
        if entry.record["status"] == "cancelled" or entry.cancel_requested:
            return True
        self._observe(task_id, entry, now)
        self._journal(task_id, entry, "cancel_requested", now)
        if entry.record["status"] == "pending":
            self._complete(task_id, entry, _cancelled(task_id), now)
        else:
            entry.cancel_requested = True
            self._emit(task_id, entry, "cancel_requested")
        return True

    def request_cancel(self, task_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            self._cleanup(_now())
            return self._cancel(task_id, _now())

    def is_cancel_requested(self, job: Job) -> bool:
        with self._lock:
            self._ensure_open()
            entry = self._owned(job, _now())
            return entry is not None and entry.cancel_requested

    def submit_ensemble(
        self,
        ensemble_id: str,
        children: list[tuple[str, dict]],
        carrier: dict,
        metadata: dict,
        idempotency_key: str | None = None,
    ) -> str:
        fingerprint, token = _ensemble_input(ensemble_id, children, metadata, idempotency_key)
        _archive_json(carrier)
        _archive_json(metadata)
        for _, payload in children:
            _archive_json(payload)
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            if token in self._idempotency:
                previous_id, previous_fingerprint = self._idempotency[token]
                if previous_fingerprint != fingerprint:
                    raise ValueError("Idempotency key conflicts with the original request")
                return previous_id
            if (
                ensemble_id in self._ensemble_archives
                or ensemble_id in self._ensembles
                or any(child in self._entries or child in self._archives for child, _ in children)
            ):
                raise ValueError("An ensemble or child already exists")
            if self._active + len(children) > self._capacity:
                raise QueueFull("Insufficient capacity for the entire ensemble")
            self._ensembles[ensemble_id] = _Ensemble(
                copy.deepcopy(metadata), [child for child, _ in children], token=token
            )
            self._ensemble_archives[ensemble_id] = {
                "metadata": copy.deepcopy(metadata),
                "children": [child for child, _ in children],
                "submitted_at": now,
            }
            if token:
                self._idempotency[token] = ensemble_id, fingerprint
            for child, payload in children:
                self._add(child, payload, carrier, now, ensemble_id)
            return ensemble_id

    def get_ensemble(self, ensemble_id: str) -> dict | None:
        with self._lock:
            self._ensure_open()
            self._cleanup(_now())
            parent = self._ensembles.get(ensemble_id)
            if parent is None:
                return None
            return {
                "ensemble_id": ensemble_id,
                "metadata": copy.deepcopy(parent.metadata),
                "children": list(parent.children),
                "results": copy.deepcopy(parent.results),
                "child_statuses": {
                    child: (
                        self._entries[child].record["status"]
                        if child in self._entries
                        else parent.results.get(child, {}).get("status", "")
                    )
                    for child in parent.children
                },
                "cancel_requested": parent.cancel_requested,
            }

    def cancel_ensemble(self, ensemble_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            parent = self._ensembles.get(ensemble_id)
            if parent is None:
                return False
            if len(parent.results) == len(parent.children):
                return parent.cancel_requested
            parent.cancel_requested = True
            for child in parent.children:
                self._cancel(child, now)
            return True

    def get_task_archive(self, task_id: str) -> dict | None:
        _identifier(task_id)
        with self._lock:
            self._ensure_open()
            archive = copy.deepcopy(self._archives.get(task_id))
            if archive is not None:
                events = self._journals[task_id]
                archive["last_event_id"] = events[-1]["event_id"] if events else None
                archive["checkpoints"] = copy.deepcopy(
                    list(self._checkpoints.get(task_id, {}).values())
                )
                archive.setdefault("selected_checkpoint", None)
                archive.setdefault("resume_count", 0)
                archive.setdefault("previous_outcomes", [])
            return archive

    def timeline(self, task_id: str, after_id: str | None = None, limit: int = 100) -> dict | None:
        _identifier(task_id)
        _timeline_cursor(after_id, limit)
        with self._lock:
            self._ensure_open()
            if task_id not in self._archives:
                return None
            cursor = tuple(map(int, (after_id or "0-0").split("-")))
            events = [
                event
                for event in self._journals[task_id]
                if tuple(map(int, event["event_id"].split("-"))) > cursor
            ]
            selected = events[:limit]
            return {
                "task_id": task_id,
                "events": copy.deepcopy(selected),
                "next_cursor": selected[-1]["event_id"] if selected else None,
                "has_more": len(events) > limit,
            }

    def register_analysis(
        self, task_id: str, attempt_id: str, analysis_id: str, record: dict, *, evidence: dict
    ) -> None:
        _analysis_record(task_id, attempt_id, analysis_id, record)
        _, locators = _evidence_records(task_id, attempt_id, analysis_id, evidence)
        item = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "analysis_id": analysis_id,
            "record": copy.deepcopy(record),
            "evidence": copy.deepcopy(evidence),
        }
        with self._lock:
            self._ensure_open()
            self._cleanup(_now())
            archive = self._archives.get(task_id)
            entry = self._entries.get(task_id)
            parent = archive["record"] if archive else entry.record if entry else None
            if (
                parent is None
                or parent.get("status") != "completed"
                or parent.get("attempt_id") != attempt_id
            ):
                raise ValueError("Analysis requires the committed completed parent attempt")
            previous = self._analyses.get(analysis_id)
            if previous is not None:
                if previous != item:
                    raise ValueError("Analysis identifier conflicts with an archived analysis")
                return
            if archive is not None and len(archive["analyses"]) >= MAX_ARCHIVED_ANALYSES:
                raise ArchiveCapacityError("The task analysis archive capacity is exhausted")
            self._check_artifacts(locators)
            now = _now()
            if archive is None:
                assert entry is not None
                archive = self._observe(task_id, entry, now)
            if entry is None:
                entry = _Entry({}, {}, archive["submitted_at"] or now, parent)
            self._analyses[analysis_id] = item
            archive["analyses"].append(copy.deepcopy(item))
            self._index_artifacts(locators)
            self._journal(task_id, entry, "analysis_completed", now, {"analysis_id": analysis_id})

    def get_analysis_archive(self, analysis_id: str) -> dict | None:
        _identifier(analysis_id)
        with self._lock:
            self._ensure_open()
            return copy.deepcopy(self._analyses.get(analysis_id))

    def get_artifact_archive(self, artifact_id: str) -> dict | None:
        if not isinstance(artifact_id, str) or not DIGEST.fullmatch(artifact_id):
            raise ValueError("Invalid artifact identifier")
        with self._lock:
            self._ensure_open()
            return copy.deepcopy(self._artifacts.get(artifact_id))

    def register_checkpoint(self, job: Job, descriptor: dict) -> bool:
        with self._lock:
            self._ensure_open()
            now = _now()
            entry = self._owned(job, now)
            if entry is None or entry.cancel_requested:
                return False
            archive = self._archives.get(job.task_id)
            original = archive["payload"] if archive is not None else entry.payload
            _, size = _checkpoint_descriptor(job, descriptor, original)
            checkpoints = self._checkpoints.get(job.task_id, {})
            checkpoint_id = descriptor["checkpoint_id"]
            previous = checkpoints.get(checkpoint_id)
            if previous is not None:
                if previous != descriptor:
                    raise CheckpointStoreError("CHECKPOINT_CONFLICT")
                return True
            if (
                checkpoints
                and descriptor["completed_steps"]
                <= next(reversed(checkpoints.values()))["completed_steps"]
            ):
                raise CheckpointStoreError("CHECKPOINT_STEP_CONFLICT")
            total_bytes = sum(
                item["manifest"]["bytes"] + item["state"]["bytes"] for item in checkpoints.values()
            )
            if (
                len(checkpoints) >= MAX_CHECKPOINTS_PER_TASK
                or total_bytes + size > MAX_CHECKPOINT_TOTAL_BYTES
            ):
                raise CheckpointStoreError("CHECKPOINT_LIMIT_EXCEEDED")
            self._observe(job.task_id, entry, now)
            self._checkpoints.setdefault(job.task_id, {})[checkpoint_id] = copy.deepcopy(descriptor)
            self._journal(
                job.task_id,
                entry,
                "checkpoint_created",
                now,
                {"checkpoint_id": checkpoint_id, "completed_steps": descriptor["completed_steps"]},
            )
            return True

    def get_checkpoint(self, task_id: str, checkpoint_id: str) -> dict | None:
        _checkpoint_identity(task_id, checkpoint_id)
        with self._lock:
            self._ensure_open()
            return copy.deepcopy(self._checkpoints.get(task_id, {}).get(checkpoint_id))

    def list_checkpoints(self, task_id: str) -> list[dict] | None:
        _identifier(task_id)
        with self._lock:
            self._ensure_open()
            if task_id not in self._archives:
                return None
            return copy.deepcopy(list(self._checkpoints.get(task_id, {}).values()))

    def get_latest_checkpoint(self, task_id: str) -> dict | None:
        checkpoints = self.list_checkpoints(task_id)
        return checkpoints[-1] if checkpoints else None

    def request_resume(self, task_id: str, checkpoint_id: str, carrier: dict) -> dict:
        _checkpoint_identity(task_id, checkpoint_id)
        _archive_json(carrier)
        with self._lock:
            self._ensure_open()
            now = _now()
            self._cleanup(now)
            archive = self._archives.get(task_id)
            if archive is None:
                raise CheckpointStoreError("TASK_NOT_FOUND")
            if archive["ensemble_id"] is not None:
                raise CheckpointStoreError("RESUME_ENSEMBLE_UNSUPPORTED")
            checkpoints = self._checkpoints.get(task_id, {})
            checkpoint = checkpoints.get(checkpoint_id)
            if checkpoint is None:
                raise CheckpointStoreError("CHECKPOINT_NOT_FOUND")
            current = self._entries.get(task_id)
            if current is not None and current.record["status"] in {"pending", "running"}:
                if current.manual_checkpoint_id != checkpoint_id:
                    raise CheckpointStoreError("RESUME_STATE_CONFLICT")
                return {
                    "task_id": task_id,
                    "status": current.record["status"],
                    "checkpoint_id": checkpoint_id,
                    "deduplicated": True,
                }
            if archive["record"]["status"] not in {"failed", "cancelled"}:
                raise CheckpointStoreError("RESUME_STATE_CONFLICT")
            if next(reversed(checkpoints)) != checkpoint_id:
                raise CheckpointStoreError("CHECKPOINT_NOT_LATEST")
            count = archive.get("resume_count", 0)
            if count >= MAX_MANUAL_RESUMES:
                raise CheckpointStoreError("RESUME_LIMIT_EXCEEDED")
            if self._active >= self._capacity:
                raise QueueFull("Pending and running task capacity has been reached")
            previous = copy.deepcopy(archive["record"])
            if current is None:
                current = _Entry({}, {}, now, {}, sequence=self._sse_sequences.get(task_id, 0))
                self._entries[task_id] = current
            current.payload = copy.deepcopy(archive["payload"])
            current.carrier = copy.deepcopy(carrier)
            current.submitted_at = now
            current.record = {"task_id": task_id, "status": "pending"}
            current.attempts = 0
            current.owner = None
            current.lease_until = 0.0
            current.expires_at = None
            current.cancel_requested = False
            current.completed_steps = checkpoint["completed_steps"]
            current.total_steps = checkpoint["total_steps"]
            current.manual_checkpoint_id = checkpoint_id
            archive.setdefault("previous_outcomes", []).append(previous)
            archive["resume_count"] = count + 1
            archive["selected_checkpoint"] = copy.deepcopy(checkpoint)
            self._active += 1
            self._pending.append(task_id)
            self._journal(
                task_id,
                current,
                "resume_requested",
                now,
                {
                    "checkpoint_id": checkpoint_id,
                    "parent_attempt_id": checkpoint["attempt_id"],
                    "completed_steps": checkpoint["completed_steps"],
                },
            )
            self._emit(task_id, current, "pending")
            return {
                "task_id": task_id,
                "status": "pending",
                "checkpoint_id": checkpoint_id,
                "deduplicated": False,
            }

    def health(self) -> bool:
        with self._lock:
            return not self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True


# All mutations execute atomically in standalone Redis, using server time.
# Raw payload/result JSON is never decoded and re-encoded by Lua: this preserves
# uint64 seeds that cannot be represented exactly by Lua's floating-point numbers.
_LUA = r"""
local prefix=ARGV[1]
local ttl=tonumber(ARGV[2])
local clock=redis.call('TIME')
local now=tonumber(clock[1])+tonumber(clock[2])/1000000
local pending=prefix..'pending'
local running=prefix..'running'
local active=prefix..'active'
local function taskkey(id) return prefix..'task:'..id end
local function streamkey(id) return prefix..'events:'..id end
local function parentkey(id) return prefix..'ensemble:'..id end
local function archivekey(id) return prefix..'archive:task:'..id end
local function journalkey(id) return prefix..'archive:timeline:'..id end
local function attemptskey(id) return prefix..'archive:attempts:'..id end
local function analyseskey(id) return prefix..'archive:analyses:'..id end
local function analysiskey(id) return prefix..'archive:analysis:'..id end
local function artifactkey(id) return prefix..'archive:artifact:'..id end
local function checkpointskey(id) return prefix..'archive:checkpoints:'..id end
local function checkpointkey(id,checkpoint) return prefix..'archive:checkpoint:'..id..':'..checkpoint end
local function outcomeskey(id) return prefix..'archive:previous-outcomes:'..id end
local function journal(id,event_type,details)
    local ak=archivekey(id)
    local state=redis.call('HGET',taskkey(id),'state') or redis.call('HGET',ak,'state')
    local data=cjson.encode({event_type=event_type,occurred_at=now,
        attempt_id=redis.call('HGET',ak,'last_attempt_id') or cjson.null,
        attempt=tonumber(redis.call('HGET',ak,'attempt_count') or '0'),
        status=state,details=details or {}})
    redis.call('XADD',journalkey(id),'*','data',data)
    redis.call('HSET',ak,'state',state)
end
local function observe(id,observed_owner)
    local ak=archivekey(id)
    if redis.call('EXISTS',ak)==1 then return end
    local tk=taskkey(id)
    local state=redis.call('HGET',tk,'state') or 'pending'
    local owner=redis.call('HGET',tk,'owner') or observed_owner
    local count=tonumber(redis.call('HGET',tk,'attempts') or '0')
    if owner and count<1 then count=1 end
    redis.call('HSET',ak,'payload',redis.call('HGET',tk,'payload') or '{}',
        'submitted_at',redis.call('HGET',tk,'submitted_at') or '',
        'ensemble_id',redis.call('HGET',tk,'ensemble_id') or '',
        'record',redis.call('HGET',tk,'record') or '{}','state',state,
        'coverage','partial','attempt_count',count)
    if owner then
        redis.call('HSET',ak,'last_attempt_id',owner)
        redis.call('RPUSH',attemptskey(id),cjson.encode({attempt_id=owner,attempt=count,
            claimed_at=cjson.null}))
        if state=='completed' then redis.call('HSET',ak,'committed_attempt_id',owner) end
    end
    journal(id,'history_observed',{coverage='partial'})
end
local function check_indexes(indexes)
    for _,item in ipairs(indexes) do
        local previous=redis.call('GET',artifactkey(item.artifact_id))
        if previous and previous~=item.raw then return false end
    end
    return true
end
local function index_artifacts(indexes)
    for _,item in ipairs(indexes) do
        redis.call('SET',artifactkey(item.artifact_id),item.raw)
    end
end
local function emit(id,phase,completed,total)
    local key=taskkey(id)
    if completed then redis.call('HSET',key,'completed_steps',completed) end
    if total then redis.call('HSET',key,'total_steps',total) end
    local data=cjson.encode({task_id=id,status=redis.call('HGET',key,'state'),phase=phase,
        attempt=tonumber(redis.call('HGET',key,'attempts') or '0'),
        completed_steps=tonumber(redis.call('HGET',key,'completed_steps') or '0'),
        total_steps=tonumber(redis.call('HGET',key,'total_steps') or '0')})
    local event=redis.call('XADD',streamkey(id),'MAXLEN',256,'*','data',data)
    redis.call('HSET',key,'progress',data,'last_event_id',event)
end
local function cancelled(id)
    return cjson.encode({task_id=id,status='cancelled',error='Simulation cancelled by request'})
end
local function complete(id,state,record,evidence,indexes)
    local key=taskkey(id)
    observe(id)
    local ak=archivekey(id)
    local owner=redis.call('HGET',key,'owner')
    redis.call('HSET',key,'state',state,'record',record)
    redis.call('HSET',ak,'state',state,'record',record)
    if state=='completed' and owner then redis.call('HSET',ak,'committed_attempt_id',owner) end
    if state=='completed' and evidence and evidence~='' then
        redis.call('HSET',ak,'evidence',evidence)
        index_artifacts(indexes or {})
    end
    journal(id,state)
    if state=='completed' then
        redis.call('HSET',key,'completed_steps',redis.call('HGET',key,'total_steps') or '0')
    end
    emit(id,state)
    redis.call('HDEL',key,'payload','carrier','owner','lease_until','cancel_requested')
    redis.call('SREM',active,id)
    redis.call('ZREM',running,id)
    redis.call('LREM',pending,0,id)
    local parent=redis.call('HGET',key,'ensemble_id')
    if parent and redis.call('EXISTS',parentkey(parent))==1 then
        local pk=parentkey(parent)
        if redis.call('HSETNX',pk,'result:'..id,record)==1 then
            redis.call('HINCRBY',pk,'finished',1)
        end
        if tonumber(redis.call('HGET',pk,'finished'))==tonumber(redis.call('HGET',pk,'count')) then
            redis.call('PEXPIRE',pk,ttl)
            local token=redis.call('HGET',pk,'idempotency')
            if token then redis.call('PEXPIRE',prefix..'idempotency:'..token,ttl) end
            local children=cjson.decode(redis.call('HGET',pk,'children'))
            for _,child in ipairs(children) do
                redis.call('PEXPIRE',taskkey(child),ttl)
                redis.call('PEXPIRE',streamkey(child),ttl)
            end
        end
    else
        redis.call('PEXPIRE',key,ttl)
        redis.call('PEXPIRE',streamkey(id),ttl)
    end
end
local function owned(id,owner)
    local key=taskkey(id)
    return redis.call('HGET',key,'state')=='running'
        and redis.call('HGET',key,'owner')==owner
        and tonumber(redis.call('HGET',key,'lease_until') or '0')>now
end
local function add(id,payload,carrier,total,parent)
    redis.call('HSET',taskkey(id),'payload',payload,'carrier',carrier,
        'submitted_at',tostring(now),'attempts',0,'state','pending','completed_steps',0,
        'total_steps',total,'record',cjson.encode({task_id=id,status='pending'}))
    if parent then redis.call('HSET',taskkey(id),'ensemble_id',parent) end
    redis.call('HSET',archivekey(id),'payload',payload,'submitted_at',tostring(now),
        'ensemble_id',parent or '','record',cjson.encode({task_id=id,status='pending'}),
        'state','pending','coverage','complete','attempt_count',0)
    redis.call('RPUSH',pending,id)
    redis.call('SADD',active,id)
    emit(id,'pending')
    journal(id,'submitted')
end
local function cancel(id)
    local key=taskkey(id)
    local state=redis.call('HGET',key,'state')
    if not state or state=='completed' or state=='failed' then return false end
    if state=='cancelled' or redis.call('HGET',key,'cancel_requested')=='1' then return true end
    observe(id)
    journal(id,'cancel_requested')
    if state=='pending' then complete(id,'cancelled',cancelled(id))
    else redis.call('HSET',key,'cancel_requested',1); emit(id,'cancel_requested') end
    return true
end
"""

_SUBMIT = (
    _LUA
    + r"""
local id=ARGV[3]
if redis.call('EXISTS',taskkey(id))==1 or redis.call('EXISTS',archivekey(id))==1 then return -2 end
if redis.call('SCARD',active)>=tonumber(ARGV[6]) then return -1 end
add(id,ARGV[4],ARGV[5],ARGV[7]); return 1
"""
)
_CLAIM = (
    _LUA
    + r"""
local expired=redis.call('ZRANGEBYSCORE',running,'-inf',now)
for _,id in ipairs(expired) do
    local key=taskkey(id)
    if redis.call('HGET',key,'state')=='running' then
        observe(id)
        if redis.call('HGET',key,'cancel_requested')=='1' then
            complete(id,'cancelled',cancelled(id))
        elseif tonumber(redis.call('HGET',key,'attempts') or '0')>=tonumber(ARGV[5]) then
            complete(id,'failed',cjson.encode({task_id=id,status='failed',
                error='Worker lease expired; maximum attempts exhausted'}))
        else
            redis.call('HSET',key,'state','pending','record',cjson.encode({task_id=id,status='pending'}))
            redis.call('HSET',archivekey(id),'record',cjson.encode({task_id=id,status='pending'}))
            journal(id,'retry',{reason='lease_expired'})
            redis.call('HDEL',key,'owner','lease_until')
            emit(id,'retry',0)
            redis.call('RPUSH',pending,id)
        end
    elseif redis.call('EXISTS',key)==0 then redis.call('SREM',active,id) end
    redis.call('ZREM',running,id)
end
while true do
    local id=redis.call('LPOP',pending)
    if not id then return nil end
    local key=taskkey(id)
    if redis.call('HGET',key,'state')=='pending' then
        observe(id)
        local checkpoint=redis.call('LINDEX',checkpointskey(id),-1)
        local checkpoint_raw=false
        if checkpoint then
            checkpoint_raw=redis.call('HGET',checkpointkey(id,checkpoint),'raw')
            if not checkpoint_raw then return redis.error_reply('CHECKPOINT_REGISTRY_INVALID') end
        end
        redis.call('HINCRBY',key,'attempts',1)
        redis.call('HSET',key,'state','running','owner',ARGV[3],
            'lease_until',tostring(now+tonumber(ARGV[4])),
            'record',cjson.encode({task_id=id,status='running'}))
        local ak=archivekey(id)
        local count=redis.call('HINCRBY',ak,'attempt_count',1)
        redis.call('HSET',ak,'last_attempt_id',ARGV[3],
            'record',cjson.encode({task_id=id,status='running'}))
        redis.call('RPUSH',attemptskey(id),cjson.encode({attempt_id=ARGV[3],attempt=count,
            claimed_at=now}))
        journal(id,'attempt_claimed')
        if checkpoint then
            local ck=checkpointkey(id,checkpoint)
            redis.call('HSET',ak,'selected_checkpoint_id',checkpoint)
            redis.call('HSET',key,'completed_steps',redis.call('HGET',ck,'completed_steps'),
                'total_steps',redis.call('HGET',ck,'total_steps'))
            journal(id,'attempt_resumed',{checkpoint_id=checkpoint,
                parent_attempt_id=redis.call('HGET',ck,'attempt_id'),
                completed_steps=tonumber(redis.call('HGET',ck,'completed_steps'))})
        end
        redis.call('ZADD',running,now+tonumber(ARGV[4]),id)
        emit(id,'running')
        return {id,redis.call('HGET',key,'payload'),redis.call('HGET',key,'carrier'),
            redis.call('HGET',key,'submitted_at'),ARGV[3],checkpoint_raw}
    elseif redis.call('EXISTS',key)==0 then redis.call('SREM',active,id) end
end
"""
)
_HEARTBEAT = (
    _LUA
    + r"""
if not owned(ARGV[3],ARGV[4]) then return 0 end
redis.call('HSET',taskkey(ARGV[3]),'lease_until',tostring(now+tonumber(ARGV[5])))
redis.call('ZADD',running,now+tonumber(ARGV[5]),ARGV[3]); return 1
"""
)
_FINISH = (
    _LUA
    + r"""
local id=ARGV[3]
if not owned(id,ARGV[4]) then return 0 end
if redis.call('HGET',taskkey(id),'cancel_requested')=='1' then
    complete(id,'cancelled',cancelled(id))
else
    local indexes=cjson.decode(ARGV[8])
    if ARGV[5]=='completed' and not check_indexes(indexes) then return -2 end
    complete(id,ARGV[5],ARGV[6],ARGV[7],indexes)
end
return 1
"""
)
_RELEASE = (
    _LUA
    + r"""
local id=ARGV[3]
if not owned(id,ARGV[4]) then return 0 end
local key=taskkey(id)
observe(id)
journal(id,'released',{reason='worker_shutdown'})
if redis.call('HGET',key,'cancel_requested')=='1' then
    complete(id,'cancelled',cancelled(id)); return 1
end
redis.call('HSET',key,'state','pending','record',cjson.encode({task_id=id,status='pending'}))
redis.call('HSET',archivekey(id),'record',cjson.encode({task_id=id,status='pending'}))
redis.call('HDEL',key,'owner','lease_until')
redis.call('HINCRBY',key,'attempts',-1)
redis.call('ZREM',running,id); redis.call('RPUSH',pending,id)
journal(id,'retry',{reason='worker_shutdown'})
emit(id,'retry',0); return 1
"""
)
_PROGRESS = (
    _LUA
    + r"""
local id=ARGV[3]
local key=taskkey(id)
if not owned(id,ARGV[4]) or redis.call('HGET',key,'cancel_requested')=='1' then return 0 end
local completed=tonumber(ARGV[5]); local total=tonumber(ARGV[6])
local previous=tonumber(redis.call('HGET',key,'completed_steps') or '0')
local expected=tonumber(redis.call('HGET',key,'total_steps') or '0')
if completed<previous or (expected~=0 and expected~=total) then return 0 end
emit(id,ARGV[7],completed,total); return 1
"""
)
_EVENTS = (
    _LUA
    + r"""
local key=taskkey(ARGV[3])
if redis.call('EXISTS',key)==0 then return nil end
-- Existing pre-progress records remain readable without rewriting their JSON.
if not redis.call('HGET',key,'progress') then
    local total=tonumber(redis.call('HGET',key,'total_steps') or '0')
    local raw=redis.call('HGET',key,'payload') or redis.call('HGET',key,'record')
    if total==0 and raw and #raw<=2097152 then
        local ok,decoded=pcall(cjson.decode,raw)
        if ok and type(decoded)=='table' and type(decoded.num_steps)=='number'
            and decoded.num_steps>=0 and decoded.num_steps<=10000 then
            total=math.floor(decoded.num_steps)
        end
    end
    local state=redis.call('HGET',key,'state') or 'pending'
    emit(ARGV[3],state,state=='completed' and total or 0,total)
    local remaining=redis.call('PTTL',key)
    if remaining>0 then redis.call('PEXPIRE',streamkey(ARGV[3]),remaining) end
end
local rows=redis.call('XRANGE',streamkey(ARGV[3]),'-','+')
local found=(ARGV[4]=='' or ARGV[4]=='0-0')
local start=1
if not found then
    for i,row in ipairs(rows) do
        if row[1]==ARGV[4] then found=true; start=i+1; break end
    end
end
local selected={}
if found then
    for i=start,math.min(#rows,start+tonumber(ARGV[5])-1) do
        selected[#selected+1]=rows[i]
    end
end
return {redis.call('HGET',key,'progress'),selected,found and 0 or 1,
        redis.call('HGET',key,'last_event_id')}
"""
)
_CANCEL = _LUA + "return cancel(ARGV[3]) and 1 or 0"
_IS_CANCEL = (
    _LUA
    + r"""
return (owned(ARGV[3],ARGV[4]) and redis.call('HGET',taskkey(ARGV[3]),'cancel_requested')=='1') and 1 or 0
"""
)
_SUBMIT_ENSEMBLE = (
    _LUA
    + r"""
local id=ARGV[3]; local token=ARGV[8]; local fingerprint=ARGV[9]
if token~='' then
    local idem=prefix..'idempotency:'..token
    local previous=redis.call('HGET',idem,'ensemble_id')
    if previous and redis.call('EXISTS',parentkey(previous))==1 then
        if redis.call('HGET',idem,'fingerprint')~=fingerprint then return {-3} end
        return {1,previous}
    end
end
local children=cjson.decode(ARGV[4])
if redis.call('EXISTS',parentkey(id))==1
    or redis.call('EXISTS',prefix..'archive:ensemble:'..id)==1 then return {-2} end
for _,child in ipairs(children) do
    if redis.call('EXISTS',taskkey(child.id))==1 or redis.call('EXISTS',archivekey(child.id))==1 then
        return {-2}
    end
end
if redis.call('SCARD',active)+#children>tonumber(ARGV[7]) then return {-1} end
local ids={}
for _,child in ipairs(children) do ids[#ids+1]=child.id end
redis.call('HSET',parentkey(id),'metadata',ARGV[6],'children',cjson.encode(ids),
    'count',#ids,'finished',0,'cancel_requested',0)
redis.call('HSET',prefix..'archive:ensemble:'..id,'metadata',ARGV[6],
    'children',cjson.encode(ids),'submitted_at',tostring(now))
if token~='' then
    redis.call('HSET',parentkey(id),'idempotency',token)
    redis.call('HSET',prefix..'idempotency:'..token,'ensemble_id',id,'fingerprint',fingerprint)
    redis.call('PERSIST',prefix..'idempotency:'..token)
end
for _,child in ipairs(children) do add(child.id,child.payload,ARGV[5],child.total,id) end
return {1,id}
"""
)
_GET_ENSEMBLE = (
    _LUA
    + r"""
local pk=parentkey(ARGV[3])
if redis.call('EXISTS',pk)==0 then return nil end
local children=cjson.decode(redis.call('HGET',pk,'children'))
local results={}; local statuses={}
for _,id in ipairs(children) do
    local result=redis.call('HGET',pk,'result:'..id)
    if result then results[#results+1]=id; results[#results+1]=result end
    local state=redis.call('HGET',taskkey(id),'state')
    if not state and result then state=cjson.decode(result).status end
    statuses[#statuses+1]=id; statuses[#statuses+1]=state or ''
end
return {redis.call('HGET',pk,'metadata'),redis.call('HGET',pk,'children'),results,statuses,
        redis.call('HGET',pk,'cancel_requested') or '0'}
"""
)
_CANCEL_ENSEMBLE = (
    _LUA
    + r"""
local pk=parentkey(ARGV[3])
if redis.call('EXISTS',pk)==0 then return 0 end
if redis.call('HGET',pk,'finished')==redis.call('HGET',pk,'count') then
    return tonumber(redis.call('HGET',pk,'cancel_requested') or '0')
end
redis.call('HSET',pk,'cancel_requested',1)
for _,id in ipairs(cjson.decode(redis.call('HGET',pk,'children'))) do cancel(id) end
return 1
"""
)

_GET_ARCHIVE = (
    _LUA
    + r"""
local id=ARGV[3]
if redis.call('EXISTS',archivekey(id))==0 then return nil end
local analyses={}
for _,analysis in ipairs(redis.call('LRANGE',analyseskey(id),0,-1)) do
    analyses[#analyses+1]=redis.call('HGETALL',analysiskey(analysis))
end
local latest=redis.call('XREVRANGE',journalkey(id),'+','-','COUNT',1)
local checkpoints={}
for _,checkpoint in ipairs(redis.call('LRANGE',checkpointskey(id),0,-1)) do
    checkpoints[#checkpoints+1]=redis.call('HGET',checkpointkey(id,checkpoint),'raw')
end
return {redis.call('HGETALL',archivekey(id)),redis.call('LRANGE',attemptskey(id),0,-1),analyses,
    #latest>0 and latest[1][1] or false,checkpoints,redis.call('LRANGE',outcomeskey(id),0,-1)}
"""
)
_TIMELINE = (
    _LUA
    + r"""
local id=ARGV[3]
if redis.call('EXISTS',archivekey(id))==0 then return nil end
if ARGV[4]=='18446744073709551615-18446744073709551615' then return {} end
local start=ARGV[4]=='' and '-' or '('..ARGV[4]
return redis.call('XRANGE',journalkey(id),start,'+','COUNT',tonumber(ARGV[5])+1)
"""
)
_REGISTER_ANALYSIS = (
    _LUA
    + r"""
local id=ARGV[3]; local owner=ARGV[4]; local analysis=ARGV[5]
local ak=archivekey(id); local tk=taskkey(id); local destination=analysiskey(analysis)
local parent=redis.call('HGET',ak,'record')
local state=redis.call('HGET',ak,'state')
if not parent then parent=redis.call('HGET',tk,'record'); state=redis.call('HGET',tk,'state') end
if not parent or parent~=ARGV[9] or state~='completed' then return -1 end
if redis.call('EXISTS',destination)==1 then
    if redis.call('HGET',destination,'task_id')~=id
        or redis.call('HGET',destination,'attempt_id')~=owner
        or redis.call('HGET',destination,'record')~=ARGV[6]
        or redis.call('HGET',destination,'evidence')~=ARGV[7] then return -2 end
    return 0
end
if redis.call('LLEN',analyseskey(id))>=tonumber(ARGV[10]) then return -4 end
local indexes=cjson.decode(ARGV[8])
if not check_indexes(indexes) then return -3 end
-- All preconditions and input decoding precede the first mutation.
observe(id,owner)
redis.call('HSET',destination,'task_id',id,'attempt_id',owner,'analysis_id',analysis,
    'record',ARGV[6],'evidence',ARGV[7])
redis.call('RPUSH',analyseskey(id),analysis)
index_artifacts(indexes)
journal(id,'analysis_completed',{analysis_id=analysis})
return 1
"""
)

_GET_CHECKPOINTS = (
    _LUA
    + r"""
local id=ARGV[3]
if redis.call('EXISTS',archivekey(id))==0 then return nil end
local checkpoints={}
for _,checkpoint in ipairs(redis.call('LRANGE',checkpointskey(id),0,-1)) do
    checkpoints[#checkpoints+1]=redis.call('HGET',checkpointkey(id,checkpoint),'raw')
end
return checkpoints
"""
)
_REGISTER_CHECKPOINT = (
    _LUA
    + r"""
local id=ARGV[3]; local checkpoint=ARGV[5]; local tk=taskkey(id); local ak=archivekey(id)
if not owned(id,ARGV[4]) or redis.call('HGET',tk,'cancel_requested')=='1' then return 0 end
local payload=redis.call('HGET',ak,'payload') or redis.call('HGET',tk,'payload')
if payload~=ARGV[10] then return -7 end
local ck=checkpointkey(id,checkpoint)
local previous=redis.call('HGET',ck,'raw')
if previous then
    if previous==ARGV[6] then return 1 else return -2 end
end
local latest=redis.call('LINDEX',checkpointskey(id),-1)
if latest and tonumber(redis.call('HGET',checkpointkey(id,latest),'completed_steps'))>=tonumber(ARGV[7]) then
    return -3
end
if redis.call('LLEN',checkpointskey(id))>=tonumber(ARGV[11])
    or tonumber(redis.call('HGET',ak,'checkpoint_bytes') or '0')+tonumber(ARGV[9])>tonumber(ARGV[12]) then
    return -4
end
observe(id)
redis.call('HSET',ck,'raw',ARGV[6],'completed_steps',ARGV[7],'total_steps',ARGV[8],
    'attempt_id',ARGV[4])
redis.call('RPUSH',checkpointskey(id),checkpoint)
redis.call('HINCRBY',ak,'checkpoint_bytes',ARGV[9])
journal(id,'checkpoint_created',{checkpoint_id=checkpoint,completed_steps=tonumber(ARGV[7])})
return 1
"""
)
_REQUEST_RESUME = (
    _LUA
    + r"""
local id=ARGV[3]; local checkpoint=ARGV[4]; local ak=archivekey(id); local tk=taskkey(id)
if redis.call('EXISTS',ak)==0 then return {-1} end
if (redis.call('HGET',ak,'ensemble_id') or '')~='' then return {-3} end
local ck=checkpointkey(id,checkpoint)
if not redis.call('HGET',ck,'raw') then return {-2} end
local current=redis.call('HGET',tk,'state')
if current=='pending' or current=='running' then
    if redis.call('HGET',tk,'manual_checkpoint_id')~=checkpoint then return {-4} end
    return {0,current}
end
local state=redis.call('HGET',ak,'state')
if state~='failed' and state~='cancelled' then return {-4} end
if redis.call('LINDEX',checkpointskey(id),-1)~=checkpoint then return {-5} end
if tonumber(redis.call('HGET',ak,'resume_count') or '0')>=tonumber(ARGV[7]) then return {-6} end
if redis.call('SCARD',active)>=tonumber(ARGV[6]) then return {-7} end
local previous=redis.call('HGET',ak,'record')
local payload=redis.call('HGET',ak,'payload')
if not previous or not payload then return {-4} end
-- No payload or terminal result is decoded/re-encoded by Lua.
redis.call('RPUSH',outcomeskey(id),previous)
redis.call('HINCRBY',ak,'resume_count',1)
redis.call('HSET',ak,'selected_checkpoint_id',checkpoint)
local record=cjson.encode({task_id=id,status='pending'})
redis.call('HSET',tk,'state','pending','record',record,'payload',payload,'carrier',ARGV[5],
    'submitted_at',tostring(now),'attempts',0,'manual_checkpoint_id',checkpoint,
    'completed_steps',redis.call('HGET',ck,'completed_steps'),
    'total_steps',redis.call('HGET',ck,'total_steps'))
redis.call('HDEL',tk,'owner','lease_until','cancel_requested')
redis.call('PERSIST',tk)
redis.call('PERSIST',streamkey(id))
redis.call('HSET',ak,'record',record)
redis.call('SADD',active,id)
redis.call('ZREM',running,id)
redis.call('LREM',pending,0,id)
redis.call('RPUSH',pending,id)
journal(id,'resume_requested',{checkpoint_id=checkpoint,
    parent_attempt_id=redis.call('HGET',ck,'attempt_id'),
    completed_steps=tonumber(redis.call('HGET',ck,'completed_steps'))})
emit(id,'pending')
return {1,'pending'}
"""
)


class RedisTaskStore:
    """Shared standalone Redis queue; namespaces isolate all state and replay."""

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
        self._capacity, self._ttl_ms = max_queue_size, max(1, int(ttl_seconds * 1000))
        self._lease, self._max_attempts = lease_seconds, max_attempts
        self._prefix = f"{namespace}:"
        self._task_prefix = f"{namespace}:task:"
        self._closed = False
        self._scripts = {
            name: self._redis.register_script(script)
            for name, script in {
                "submit": _SUBMIT,
                "claim": _CLAIM,
                "heartbeat": _HEARTBEAT,
                "finish": _FINISH,
                "release": _RELEASE,
                "progress": _PROGRESS,
                "events": _EVENTS,
                "cancel": _CANCEL,
                "is_cancel": _IS_CANCEL,
                "ensemble": _SUBMIT_ENSEMBLE,
                "get_ensemble": _GET_ENSEMBLE,
                "cancel_ensemble": _CANCEL_ENSEMBLE,
                "archive": _GET_ARCHIVE,
                "timeline": _TIMELINE,
                "register_analysis": _REGISTER_ANALYSIS,
                "get_checkpoints": _GET_CHECKPOINTS,
                "register_checkpoint": _REGISTER_CHECKPOINT,
                "request_resume": _REQUEST_RESUME,
            }.items()
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Task backend is closed")

    def _call(self, name: str, *args: Any) -> Any:
        self._ensure_open()
        return self._scripts[name](args=[self._prefix, self._ttl_ms, *args])

    def submit(self, task_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None:
        _identifier(task_id)
        payload_raw = _archive_json(payload)
        carrier_raw = _archive_json(carrier)
        code = self._call(
            "submit",
            task_id,
            payload_raw,
            carrier_raw,
            self._capacity,
            _total(payload),
        )
        if code == -1:
            raise QueueFull("Pending and running task capacity has been reached")
        if code == -2:
            raise ValueError(f"Task already exists: {task_id}")

    def get(self, task_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        raw = self._redis.hget(self._task_prefix + task_id, "record")
        return json.loads(raw) if raw is not None else None

    def claim(self) -> Job | None:
        row = self._call("claim", str(uuid4()), self._lease, self._max_attempts)
        return (
            Job(
                row[0],
                json.loads(row[1]),
                json.loads(row[2]),
                float(row[3]),
                row[4],
                json.loads(row[5]) if row[5] is not None else None,
            )
            if row
            else None
        )

    def heartbeat(self, job: Job) -> bool:
        return bool(self._call("heartbeat", job.task_id, job.owner, self._lease))

    def finish(self, job: Job, result: dict[str, Any], *, evidence: dict | None = None) -> bool:
        record = _terminal_result(job.task_id, result)
        record_raw = _archive_json(record)
        evidence_raw, locators = _evidence_records(job.task_id, job.owner, None, evidence)
        if evidence is not None and record.get("attempt_id") != job.owner:
            raise ValueError("Evidence must belong to the completed attempt")
        code = self._call(
            "finish",
            job.task_id,
            job.owner,
            record["status"],
            record_raw,
            evidence_raw,
            self._packed_indexes(locators),
        )
        if code == -2:
            raise ValueError("Artifact identifier conflicts with an archived occurrence")
        return bool(code)

    @staticmethod
    def _packed_indexes(locators: list[dict]) -> str:
        return json.dumps(
            [
                {"artifact_id": locator["artifact_id"], "raw": _archive_json(locator)}
                for locator in locators
            ],
            separators=(",", ":"),
        )

    def release(self, job: Job) -> bool:
        return bool(self._call("release", job.task_id, job.owner))

    def report_progress(self, job: Job, completed_steps: int, total_steps: int, phase: str) -> bool:
        _progress(completed_steps, total_steps, phase)
        return bool(
            self._call("progress", job.task_id, job.owner, completed_steps, total_steps, phase)
        )

    def events(self, task_id: str, after_id: str | None = None, limit: int = 100) -> dict | None:
        _cursor(after_id, limit)
        row = self._call("events", task_id, after_id or "", limit)
        if not row:
            return None
        return {
            "snapshot": json.loads(row[0]),
            "events": [{"event_id": event[0], "data": json.loads(event[1][1])} for event in row[1]],
            "reset": bool(row[2]),
            "last_event_id": row[3],
        }

    def request_cancel(self, task_id: str) -> bool:
        return bool(self._call("cancel", task_id))

    def is_cancel_requested(self, job: Job) -> bool:
        return bool(self._call("is_cancel", job.task_id, job.owner))

    def submit_ensemble(
        self,
        ensemble_id: str,
        children: list[tuple[str, dict]],
        carrier: dict,
        metadata: dict,
        idempotency_key: str | None = None,
    ) -> str:
        fingerprint, token = _ensemble_input(ensemble_id, children, metadata, idempotency_key)
        packed = [
            {"id": child, "payload": _archive_json(payload), "total": _total(payload)}
            for child, payload in children
        ]
        row = self._call(
            "ensemble",
            ensemble_id,
            json.dumps(packed),
            _archive_json(carrier),
            _archive_json(metadata),
            self._capacity,
            token or "",
            fingerprint,
        )
        if row[0] == -1:
            raise QueueFull("Insufficient capacity for the entire ensemble")
        if row[0] == -2:
            raise ValueError("An ensemble or child already exists")
        if row[0] == -3:
            raise ValueError("Idempotency key conflicts with the original request")
        return row[1]

    def get_ensemble(self, ensemble_id: str) -> dict | None:
        row = self._call("get_ensemble", ensemble_id)
        if not row:
            return None
        return {
            "ensemble_id": ensemble_id,
            "metadata": json.loads(row[0]),
            "children": json.loads(row[1]),
            "results": {row[2][i]: json.loads(row[2][i + 1]) for i in range(0, len(row[2]), 2)},
            "child_statuses": dict(zip(row[3][::2], row[3][1::2], strict=True)),
            "cancel_requested": row[4] == "1",
        }

    def cancel_ensemble(self, ensemble_id: str) -> bool:
        return bool(self._call("cancel_ensemble", ensemble_id))

    @staticmethod
    def _analysis_archive(values: dict[str, str]) -> dict:
        return {
            "task_id": values["task_id"],
            "attempt_id": values["attempt_id"],
            "analysis_id": values["analysis_id"],
            "record": json.loads(values["record"]),
            "evidence": json.loads(values["evidence"]),
        }

    def get_task_archive(self, task_id: str) -> dict | None:
        _identifier(task_id)
        row = self._call("archive", task_id)
        if not row:
            return None
        fields = dict(zip(row[0][::2], row[0][1::2], strict=True))
        checkpoints = [json.loads(item) for item in row[4]]
        return {
            "task_id": task_id,
            "payload": json.loads(fields["payload"]),
            "submitted_at": float(fields["submitted_at"]) if fields["submitted_at"] else None,
            "ensemble_id": fields["ensemble_id"] or None,
            "record": json.loads(fields["record"]),
            "evidence": json.loads(fields["evidence"]) if fields.get("evidence") else None,
            "attempts": [json.loads(item) for item in row[1]],
            "analyses": [
                self._analysis_archive(dict(zip(item[::2], item[1::2], strict=True)))
                for item in row[2]
            ],
            "coverage": fields["coverage"],
            "storage": "redis",
            "last_event_id": row[3] or None,
            "checkpoints": checkpoints,
            "selected_checkpoint": next(
                (
                    item
                    for item in checkpoints
                    if item["checkpoint_id"] == fields.get("selected_checkpoint_id")
                ),
                None,
            ),
            "resume_count": int(fields.get("resume_count", "0")),
            "previous_outcomes": [json.loads(item) for item in row[5]],
        }

    def timeline(self, task_id: str, after_id: str | None = None, limit: int = 100) -> dict | None:
        _identifier(task_id)
        _timeline_cursor(after_id, limit)
        rows = self._call("timeline", task_id, after_id or "", limit)
        if rows is None:
            return None
        selected = rows[:limit]
        return {
            "task_id": task_id,
            "events": [{"event_id": item[0], **json.loads(item[1][1])} for item in selected],
            "next_cursor": selected[-1][0] if selected else None,
            "has_more": len(rows) > limit,
        }

    def register_analysis(
        self, task_id: str, attempt_id: str, analysis_id: str, record: dict, *, evidence: dict
    ) -> None:
        record_raw = _analysis_record(task_id, attempt_id, analysis_id, record)
        evidence_raw, locators = _evidence_records(task_id, attempt_id, analysis_id, evidence)
        self._ensure_open()
        parent_raw = self._redis.hget(self._prefix + "archive:task:" + task_id, "record")
        if parent_raw is None:
            parent_raw = self._redis.hget(self._task_prefix + task_id, "record")
        parent = json.loads(parent_raw) if parent_raw is not None else None
        if (
            parent is None
            or parent.get("status") != "completed"
            or parent.get("attempt_id") != attempt_id
        ):
            raise ValueError("Analysis requires the committed completed parent attempt")
        code = self._call(
            "register_analysis",
            task_id,
            attempt_id,
            analysis_id,
            record_raw,
            evidence_raw,
            self._packed_indexes(locators),
            parent_raw,
            MAX_ARCHIVED_ANALYSES,
        )
        if code == -1:
            raise ValueError("Analysis requires the committed completed parent attempt")
        if code == -2:
            raise ValueError("Analysis identifier conflicts with an archived analysis")
        if code == -3:
            raise ValueError("Artifact identifier conflicts with an archived occurrence")
        if code == -4:
            raise ArchiveCapacityError("The task analysis archive capacity is exhausted")

    def get_analysis_archive(self, analysis_id: str) -> dict | None:
        _identifier(analysis_id)
        self._ensure_open()
        fields = self._redis.hgetall(self._prefix + "archive:analysis:" + analysis_id)
        return self._analysis_archive(fields) if fields else None

    def get_artifact_archive(self, artifact_id: str) -> dict | None:
        if not isinstance(artifact_id, str) or not DIGEST.fullmatch(artifact_id):
            raise ValueError("Invalid artifact identifier")
        self._ensure_open()
        raw = self._redis.get(self._prefix + "archive:artifact:" + artifact_id)
        return json.loads(raw) if raw is not None else None

    def register_checkpoint(self, job: Job, descriptor: dict) -> bool:
        self._ensure_open()
        original_raw = self._redis.hget(self._prefix + "archive:task:" + job.task_id, "payload")
        if original_raw is None:
            original_raw = self._redis.hget(self._task_prefix + job.task_id, "payload")
        if original_raw is None:
            return False
        raw, size = _checkpoint_descriptor(job, descriptor, json.loads(original_raw))
        code = self._call(
            "register_checkpoint",
            job.task_id,
            job.owner,
            descriptor["checkpoint_id"],
            raw,
            descriptor["completed_steps"],
            descriptor["total_steps"],
            size,
            original_raw,
            MAX_CHECKPOINTS_PER_TASK,
            MAX_CHECKPOINT_TOTAL_BYTES,
        )
        errors = {
            -2: "CHECKPOINT_CONFLICT",
            -3: "CHECKPOINT_STEP_CONFLICT",
            -4: "CHECKPOINT_LIMIT_EXCEEDED",
            -7: "CHECKPOINT_PARAMETERS_MISMATCH",
        }
        if code in errors:
            raise CheckpointStoreError(errors[code])
        return bool(code)

    def get_checkpoint(self, task_id: str, checkpoint_id: str) -> dict | None:
        _checkpoint_identity(task_id, checkpoint_id)
        self._ensure_open()
        raw = self._redis.hget(
            self._prefix + f"archive:checkpoint:{task_id}:{checkpoint_id}", "raw"
        )
        return json.loads(raw) if raw is not None else None

    def list_checkpoints(self, task_id: str) -> list[dict] | None:
        _identifier(task_id)
        rows = self._call("get_checkpoints", task_id)
        return [json.loads(item) for item in rows] if rows is not None else None

    def get_latest_checkpoint(self, task_id: str) -> dict | None:
        checkpoints = self.list_checkpoints(task_id)
        return checkpoints[-1] if checkpoints else None

    def request_resume(self, task_id: str, checkpoint_id: str, carrier: dict) -> dict:
        _checkpoint_identity(task_id, checkpoint_id)
        row = self._call(
            "request_resume",
            task_id,
            checkpoint_id,
            _archive_json(carrier),
            self._capacity,
            MAX_MANUAL_RESUMES,
        )
        errors = {
            -1: "TASK_NOT_FOUND",
            -2: "CHECKPOINT_NOT_FOUND",
            -3: "RESUME_ENSEMBLE_UNSUPPORTED",
            -4: "RESUME_STATE_CONFLICT",
            -5: "CHECKPOINT_NOT_LATEST",
            -6: "RESUME_LIMIT_EXCEEDED",
        }
        if row[0] in errors:
            raise CheckpointStoreError(errors[row[0]])
        if row[0] == -7:
            raise QueueFull("Pending and running task capacity has been reached")
        return {
            "task_id": task_id,
            "status": row[1],
            "checkpoint_id": checkpoint_id,
            "deduplicated": row[0] == 0,
        }

    def health(self) -> bool:
        return not self._closed and bool(self._redis.ping())

    def close(self) -> None:
        if not self._closed:
            self._redis.close()
            self._closed = True
