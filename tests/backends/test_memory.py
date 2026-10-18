"""Ownership, admission and retention guarantees for the local task queue."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from scientific_api.core import task_store
from scientific_api.core.task_store import MemoryTaskStore, QueueFull


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    value = [1_000.0]
    monkeypatch.setattr(task_store, "_now", lambda: value[0])
    return value


def test_status_lifecycle_keeps_transport_metadata_private() -> None:
    store = MemoryTaskStore()
    payload: dict[str, Any] = {"parameters": {"seed": 7}}
    carrier = {"traceparent": "private-trace-context"}
    store.submit("one", payload, carrier)
    payload["parameters"]["seed"] = 99
    carrier.clear()
    assert store.get("one") == {"task_id": "one", "status": "pending"}
    job = store.claim()
    assert job is not None
    assert job.payload == {"parameters": {"seed": 7}}
    assert job.carrier == {"traceparent": "private-trace-context"}
    assert store.get("one") == {"task_id": "one", "status": "running"}
    assert store.claim() is None
    assert not store.finish(replace(job, owner="another-worker"), {"status": "failed"})
    assert store.finish(job, {"status": "completed", "task_id": "wrong", "value": 42})
    record = store.get("one")
    assert record == {"task_id": "one", "status": "completed", "value": 42}
    assert record is not None
    record["value"] = 0
    assert store.get("one") == {"task_id": "one", "status": "completed", "value": 42}
    assert not store.finish(job, {"status": "failed"})
    assert not store.heartbeat(job)


def test_capacity_counts_pending_and_running() -> None:
    store = MemoryTaskStore(max_queue_size=1)
    store.submit("one", {}, {})
    with pytest.raises(QueueFull):
        store.submit("two", {}, {})
    job = store.claim()
    assert job is not None
    with pytest.raises(QueueFull):
        store.submit("two", {}, {})
    assert store.finish(job, {"status": "failed", "error": "expected"})
    store.submit("two", {}, {})
    with pytest.raises(ValueError, match="already exists"):
        store.submit("two", {}, {})


def test_admission_is_atomic_across_threads() -> None:
    store = MemoryTaskStore(max_queue_size=8)

    def submit(index: int) -> bool:
        try:
            store.submit(str(index), {}, {})
            return True
        except QueueFull:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(submit, range(64))) == 8
    claimed = []
    while (job := store.claim()) is not None:
        claimed.append(job.task_id)
    assert len(set(claimed)) == 8


def test_reclaim_preserves_job_and_rejects_obsolete_worker(clock: list[float]) -> None:
    store = MemoryTaskStore(lease_seconds=10)
    store.submit("one", {"seed": 8}, {"traceparent": "context"})
    old = store.claim()
    assert old is not None
    old.payload["seed"] = 55
    clock[0] += 11
    assert not store.heartbeat(old)
    assert not store.finish(old, {"status": "completed"})
    new = store.claim()
    assert new is not None
    assert new.task_id == old.task_id
    assert new.owner != old.owner
    assert new.submitted_at == old.submitted_at
    assert new.payload == {"seed": 8}
    assert new.carrier == {"traceparent": "context"}
    assert not store.release(old)
    assert store.finish(new, {"status": "completed"})


def test_heartbeat_extends_lease(clock: list[float]) -> None:
    store = MemoryTaskStore(lease_seconds=10)
    store.submit("one", {}, {})
    job = store.claim()
    assert job is not None
    clock[0] += 9
    assert store.heartbeat(job)
    clock[0] += 9
    assert store.claim() is None
    assert store.finish(job, {"status": "completed"})


def test_attempt_limit_fails_job_and_frees_capacity(clock: list[float]) -> None:
    store = MemoryTaskStore(max_queue_size=1, lease_seconds=1, max_attempts=2)
    store.submit("one", {}, {})
    assert store.claim() is not None
    clock[0] += 2
    assert store.claim() is not None
    clock[0] += 2
    assert store.claim() is None
    record = store.get("one")
    assert record is not None and record["status"] == "failed"
    assert "maximum attempts" in record["error"]
    store.submit("two", {}, {})


def test_only_terminal_results_expire(clock: list[float]) -> None:
    store = MemoryTaskStore(ttl_seconds=1, lease_seconds=100)
    store.submit("one", {}, {})
    clock[0] += 2
    assert store.get("one") == {"task_id": "one", "status": "pending"}
    job = store.claim()
    assert job is not None
    clock[0] += 2
    assert store.get("one") == {"task_id": "one", "status": "running"}
    assert store.finish(job, {"status": "completed"})
    clock[0] += 2
    assert store.get("one") is None


def test_local_release_marks_cancelled_and_instances_are_independent() -> None:
    store, another = MemoryTaskStore(), MemoryTaskStore()
    store.submit("one", {}, {})
    job = store.claim()
    assert job is not None
    assert store.release(job)
    assert store.get("one") == {
        "task_id": "one",
        "status": "cancelled",
        "error": "Simulation cancelled during shutdown",
    }
    assert another.get("one") is None
    assert store.health()
    store.close()
    store.close()
    assert not store.health()
    assert another.health()
    with pytest.raises(RuntimeError, match="closed"):
        store.get("one")


@pytest.mark.parametrize(
    "options",
    [
        {"max_queue_size": 0},
        {"max_attempts": 0},
        {"lease_seconds": 0},
        {"ttl_seconds": -1},
        {"ttl_seconds": float("inf")},
        {"lease_seconds": float("nan")},
    ],
)
def test_invalid_backend_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MemoryTaskStore(**options)
