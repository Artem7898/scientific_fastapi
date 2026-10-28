"""Replay, cancellation races and atomic ensemble retention on both backends."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest

from scientific_api.core.task_store import MemoryTaskStore, QueueFull, RedisTaskStore


@pytest.fixture(params=["memory", "redis"])
def backend_factory(request):
    stores = []
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    namespace = f"scientific-api-progress-test-{uuid4().hex}"
    redis = None
    if request.param == "redis":
        if not uri:
            pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
        redis = pytest.importorskip("redis")

    def make(**options):
        store = (
            MemoryTaskStore(**options)
            if request.param == "memory"
            else RedisTaskStore(uri, namespace=namespace, **options)
        )
        stores.append(store)
        return store

    yield make
    for store in stores:
        store.close()
    if redis is not None:
        client = redis.Redis.from_url(uri)
        try:
            keys = list(client.scan_iter(match=namespace + ":*"))
            if keys:
                client.delete(*keys)
        finally:
            client.close()


def children(prefix="child", count=2):
    return [
        (f"{prefix}{index}", {"num_steps": 400, "seed": 2**64 - 1 - index})
        for index in range(count)
    ]


def test_progress_replay_preserves_old_status_contract_and_private_payload(backend_factory):
    store = backend_factory()
    store.submit("one", {"num_steps": 10, "seed": 7}, {"traceparent": "private"})
    first = store.events("one")
    assert first["snapshot"] == {
        "task_id": "one",
        "status": "pending",
        "phase": "pending",
        "attempt": 0,
        "completed_steps": 0,
        "total_steps": 10,
    }
    job = store.claim()
    assert store.report_progress(job, 4, 10, "computing")
    assert not store.report_progress(job, 3, 10, "computing")
    assert not store.report_progress(job, 5, 11, "computing")
    replay = store.events("one", first["last_event_id"])
    assert [item["data"]["phase"] for item in replay["events"]] == ["running", "computing"]
    assert store.get("one") == {"task_id": "one", "status": "running"}
    assert "seed" not in json.dumps(replay) and "private" not in json.dumps(replay)
    assert store.finish(job, {"status": "completed", "answer": 42})
    terminal = store.events("one", replay["last_event_id"])
    assert terminal["snapshot"]["completed_steps"] == 10
    assert terminal["events"][-1]["data"]["status"] == "completed"
    assert store.events("one", terminal["last_event_id"])["events"] == []
    assert store.events("missing") is None


def test_progress_rejects_expired_or_obsolete_lease_and_announces_retry(backend_factory):
    store = backend_factory(lease_seconds=0.04)
    store.submit("one", {"num_steps": 10}, {})
    old = store.claim()
    assert store.report_progress(old, 7, 10, "computing")
    time.sleep(0.07)
    assert not store.report_progress(old, 8, 10, "computing")
    new = store.claim()
    assert new is not None and new.owner != old.owner
    assert not store.report_progress(old, 9, 10, "computing")
    assert store.events("one")["snapshot"]["completed_steps"] == 0
    assert store.events("one")["snapshot"]["attempt"] == 2
    assert "retry" in [item["data"]["phase"] for item in store.events("one")["events"]]
    assert not store.report_progress(replace(new, owner="wrong"), 3, 10, "computing")


def test_bounded_history_resets_trimmed_cursor_and_recovers_snapshot(backend_factory):
    store = backend_factory()
    store.submit("one", {"num_steps": 400}, {})
    cursor = store.events("one")["last_event_id"]
    job = store.claim()
    for completed in range(300):
        assert store.report_progress(job, completed, 400, "computing")
    all_events = store.events("one", limit=256)
    assert len(all_events["events"]) == 256
    reset = store.events("one", cursor)
    assert reset["reset"] and reset["events"] == []
    assert reset["snapshot"]["completed_steps"] == 299
    assert store.events("one", reset["last_event_id"])["events"] == []
    with pytest.raises(ValueError):
        store.events("one", "not-an-id")


def test_cancel_pending_frees_slot_without_execution(backend_factory):
    store = backend_factory(max_queue_size=1)
    store.submit("one", {}, {})
    assert store.request_cancel("one")
    assert store.request_cancel("one")
    assert store.get("one")["status"] == "cancelled"
    assert store.claim() is None
    store.submit("two", {}, {})
    job = store.claim()
    assert job.task_id == "two"
    assert store.finish(job, {"status": "completed"})
    assert not store.request_cancel("two")
    assert not store.request_cancel("missing")


def test_cancel_wins_finish_and_release_cannot_resurrect_it(backend_factory):
    store = backend_factory()
    store.submit("one", {"num_steps": 10}, {})
    job = store.claim()
    assert store.request_cancel("one")
    assert store.is_cancel_requested(job)
    assert store.get("one")["status"] == "running"
    assert store.events("one")["snapshot"]["phase"] == "cancel_requested"
    assert not store.report_progress(job, 5, 10, "computing")
    assert store.finish(job, {"status": "completed", "must_not_publish": True})
    assert store.get("one")["status"] == "cancelled"
    assert "must_not_publish" not in store.get("one")
    assert not store.release(job)
    assert store.claim() is None
    store.submit("two", {}, {})
    second = store.claim()
    assert store.request_cancel("two")
    assert store.release(second)
    assert store.get("two")["status"] == "cancelled"
    assert store.claim() is None


def test_cancel_requested_with_dead_worker_becomes_terminal_at_recovery(backend_factory):
    store = backend_factory(lease_seconds=0.04)
    store.submit("one", {}, {})
    job = store.claim()
    assert store.request_cancel("one")
    time.sleep(0.07)
    assert store.claim() is None
    assert store.get("one")["status"] == "cancelled"
    assert not store.finish(job, {"status": "completed"})


def test_ensemble_admission_is_atomic_and_counts_children_not_parent(backend_factory):
    store = backend_factory(max_queue_size=3)
    store.submit("existing", {}, {})
    with pytest.raises(QueueFull):
        store.submit_ensemble("too-big", children(count=3), {}, {"seeds": [1, 2, 3]})
    assert store.get_ensemble("too-big") is None
    assert store.get("child0") is None
    assert (
        store.submit_ensemble("parent", children(), {}, {"seeds": [2**64 - 1, 2**64 - 2]})
        == "parent"
    )
    parent = store.get_ensemble("parent")
    assert parent["children"] == ["child0", "child1"]
    assert parent["child_statuses"] == {"child0": "pending", "child1": "pending"}
    assert parent["metadata"]["seeds"][0] == 2**64 - 1
    with pytest.raises(QueueFull):
        store.submit("overflow", {}, {})
    claimed = [store.claim() for _ in range(3)]
    assert {job.task_id for job in claimed} == {"existing", "child0", "child1"}
    assert claimed[1].payload["seed"] == 2**64 - 1
    assert store.claim() is None


def test_ensemble_idempotency_reuses_original_children_and_rejects_conflict(backend_factory):
    store = backend_factory(max_queue_size=2)
    metadata = {"request_fingerprint": hashlib.sha256(b"request").hexdigest(), "seeds": [5, 6]}
    store.submit_ensemble("first", children(), {}, metadata, "request-key")
    changed_version = {**metadata, "seed_policy_version": 2}
    assert (
        store.submit_ensemble("second", children("different"), {}, changed_version, "request-key")
        == "first"
    )
    assert store.get_ensemble("second") is None
    assert store.get("different0") is None
    with pytest.raises(ValueError, match="conflicts"):
        store.submit_ensemble("third", children("third"), {}, {"seeds": [8, 9]}, "request-key")
    assert store.get_ensemble("first")["metadata"] == metadata


def test_concurrent_ensemble_admission_never_partially_accepts_children(backend_factory):
    store = backend_factory(max_queue_size=4)

    def submit(index):
        try:
            return store.submit_ensemble(f"parent{index}", children(f"c{index}_"), {}, {})
        except QueueFull:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = [item for item in pool.map(submit, range(12)) if item]
    assert len(admitted) == 2
    jobs = []
    while (job := store.claim()) is not None:
        jobs.append(job.task_id)
    assert len(jobs) == len(set(jobs)) == 4
    assert sum(store.get_ensemble(f"parent{i}") is not None for i in range(12)) == 2


def test_ensemble_keeps_early_contributions_and_children_until_all_terminal(backend_factory):
    store = backend_factory(ttl_seconds=0.06)
    store.submit_ensemble("parent", children(), {}, {"seeds": [3, 4]})
    first, second = store.claim(), store.claim()
    record = {"status": "completed", "seed": 2**64 - 1, "answer": 42}
    assert store.finish(first, record)
    assert not store.finish(first, {"status": "failed"})
    time.sleep(0.09)
    assert store.get(first.task_id)["answer"] == 42
    assert store.events(first.task_id)["snapshot"]["status"] == "completed"
    parent = store.get_ensemble("parent")
    assert parent["results"][first.task_id]["seed"] == 2**64 - 1
    assert parent["child_statuses"][second.task_id] == "running"
    assert store.finish(second, {"status": "failed", "error": "expected"})
    assert len(store.get_ensemble("parent")["results"]) == 2
    time.sleep(0.09)
    assert store.get_ensemble("parent") is None
    assert store.get(first.task_id) is None
    assert store.events(first.task_id) is None


def test_ensemble_lease_exhaustion_is_a_durable_terminal_contribution(backend_factory):
    store = backend_factory(lease_seconds=0.04, max_attempts=1)
    store.submit_ensemble("parent", children(count=1), {}, {})
    old = store.claim()
    time.sleep(0.07)
    assert store.claim() is None
    snapshot = store.get_ensemble("parent")
    assert snapshot["results"][old.task_id]["status"] == "failed"
    assert not store.finish(old, {"status": "completed"})
    assert len(snapshot["results"]) == 1


def test_ensemble_cancel_preserves_completed_children_and_finishes_rest(backend_factory):
    store = backend_factory(max_queue_size=3)
    store.submit_ensemble("parent", children(count=3), {}, {})
    completed, running = store.claim(), store.claim()
    assert store.finish(completed, {"status": "completed", "answer": 1})
    assert store.cancel_ensemble("parent")
    parent = store.get_ensemble("parent")
    assert parent["cancel_requested"]
    assert parent["results"][completed.task_id]["answer"] == 1
    assert parent["child_statuses"]["child2"] == "cancelled"
    assert store.is_cancel_requested(running)
    assert store.finish(running, {"status": "completed"})
    parent = store.get_ensemble("parent")
    assert len(parent["results"]) == 3
    assert parent["results"][running.task_id]["status"] == "cancelled"
    assert store.cancel_ensemble("parent")
    assert store.claim() is None
    assert not store.cancel_ensemble("missing")


def test_legacy_redis_terminal_record_gets_replay_without_changing_raw_seed(backend_factory):
    store = backend_factory(ttl_seconds=0.1)
    if not isinstance(store, RedisTaskStore):
        pytest.skip("Legacy persistence concerns Redis only")
    key = store._task_prefix + "legacy"
    record = {"task_id": "legacy", "status": "completed", "seed": 2**64 - 1, "num_steps": 23}
    raw = json.dumps(record)
    store._redis.hset(key, mapping={"state": "completed", "record": raw})
    store._redis.pexpire(key, 150)
    assert store.get("legacy") == record
    event = store.events("legacy")
    assert event["snapshot"]["status"] == "completed"
    assert event["snapshot"]["completed_steps"] == 23
    assert store._redis.hget(key, "record") == raw
    assert 0 < store._redis.pttl(store._prefix + "events:legacy") <= 150
    assert store.events("legacy")["last_event_id"] == event["last_event_id"]


def test_parent_contribution_survives_missing_child_record(backend_factory):
    store = backend_factory()
    store.submit_ensemble("parent", children(), {}, {})
    first, second = store.claim(), store.claim()
    assert store.finish(first, {"status": "completed", "answer": 17})
    if isinstance(store, MemoryTaskStore):
        del store._entries[first.task_id]
    else:
        store._redis.delete(store._task_prefix + first.task_id)
    snapshot = store.get_ensemble("parent")
    assert snapshot["results"][first.task_id]["answer"] == 17
    assert snapshot["child_statuses"][first.task_id] == "completed"
    assert store.finish(second, {"status": "completed"})
    assert len(store.get_ensemble("parent")["results"]) == 2


def test_concurrent_idempotent_ensemble_retries_reserve_capacity_once(backend_factory):
    store = backend_factory(max_queue_size=2)
    metadata = {"request_fingerprint": hashlib.sha256(b"same-body").hexdigest()}

    def retry(index):
        return store.submit_ensemble(
            f"parent{index}", children(f"c{index}_"), {}, metadata, "same-key"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        parents = list(pool.map(retry, range(16)))
    assert len(set(parents)) == 1
    parent = store.get_ensemble(parents[0])
    jobs = [store.claim(), store.claim()]
    assert [job.task_id for job in jobs] == parent["children"]
    assert store.claim() is None
    assert sum(store.get_ensemble(f"parent{i}") is not None for i in range(16)) == 1
