"""Lifecycle archives outlive status TTL and only index committed evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scientific_api.core import task_store
from scientific_api.core.task_store import (
    ArchiveCapacityError,
    MemoryTaskStore,
    QueueFull,
    RedisTaskStore,
)


@pytest.fixture(params=["memory", "redis"])
def backend(request, monkeypatch):
    stores = []
    namespace = "scientific-api-journal-test-" + uuid4().hex
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    client = None
    clock = [1000.0]
    if request.param == "memory":
        monkeypatch.setattr(task_store, "_now", lambda: clock[0])
    else:
        if not uri:
            pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
        redis = pytest.importorskip("redis")
        client = redis.Redis.from_url(uri, decode_responses=True)
        client.ping()

    def make(**options):
        store = (
            MemoryTaskStore(**options)
            if request.param == "memory"
            else RedisTaskStore(uri, namespace=namespace, **options)
        )
        stores.append(store)
        return store

    def advance(seconds):
        if request.param == "memory":
            clock[0] += seconds
        else:
            time.sleep(seconds)

    yield SimpleNamespace(
        make=make, advance=advance, kind=request.param, client=client, namespace=namespace
    )
    for store in stores:
        store.close()
    if client is not None:
        keys = list(client.scan_iter(match=namespace + ":*"))
        if keys:
            client.delete(*keys)
        client.close()


def evidence(task_id, attempt_id, analysis_id=None):
    records = []
    for name, filename in (
        ("manifest", "manifest.json"),
        ("analysis", "analysis.json") if analysis_id else ("final_state", "final_state.npz"),
    ):
        identity = {
            "schema_version": 1,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "analysis_id": analysis_id,
            "name": name,
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        prefix = f"{task_id}/{attempt_id}"
        if analysis_id:
            prefix += f"/analyses/{analysis_id}"
        records.append(
            {
                "artifact_id": hashlib.sha256(raw).hexdigest(),
                "name": name,
                "path": f"{prefix}/{filename}",
                "sha256": hashlib.sha256(filename.encode()).hexdigest(),
                "bytes": 12,
                "media_type": "application/json"
                if filename.endswith(".json")
                else "application/octet-stream",
            }
        )
    return {
        "schema_version": 1,
        "model_version": "ideal-gas-ou-v1",
        "model_contract_sha256": "a" * 64,
        "manifest_sha256": records[0]["sha256"],
        "manifest": records[0],
        "artifacts": records,
    }


def result(job, status="completed"):
    return {
        "task_id": job.task_id,
        "attempt_id": job.owner,
        "status": status,
        "seed": 2**64 - 1,
    }


def analysis_record(job, analysis_id):
    return {
        **result(job),
        "analysis_id": analysis_id,
        "analysis_version": "velocity-distribution-v1",
    }


def event_types(store, task_id):
    return [item["event_type"] for item in store.timeline(task_id, limit=256)["events"]]


def test_journal_survives_status_ttl_and_keeps_exact_original_payload(backend):
    store = backend.make(ttl_seconds=0.03)
    payload = {"seed": 2**64 - 1, "num_steps": 10, "nested": {"model": "original"}}
    store.submit("task", payload, {"traceparent": "private-carrier"})
    payload["nested"]["model"] = "changed"
    job = store.claim()
    saved_evidence = evidence("task", job.owner)
    accepted = result(job)
    assert store.finish(job, accepted, evidence=saved_evidence)
    saved_evidence["artifacts"][1]["bytes"] = 999
    backend.advance(0.06)
    assert store.get("task") is None
    assert store.events("task") is None
    archive = store.get_task_archive("task")
    assert archive["record"] == accepted
    assert archive["payload"] == {
        "seed": 2**64 - 1,
        "num_steps": 10,
        "nested": {"model": "original"},
    }
    assert archive["coverage"] == "complete"
    assert archive["storage"] == backend.kind
    assert "private-carrier" not in json.dumps(archive)
    assert archive["evidence"]["artifacts"][1]["bytes"] == 12
    for descriptor in archive["evidence"]["artifacts"]:
        assert store.get_artifact_archive(descriptor["artifact_id"]) == {
            "artifact_id": descriptor["artifact_id"],
            "task_id": "task",
            "attempt_id": job.owner,
            "analysis_id": None,
            "descriptor": descriptor,
        }
    assert event_types(store, "task") == ["submitted", "attempt_claimed", "completed"]
    archive["payload"]["seed"] = 0
    assert store.get_task_archive("task")["payload"]["seed"] == 2**64 - 1
    with pytest.raises(ValueError, match="already exists"):
        store.submit("task", {}, {})


def test_lifecycle_has_no_progress_heartbeat_or_rejected_owner_events(backend):
    store = backend.make()
    store.submit("task", {"num_steps": 400}, {})
    job = store.claim()
    for step in range(300):
        assert store.report_progress(job, step, 400, "computing")
    assert store.heartbeat(job)
    assert len(store.events("task", limit=256)["events"]) == 256
    assert event_types(store, "task") == ["submitted", "attempt_claimed"]
    assert store.finish(job, result(job, "failed"))
    assert not store.finish(job, result(job))
    assert not store.heartbeat(job)
    assert not store.release(job)
    assert event_types(store, "task") == ["submitted", "attempt_claimed", "failed"]


def test_cancel_wins_evidence_commit_and_journal_uses_actual_outcome(backend):
    store = backend.make()
    store.submit("task", {}, {})
    job = store.claim()
    details = evidence("task", job.owner)
    assert store.request_cancel("task")
    assert store.request_cancel("task")
    assert store.finish(job, result(job), evidence=details)
    archive = store.get_task_archive("task")
    assert archive["record"]["status"] == "cancelled"
    assert archive["evidence"] is None
    assert event_types(store, "task") == [
        "submitted",
        "attempt_claimed",
        "cancel_requested",
        "cancelled",
    ]
    assert all(
        store.get_artifact_archive(item["artifact_id"]) is None for item in details["artifacts"]
    )


def test_pending_cancel_is_recorded_once_without_a_fabricated_attempt(backend):
    store = backend.make()
    store.submit("task", {}, {})
    assert store.request_cancel("task")
    assert store.request_cancel("task")
    assert store.get_task_archive("task")["attempts"] == []
    events = store.timeline("task")["events"]
    assert [item["event_type"] for item in events] == ["submitted", "cancel_requested", "cancelled"]
    assert all(item["attempt_id"] is None and item["attempt"] == 0 for item in events)


def test_retries_keep_both_attempts_but_only_index_the_winner(backend):
    store = backend.make(lease_seconds=0.03)
    store.submit("task", {"seed": 2**64 - 1}, {})
    old = store.claim()
    backend.advance(0.06)
    new = store.claim()
    assert old.owner != new.owner
    assert not store.finish(old, result(old), evidence=evidence("task", old.owner))
    assert store.finish(new, result(new), evidence=evidence("task", new.owner))
    archive = store.get_task_archive("task")
    assert [(item["attempt_id"], item["attempt"]) for item in archive["attempts"]] == [
        (old.owner, 1),
        (new.owner, 2),
    ]
    assert archive["record"]["attempt_id"] == new.owner
    assert event_types(store, "task") == [
        "submitted",
        "attempt_claimed",
        "retry",
        "attempt_claimed",
        "completed",
    ]
    assert (
        store.get_artifact_archive(evidence("task", old.owner)["manifest"]["artifact_id"]) is None
    )


def test_lease_exhaustion_and_dead_worker_cancel_have_terminal_history(backend):
    store = backend.make(lease_seconds=0.03, max_attempts=1)
    store.submit("failed", {}, {})
    assert store.claim() is not None
    backend.advance(0.06)
    assert store.claim() is None
    assert event_types(store, "failed") == ["submitted", "attempt_claimed", "failed"]
    assert store.get_task_archive("failed")["record"]["status"] == "failed"
    store.submit("cancelled", {}, {})
    job = store.claim()
    assert store.request_cancel("cancelled")
    backend.advance(0.06)
    assert store.claim() is None
    assert not store.finish(job, result(job))
    assert event_types(store, "cancelled")[-1] == "cancelled"


def test_release_is_a_lifecycle_event_and_does_not_reuse_archive_attempt_numbers(backend):
    store = backend.make(max_attempts=1)
    store.submit("task", {}, {})
    old = store.claim()
    assert store.release(old)
    assert "released" in event_types(store, "task")
    if backend.kind == "redis":
        new = store.claim()
        assert new.owner != old.owner
        assert store.finish(new, result(new))
        assert [item["attempt"] for item in store.get_task_archive("task")["attempts"]] == [1, 2]
        assert event_types(store, "task") == [
            "submitted",
            "attempt_claimed",
            "released",
            "retry",
            "attempt_claimed",
            "completed",
        ]
    else:
        assert store.get_task_archive("task")["record"]["status"] == "cancelled"


def test_timeline_pagination_and_unknown_numeric_cursor(backend):
    store = backend.make()
    assert store.timeline("unknown") is None
    store.submit("task", {}, {})
    job = store.claim()
    assert store.finish(job, result(job))
    cursor = None
    events = []
    while True:
        page = store.timeline("task", cursor, limit=1)
        events.extend(page["events"])
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
    assert [item["event_type"] for item in events] == ["submitted", "attempt_claimed", "completed"]
    assert store.timeline("task", cursor)["events"] == []
    assert store.timeline("task", "18446744073709551615-18446744073709551615")["events"] == []
    assert len({item["event_id"] for item in events}) == 3
    assert all(isinstance(item["occurred_at"], (int, float)) for item in events)


@pytest.mark.parametrize(
    "cursor,limit",
    [
        ("bad", 1),
        ("١-١", 1),
        ("18446744073709551616-0", 1),
        ("-1-0", 1),
        ("1-0", 0),
        (None, 257),
        (None, True),
    ],
)
def test_invalid_timeline_arguments_do_not_mutate_history(backend, cursor, limit):
    store = backend.make()
    store.submit("task", {}, {})
    before = store.timeline("task")
    with pytest.raises(ValueError):
        store.timeline("task", cursor, limit)
    assert store.timeline("task") == before


def test_analysis_registration_is_idempotent_after_parent_status_expiry(backend):
    store = backend.make(ttl_seconds=0.03)
    store.submit("task", {}, {})
    job = store.claim()
    assert store.finish(job, result(job), evidence=evidence("task", job.owner))
    backend.advance(0.06)
    assert store.get("task") is None
    analysis_id = uuid4().hex
    record = analysis_record(job, analysis_id)
    saved = evidence("task", job.owner, analysis_id)
    for _ in range(2):
        store.register_analysis("task", job.owner, analysis_id, record, evidence=saved)
    archive = store.get_analysis_archive(analysis_id)
    assert archive == {
        "task_id": "task",
        "attempt_id": job.owner,
        "analysis_id": analysis_id,
        "record": record,
        "evidence": saved,
    }
    assert store.get_task_archive("task")["analyses"] == [archive]
    assert event_types(store, "task").count("analysis_completed") == 1
    for item in saved["artifacts"]:
        assert store.get_artifact_archive(item["artifact_id"])["analysis_id"] == analysis_id
    changed = {**record, "extra": "changed"}
    before = store.timeline("task")
    with pytest.raises(ValueError, match="conflicts"):
        store.register_analysis("task", job.owner, analysis_id, changed, evidence=saved)
    assert store.timeline("task") == before


def test_analysis_rejects_foreign_parent_and_index_conflicts_before_any_mutation(backend):
    store = backend.make()
    store.submit("task", {}, {})
    job = store.claim()
    assert store.finish(job, result(job))
    original_id = uuid4().hex
    original_evidence = evidence("task", job.owner, original_id)
    store.register_analysis(
        "task",
        job.owner,
        original_id,
        analysis_record(job, original_id),
        evidence=original_evidence,
    )
    new_id = uuid4().hex
    conflicting = evidence("task", job.owner, new_id)
    conflicting["artifacts"][1]["artifact_id"] = original_evidence["artifacts"][1]["artifact_id"]
    before = store.timeline("task")
    with pytest.raises(ValueError, match="conflicts"):
        store.register_analysis(
            "task", job.owner, new_id, analysis_record(job, new_id), evidence=conflicting
        )
    assert store.get_analysis_archive(new_id) is None
    assert store.timeline("task") == before
    with pytest.raises(ValueError):
        store.register_analysis(
            "missing", job.owner, new_id, analysis_record(job, new_id), evidence=conflicting
        )
    assert store.get_task_archive("missing") is None


def test_concurrent_analysis_retry_creates_one_record_and_one_lifecycle_event(backend):
    store = backend.make()
    store.submit("task", {}, {})
    job = store.claim()
    assert store.finish(job, result(job))
    analysis_id = uuid4().hex
    record = analysis_record(job, analysis_id)
    saved = evidence("task", job.owner, analysis_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: store.register_analysis(
                    "task", job.owner, analysis_id, record, evidence=saved
                ),
                range(8),
            )
        )
    assert len(store.get_task_archive("task")["analyses"]) == 1
    assert event_types(store, "task").count("analysis_completed") == 1


def test_invalid_evidence_and_admission_do_not_append_history(backend):
    store = backend.make(max_queue_size=1)
    store.submit("task", {}, {})
    with pytest.raises(QueueFull):
        store.submit("overflow", {}, {})
    with pytest.raises(ValueError):
        store.submit_ensemble("bad", [("bad-child", {"seed": float("nan")})], {}, {})
    assert store.get_task_archive("overflow") is None
    assert store.get_task_archive("bad-child") is None
    job = store.claim()
    before = store.timeline("task")
    invalid = evidence("task", job.owner)
    invalid["artifacts"][1]["path"] = "outside/state.npz"
    with pytest.raises(ValueError):
        store.finish(job, result(job), evidence=invalid)
    assert store.timeline("task") == before
    assert store.get("task")["status"] == "running"
    assert store.finish(job, result(job))


def test_ensemble_admission_journals_each_child_once_and_preserves_uint64(backend):
    store = backend.make(ttl_seconds=0.03)
    children = [("child-0", {"seed": 2**64 - 1}), ("child-1", {"seed": 2**64 - 2})]
    metadata = {
        "seeds": [2**64 - 1, 2**64 - 2],
        "request_fingerprint": hashlib.sha256(b"same-request").hexdigest(),
    }
    assert store.submit_ensemble("parent", children, {}, metadata, "same-request") == "parent"
    assert (
        store.submit_ensemble("other", [("unused", {})], {}, metadata, "same-request") == "parent"
    )
    assert store.get_task_archive("unused") is None
    for name, payload in children:
        archive = store.get_task_archive(name)
        assert archive["payload"] == payload
        assert archive["ensemble_id"] == "parent"
        assert event_types(store, name) == ["submitted"]
    for _ in children:
        job = store.claim()
        assert store.finish(job, result(job))
    backend.advance(0.06)
    assert store.get_ensemble("parent") is None
    with pytest.raises(ValueError, match="already exists"):
        store.submit_ensemble("parent", [("new-child", {})], {}, {})
    with pytest.raises(ValueError, match="already exists"):
        store.submit_ensemble("new-parent", children, {}, {})
    assert store.get_task_archive("new-child") is None


def test_redis_archive_and_indexes_survive_a_new_client(backend):
    if backend.kind != "redis":
        pytest.skip("Memory provenance is explicitly process-local and volatile")
    first = backend.make(ttl_seconds=0.03)
    first.submit("task", {"seed": 2**64 - 1}, {})
    job = first.claim()
    saved = evidence("task", job.owner)
    assert first.finish(job, result(job), evidence=saved)
    before = first.get_task_archive("task")
    first.close()
    backend.advance(0.06)
    second = backend.make()
    assert second.get("task") is None
    assert second.get_task_archive("task") == before
    assert second.get_artifact_archive(saved["manifest"]["artifact_id"]) is not None
    assert event_types(second, "task")[-1] == "completed"


def test_legacy_live_completed_parent_is_observed_without_fabricated_past(backend):
    store = backend.make()
    task_id, owner, analysis_id = "legacy", "legacy-owner", uuid4().hex
    record = {"task_id": task_id, "attempt_id": owner, "status": "completed", "seed": 2**64 - 1}
    if backend.kind == "redis":
        backend.client.hset(
            backend.namespace + ":task:" + task_id,
            mapping={
                "state": "completed",
                "record": json.dumps(record),
                "submitted_at": "10.5",
                "attempts": "1",
            },
        )
    else:
        store._entries[task_id] = task_store._Entry({}, {}, 10.5, record, attempts=1)
    saved_analysis = {
        **record,
        "analysis_id": analysis_id,
        "analysis_version": "velocity-distribution-v1",
    }
    store.register_analysis(
        task_id, owner, analysis_id, saved_analysis, evidence=evidence(task_id, owner, analysis_id)
    )
    archive = store.get_task_archive(task_id)
    assert archive["coverage"] == "partial"
    assert archive["payload"] == {}
    assert archive["submitted_at"] == 10.5
    assert archive["attempts"] == [{"attempt_id": owner, "attempt": 1, "claimed_at": None}]
    assert event_types(store, task_id) == ["history_observed", "analysis_completed"]


def test_legacy_running_retry_records_only_observed_history(backend):
    store = backend.make()
    task_id, owner = "legacy-running", "legacy-owner"
    payload = {"seed": 2**64 - 1}
    record = {"task_id": task_id, "status": "running"}
    if backend.kind == "redis":
        prefix = backend.namespace + ":"
        backend.client.hset(
            prefix + "task:" + task_id,
            mapping={
                "state": "running",
                "record": json.dumps(record),
                "payload": json.dumps(payload),
                "carrier": "{}",
                "owner": owner,
                "lease_until": "0",
                "submitted_at": "10.5",
                "attempts": "1",
            },
        )
        backend.client.zadd(prefix + "running", {task_id: 0})
        backend.client.sadd(prefix + "active", task_id)
    else:
        store._entries[task_id] = task_store._Entry(
            payload, {}, 10.5, record, attempts=1, owner=owner, lease_until=0
        )
        store._active = 1
    job = store.claim()
    assert job.task_id == task_id and job.owner != owner
    assert job.payload == payload
    archive = store.get_task_archive(task_id)
    assert archive["coverage"] == "partial"
    assert archive["attempts"][0]["claimed_at"] is None
    assert [item["attempt"] for item in archive["attempts"]] == [1, 2]
    assert event_types(store, task_id) == ["history_observed", "retry", "attempt_claimed"]


def test_memory_archive_does_not_persist_into_an_unrelated_store():
    first, second = MemoryTaskStore(), MemoryTaskStore()
    first.submit("task", {}, {})
    assert first.get_task_archive("task") is not None
    assert second.get_task_archive("task") is None
    first.close()
    second.close()


def test_archive_snapshot_records_its_atomic_timeline_cutoff(backend):
    store = backend.make()
    store.submit("task", {}, {})
    snapshot = store.get_task_archive("task")
    assert snapshot["record"]["status"] == "pending"
    assert snapshot["last_event_id"] == store.timeline("task")["events"][-1]["event_id"]
    job = store.claim()
    assert store.finish(job, result(job))
    latest = store.get_task_archive("task")
    assert latest["record"]["status"] == "completed"
    assert latest["last_event_id"] != snapshot["last_event_id"]
    cutoff = tuple(map(int, snapshot["last_event_id"].split("-")))
    old_events = [
        item
        for item in store.timeline("task")["events"]
        if tuple(map(int, item["event_id"].split("-"))) <= cutoff
    ]
    assert [item["event_type"] for item in old_events] == ["submitted"]


def test_analysis_capacity_is_atomic_and_does_not_block_idempotent_retry(backend, monkeypatch):
    monkeypatch.setattr(task_store, "MAX_ARCHIVED_ANALYSES", 2)
    store = backend.make()
    store.submit("task", {}, {})
    job = store.claim()
    assert store.finish(job, result(job))
    registrations = []
    for _ in range(2):
        analysis_id = uuid4().hex
        record = analysis_record(job, analysis_id)
        saved = evidence("task", job.owner, analysis_id)
        store.register_analysis("task", job.owner, analysis_id, record, evidence=saved)
        registrations.append((analysis_id, record, saved))
    before = store.get_task_archive("task")
    history = store.timeline("task")
    analysis_id, record, saved = registrations[0]
    store.register_analysis("task", job.owner, analysis_id, record, evidence=saved)
    assert store.get_task_archive("task") == before
    assert store.timeline("task") == history
    rejected_id = uuid4().hex
    rejected_evidence = evidence("task", job.owner, rejected_id)
    with pytest.raises(ArchiveCapacityError):
        store.register_analysis(
            "task",
            job.owner,
            rejected_id,
            analysis_record(job, rejected_id),
            evidence=rejected_evidence,
        )
    assert store.get_analysis_archive(rejected_id) is None
    assert store.get_task_archive("task") == before
    assert store.timeline("task") == history
    assert all(
        store.get_artifact_archive(item["artifact_id"]) is None
        for item in rejected_evidence["artifacts"]
    )
