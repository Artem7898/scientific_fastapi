"""Independent verification deliveries retain source pins and one fenced winner."""

from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scientific_api.core import task_store
from scientific_api.core.task_store import MemoryTaskStore, RedisTaskStore
from scientific_api.core.verification_store import (
    MemoryVerificationStore,
    RedisVerificationStore,
    VerificationCapacityError,
    VerificationConflictError,
)


@pytest.fixture(params=["memory", "redis"])
def backend(request, monkeypatch):
    stores = []
    namespace = "scientific-api-verification-store-test-" + uuid4().hex
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    client = None
    clock = [1000.0]
    if request.param == "memory":
        monkeypatch.setattr(task_store, "_now", lambda: clock[0])
    else:
        if not uri:
            pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
        redis = pytest.importorskip("redis")
        client = redis.Redis.from_url(
            uri, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()

    def make(**options):
        store = (
            MemoryVerificationStore(**options)
            if request.param == "memory"
            else RedisVerificationStore(uri, namespace=namespace, **options)
        )
        stores.append(store)
        return store

    def source_store(**options):
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

    yield SimpleNamespace(make=make, source_store=source_store, advance=advance, kind=request.param)
    for store in stores:
        store.close()
    if client is not None:
        keys = list(client.scan_iter(match=namespace + ":*"))
        if keys:
            client.delete(*keys)
        client.close()


def payload(source_id="source"):
    return {
        "source_id": source_id,
        "source": {
            "task_id": source_id,
            "attempt_id": "source-owner",
            "model_version": "test-model-is-opaque-to-the-store",
            "evidence": {"manifest_sha256": "a" * 64},
        },
        "options": {"seed": 2**64 - 1, "backend": "explicit-test-backend"},
    }


def certificate(verification_id="verification", verdict="passed"):
    return {
        "verification_id": verification_id,
        "status": "completed",
        "verdict": verdict,
        "certificate": {
            "source_manifest_sha256": "a" * 64,
            "seed": 2**64 - 1,
            "checks": [{"name": "independent_check", "outcome": verdict}],
        },
        "certificate_sha256": "b" * 64,
    }


def test_requests_are_immutable_and_preserve_uint64_without_physics_normalization(backend):
    store = backend.make()
    request = payload()
    saved = copy.deepcopy(request)
    carrier = {"traceparent": "private-trace"}
    store.submit("verification", request, carrier)
    request["options"]["seed"] = 0
    carrier.clear()
    accepted = store.get("verification")
    assert accepted == {
        "verification_id": "verification",
        "status": "pending",
        "request": saved,
        "submitted_at": accepted["submitted_at"],
        "attempts": 0,
    }
    assert isinstance(accepted["submitted_at"], (int, float))
    job = store.claim()
    assert job.task_id == "verification"
    assert job.payload == saved
    assert job.carrier == {"traceparent": "private-trace"}
    assert job.checkpoint is None
    job.payload["source"]["attempt_id"] = "tampered-local-copy"
    assert store.get("verification")["request"] == saved
    assert store.get("verification")["attempts"] == 1


@pytest.mark.parametrize("verdict", ["passed", "failed", "inconclusive"])
def test_completed_certificates_are_durable_and_not_execution_errors(backend, verdict):
    store = backend.make(ttl_seconds=0.03)
    store.submit("verification", payload())
    job = store.claim()
    completed = certificate(verdict=verdict)
    expected = copy.deepcopy(completed)
    assert store.finish(job, completed)
    completed["certificate"]["checks"][0]["outcome"] = "changed-after-commit"
    before = store.get("verification")
    assert before["status"] == "completed" and before["verdict"] == verdict
    assert before["certificate"] == expected["certificate"]
    assert "task_id" not in before
    assert not store.finish(job, certificate(verdict="replacement"))
    assert not store.finish(replace(job, owner="foreign"), certificate())
    backend.advance(0.06)
    assert store._queue.get("verification") is None
    assert store.get("verification") == before
    assert store.list_for_source("source") == [before]
    before["certificate"]["seed"] = 0
    assert store.get("verification")["certificate"]["seed"] == 2**64 - 1


def test_expired_lease_cannot_publish_and_retry_keeps_pinned_source(backend):
    store = backend.make(lease_seconds=0.1)
    store.submit("verification", payload(), {"traceparent": "same-context"})
    old = store.claim()
    backend.advance(0.15)
    assert not store.finish(old, certificate())
    assert not store.heartbeat(old)
    assert not store.release(old)
    new = store.claim()
    assert new.owner != old.owner
    assert new.payload == old.payload
    assert new.carrier == old.carrier
    assert not store.finish(old, certificate(verdict="obsolete"))
    assert store.finish(new, certificate())
    assert store.get("verification")["attempts"] == 2
    assert store.get("verification")["verdict"] == "passed"


def test_heartbeat_preserves_a_live_exclusive_delivery(backend):
    store = backend.make(lease_seconds=0.25)
    store.submit("verification", payload())
    job = store.claim()
    backend.advance(0.15)
    assert store.heartbeat(job)
    backend.advance(0.15)
    assert store.claim() is None
    assert store.finish(job, certificate())


def test_attempt_exhaustion_is_a_durable_execution_failure(backend):
    store = backend.make(lease_seconds=0.05, max_attempts=2, ttl_seconds=0.03)
    store.submit("verification", payload())
    assert store.claim() is not None
    backend.advance(0.08)
    assert store.claim() is not None
    backend.advance(0.08)
    assert store.claim() is None
    failed = store.get("verification")
    assert failed["status"] == "failed"
    assert "maximum attempts" in failed["error"]
    assert "certificate" not in failed
    assert failed["attempts"] == 2
    backend.advance(0.06)
    assert store.get("verification") == failed


def test_release_requeues_on_both_backends_without_spending_retry_budget(backend):
    store = backend.make(max_attempts=1)
    store.submit("verification", payload())
    old = store.claim()
    assert store.release(old)
    assert store.get("verification")["status"] == "pending"
    assert not store.release(old)
    new = store.claim()
    assert new.owner != old.owner
    assert new.payload == old.payload
    assert not store.finish(old, certificate(verdict="obsolete"))
    assert store.finish(new, certificate())
    assert store.get("verification")["attempts"] == 2


@pytest.mark.parametrize("when", ["pending", "running", "release"])
def test_cancellation_wins_and_never_publishes_a_certificate(backend, when):
    store = backend.make(ttl_seconds=0.03)
    store.submit("verification", payload())
    job = store.claim() if when != "pending" else None
    assert store.request_cancel("verification")
    assert store.request_cancel("verification")
    if when == "pending":
        assert store.claim() is None
    else:
        assert store.is_cancel_requested(job)
        assert store.release(job) if when == "release" else store.finish(job, certificate())
    cancelled = store.get("verification")
    assert cancelled["status"] == "cancelled"
    assert "certificate" not in cancelled
    assert cancelled["error"] == "Verification cancelled by request"
    backend.advance(0.06)
    assert store.get("verification") == cancelled


def test_dead_cancelled_delivery_is_not_retried(backend):
    store = backend.make(lease_seconds=0.05)
    store.submit("verification", payload())
    job = store.claim()
    assert store.request_cancel("verification")
    backend.advance(0.08)
    assert store.claim() is None
    assert store.get("verification")["status"] == "cancelled"
    assert not store.finish(job, certificate())


def test_concurrent_cancel_finish_has_one_actual_terminal_outcome(backend):
    store = backend.make()
    store.submit("verification", payload())
    job = store.claim()
    with ThreadPoolExecutor(max_workers=2) as pool:
        publishing = pool.submit(store.finish, job, certificate())
        cancelling = pool.submit(store.request_cancel, "verification")
        assert publishing.result()
        cancelled = cancelling.result()
    terminal = store.get("verification")
    assert terminal["status"] == ("cancelled" if cancelled else "completed")
    assert ("certificate" in terminal) == (not cancelled)
    assert not store.finish(job, certificate(verdict="replacement"))


def test_capacity_rejections_do_not_allocate_source_slots_or_ids(backend):
    store = backend.make(max_queue_size=1, max_per_source=1)
    store.submit("one", payload("source-a"))
    with pytest.raises(VerificationCapacityError) as error:
        store.submit("rejected", payload("source-b"))
    assert error.value.code == "VERIFICATION_QUEUE_FULL"
    assert store.get("rejected") is None
    assert store.list_for_source("source-b") == []
    job = store.claim()
    assert store.finish(job, certificate("one"))
    with pytest.raises(VerificationCapacityError) as error:
        store.submit("same-source", payload("source-a"))
    assert error.value.code == "VERIFICATION_SOURCE_LIMIT_EXCEEDED"
    store.submit("rejected", payload("source-b"))
    assert store.get("rejected")["status"] == "pending"


def test_source_limit_survives_terminal_ttl_and_duplicates_do_not_mix_sources(backend):
    store = backend.make(max_per_source=2, ttl_seconds=0.03)
    for verification_id in ("first", "second"):
        store.submit(verification_id, payload())
        job = store.claim()
        assert store.finish(job, certificate(verification_id))
    backend.advance(0.06)
    with pytest.raises(VerificationCapacityError) as error:
        store.submit("third", payload())
    assert error.value.code == "VERIFICATION_SOURCE_LIMIT_EXCEEDED"
    with pytest.raises(VerificationConflictError) as error:
        store.submit("first", payload("different-source"))
    assert error.value.code == "VERIFICATION_ALREADY_EXISTS"
    assert store.list_for_source("different-source") == []
    assert [item["verification_id"] for item in store.list_for_source("source")] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize("kind", ["global", "source"])
def test_concurrent_admission_respects_one_atomic_capacity(backend, kind):
    store = backend.make(max_queue_size=4, max_per_source=2 if kind == "source" else 32)

    def admit(index):
        try:
            store.submit(
                f"verification-{index}",
                payload("source" if kind == "source" else f"source-{index}"),
            )
            return index
        except VerificationCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = [value for value in pool.map(admit, range(20)) if value is not None]
    assert len(accepted) == (2 if kind == "source" else 4)
    jobs = []
    while (job := store.claim()) is not None:
        jobs.append(job)
    assert len({job.task_id for job in jobs}) == len(accepted)
    for index in range(20):
        assert (store.get(f"verification-{index}") is not None) == (index in accepted)


def test_verification_queue_is_separate_and_source_provenance_is_read_only(backend):
    source = backend.source_store()
    source.submit("same-id", {"seed": 2**64 - 1}, {})
    source_job = source.claim()
    assert source.finish(source_job, {"status": "completed", "attempt_id": source_job.owner})
    original = source.get_task_archive("same-id")
    original_timeline = source.timeline("same-id")
    store = backend.make()
    store.submit("same-id", payload("same-id"))
    assert source.claim() is None
    verification_job = store.claim()
    assert verification_job.task_id == "same-id"
    assert store.finish(verification_job, certificate("same-id"))
    assert source.get_task_archive("same-id") == original
    assert source.timeline("same-id") == original_timeline
    assert store.get("same-id")["certificate"]["seed"] == 2**64 - 1


@pytest.mark.parametrize(
    "invalid", ["missing-source", "nan", "unsafe-id", "bad-source", "bad-carrier"]
)
def test_invalid_admission_is_rejected_before_any_mutation(backend, invalid):
    store = backend.make()
    request, verification_id, carrier = payload(), "verification", None
    if invalid == "missing-source":
        request.pop("source_id")
    elif invalid == "nan":
        request["options"]["threshold"] = float("nan")
    elif invalid == "unsafe-id":
        verification_id = "../escape"
    elif invalid == "bad-source":
        request["source_id"] = "source/path"
    else:
        carrier = {"private": 42}
    with pytest.raises(ValueError):
        store.submit(verification_id, request, carrier)
    assert store.get("verification") is None
    assert store.list_for_source("source") == []
    assert store.claim() is None


@pytest.mark.parametrize(
    "field", ["verification_id", "request", "submitted_at", "attempts", "status"]
)
def test_terminal_records_cannot_replace_identity_or_admission_metadata(backend, field):
    store = backend.make()
    store.submit("verification", payload())
    job = store.claim()
    record = certificate()
    record[field] = {
        "verification_id": "other",
        "request": payload("other"),
        "submitted_at": 0,
        "attempts": 0,
        "status": "running",
    }[field]
    with pytest.raises(ValueError):
        store.finish(job, record)
    assert store.get("verification")["status"] == "running"
    assert store.finish(job, certificate())


def test_closed_stores_refuse_reads_even_when_the_source_has_no_entries(backend):
    store = backend.make()
    store.close()
    assert not store.health()
    with pytest.raises(RuntimeError, match="closed"):
        store.list_for_source("unknown")
    with pytest.raises(RuntimeError, match="closed"):
        store.get("unknown")
    with pytest.raises(RuntimeError, match="closed"):
        store.submit("verification", payload())


def test_redis_requests_and_certificates_survive_client_restart_and_lower_admission_limit(backend):
    if backend.kind != "redis":
        pytest.skip("Memory verification records are explicitly volatile")
    first = backend.make(ttl_seconds=0.03, max_per_source=2)
    for verification_id in ("first", "second"):
        first.submit(verification_id, payload())
        job = first.claim()
        assert first.finish(job, certificate(verification_id))
    expected = first.list_for_source("source")
    first.close()
    backend.advance(0.06)
    second = backend.make(max_per_source=1)
    assert second.list_for_source("source") == expected
    assert second.get("first")["certificate"]["seed"] == 2**64 - 1
    with pytest.raises(VerificationCapacityError):
        second.submit("third", payload())


@pytest.mark.parametrize(
    "options",
    [
        {"max_queue_size": 0},
        {"max_queue_size": True},
        {"max_attempts": 0},
        {"max_attempts": 101},
        {"max_per_source": 0},
        {"max_per_source": 33},
        {"lease_seconds": 0},
        {"ttl_seconds": -1},
    ],
)
def test_invalid_configuration_fails_before_creating_a_usable_store(options):
    with pytest.raises(ValueError):
        MemoryVerificationStore(**options)


def test_memory_verification_instances_do_not_share_requests():
    first, second = MemoryVerificationStore(), MemoryVerificationStore()
    try:
        first.submit("verification", payload())
        assert second.get("verification") is None
        assert second.list_for_source("source") == []
    finally:
        first.close()
        second.close()
