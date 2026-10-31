"""Research adapters preserve durable leases, opaque reports and source isolation."""

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
from scientific_api.core.research_store import (
    MemoryResearchStore,
    RedisResearchStore,
    ResearchCapacityError,
    ResearchConflictError,
)
from scientific_api.core.task_store import MemoryTaskStore, RedisTaskStore
from scientific_api.core.verification_store import MemoryVerificationStore, RedisVerificationStore


@pytest.fixture(params=["memory", "redis"])
def backend(request, monkeypatch):
    stores = []
    namespace = "scientific-api-research-store-test-" + uuid4().hex
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    client = None
    clock = [1000.0]
    if request.param == "memory":
        monkeypatch.setattr(task_store, "_now", lambda: clock[0])
    else:
        if not uri:
            pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
        redis = pytest.importorskip("redis")
        client = redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=2)
        client.ping()

    def make(**options):
        store = (
            MemoryResearchStore(**options)
            if request.param == "memory"
            else RedisResearchStore(uri, namespace=namespace, **options)
        )
        stores.append(store)
        return store

    def other_store(kind):
        if kind == "source":
            store = (
                MemoryTaskStore()
                if request.param == "memory"
                else RedisTaskStore(uri, namespace=namespace)
            )
        else:
            store = (
                MemoryVerificationStore()
                if request.param == "memory"
                else RedisVerificationStore(uri, namespace=namespace)
            )
        stores.append(store)
        return store

    def advance(seconds):
        if request.param == "memory":
            clock[0] += seconds
        else:
            time.sleep(seconds)

    yield SimpleNamespace(make=make, other_store=other_store, advance=advance, kind=request.param)
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
        "source": {"task_id": source_id, "attempt_id": "source-owner", "manifest_sha256": "a" * 64},
        "options": {"bootstrap_seed": 2**64 - 1, "method": "opaque-to-storage"},
    }


def report(analysis_id="analysis"):
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "report": {"seed": 2**64 - 1, "availability": "partial", "diagnostics": [1.0, 2.0]},
        "report_sha256": "b" * 64,
    }


def test_public_ids_are_analysis_ids_and_requests_remain_opaque(backend):
    store = backend.make()
    request, carrier = payload(), {"traceparent": "private-carrier"}
    expected = copy.deepcopy(request)
    store.submit("analysis", request, carrier)
    request["options"]["bootstrap_seed"] = 0
    carrier.clear()
    pending = store.get("analysis")
    assert pending["analysis_id"] == "analysis"
    assert "verification_id" not in pending
    assert pending["request"] == expected
    assert pending["status"] == "pending"
    job = store.claim()
    assert job.task_id == "analysis" and job.payload == expected
    assert job.carrier == {"traceparent": "private-carrier"}
    assert job.checkpoint is None
    assert store.list_for_source("source") == [store.get("analysis")]


def test_reports_are_immutable_and_remain_available_after_operational_ttl(backend):
    store = backend.make(ttl_seconds=0.03)
    store.submit("analysis", payload())
    job = store.claim()
    completed = report()
    saved = copy.deepcopy(completed)
    assert store.finish(job, completed)
    assert completed == saved, "ID translation must not mutate the caller's terminal record"
    completed["report"]["seed"] = 0
    before = store.get("analysis")
    assert before["report"] == saved["report"]
    assert before["report_sha256"] == saved["report_sha256"]
    assert "verification_id" not in before and "task_id" not in before
    assert not store.finish(job, report())
    backend.advance(0.06)
    assert store._store._queue.get("analysis") is None
    assert store.get("analysis") == before
    assert store.list_for_source("source") == [before]
    before["report"]["diagnostics"].clear()
    assert store.get("analysis")["report"]["diagnostics"] == [1.0, 2.0]


@pytest.mark.parametrize("field", ["analysis_id", "verification_id"])
def test_foreign_terminal_identity_is_rejected_before_publication(backend, field):
    store = backend.make()
    store.submit("analysis", payload())
    job = store.claim()
    terminal = report()
    terminal[field] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        store.finish(job, terminal)
    assert store.get("analysis")["status"] == "running"
    assert store.finish(job, report())


def test_stale_attempt_cannot_overwrite_winning_report(backend):
    store = backend.make(lease_seconds=0.1)
    store.submit("analysis", payload())
    old = store.claim()
    backend.advance(0.15)
    assert not store.finish(old, report())
    assert not store.heartbeat(old)
    current = store.claim()
    assert current.owner != old.owner and current.payload == old.payload
    assert store.finish(current, report())
    assert not store.finish(old, {**report(), "report_sha256": "c" * 64})
    assert store.get("analysis")["attempts"] == 2
    assert store.get("analysis")["report_sha256"] == "b" * 64


def test_release_requeues_without_consuming_attempt_budget_and_cancellation_wins(backend):
    store = backend.make(max_attempts=1)
    store.submit("analysis", payload())
    old = store.claim()
    assert store.release(old)
    assert store.get("analysis")["status"] == "pending"
    job = store.claim()
    assert job.owner != old.owner and job.payload == old.payload
    assert not store.finish(old, report())
    assert store.request_cancel("analysis")
    assert store.is_cancel_requested(job)
    assert store.finish(job, report())
    cancelled = store.get("analysis")
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] == "Research analysis cancelled by request"
    assert "report" not in cancelled
    assert not store.release(replace(job, owner="wrong-owner"))


def test_pending_cancel_does_not_run_and_releases_global_admission(backend):
    store = backend.make(max_queue_size=1)
    store.submit("analysis", payload())
    assert store.request_cancel("analysis")
    assert store.claim() is None
    assert store.get("analysis")["status"] == "cancelled"
    store.submit("another", payload())
    assert store.claim().task_id == "another"


def test_retry_exhaustion_is_an_execution_failure_without_report(backend):
    store = backend.make(lease_seconds=0.05, max_attempts=1)
    store.submit("analysis", payload())
    assert store.claim() is not None
    backend.advance(0.08)
    assert store.claim() is None
    failed = store.get("analysis")
    assert failed["status"] == "failed" and failed["attempts"] == 1
    assert "report" not in failed


def test_capacity_and_conflict_codes_use_research_terminology_without_partial_admission(backend):
    store = backend.make(max_queue_size=1, max_per_source=1)
    store.submit("analysis", payload())
    with pytest.raises(ResearchCapacityError) as error:
        store.submit("rejected", payload("new-source"))
    assert error.value.code == "RESEARCH_QUEUE_FULL"
    assert store.list_for_source("new-source") == []
    assert store.get("rejected") is None
    job = store.claim()
    assert store.finish(job, report())
    with pytest.raises(ResearchCapacityError) as error:
        store.submit("source-rejected", payload())
    assert error.value.code == "RESEARCH_SOURCE_LIMIT_EXCEEDED"
    with pytest.raises(ResearchConflictError) as error:
        store.submit("analysis", payload("different-source"))
    assert error.value.code == "RESEARCH_ALREADY_EXISTS"
    assert store.list_for_source("different-source") == []
    store.submit("rejected", payload("new-source"))


def test_concurrent_source_admission_is_bounded_and_has_no_ghost_records(backend):
    store = backend.make(max_per_source=2)

    def admit(index):
        try:
            store.submit(f"analysis-{index}", payload())
            return index
        except ResearchCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = [index for index in pool.map(admit, range(12)) if index is not None]
    assert len(accepted) == 2
    assert len(store.list_for_source("source")) == 2
    for index in range(12):
        assert (store.get(f"analysis-{index}") is not None) == (index in accepted)


def test_simulations_verifications_and_research_share_no_deliveries_or_provenance_writes(backend):
    simulations = backend.other_store("source")
    verifications = backend.other_store("verification")
    simulations.submit("same-id", {"seed": 2**64 - 1}, {})
    source_job = simulations.claim()
    assert simulations.finish(source_job, {"status": "completed", "attempt_id": source_job.owner})
    original_source = simulations.get_task_archive("same-id")
    original_timeline = simulations.timeline("same-id")
    verifications.submit("same-id", payload("same-id"))
    original_verification = verifications.get("same-id")
    research = backend.make()
    research.submit("same-id", payload("same-id"))
    research_job = research.claim()
    assert research_job.task_id == "same-id"
    assert research.finish(research_job, report("same-id"))
    assert simulations.claim() is None
    assert simulations.get_task_archive("same-id") == original_source
    assert simulations.timeline("same-id") == original_timeline
    assert verifications.get("same-id") == original_verification
    assert verifications.claim().task_id == "same-id"


def test_redis_research_records_survive_a_new_client_and_lower_admission_limit(backend):
    if backend.kind != "redis":
        pytest.skip("Memory research records are explicitly volatile")
    first = backend.make(max_per_source=2, ttl_seconds=0.03)
    for identifier in ("first", "second"):
        first.submit(identifier, payload())
        assert first.finish(first.claim(), report(identifier))
    expected = first.list_for_source("source")
    first.close()
    backend.advance(0.06)
    second = backend.make(max_per_source=1)
    assert second.list_for_source("source") == expected
    assert second.get("first")["report"]["seed"] == 2**64 - 1


def test_closed_research_store_rejects_even_empty_source_lookup(backend):
    store = backend.make()
    store.close()
    assert not store.health()
    with pytest.raises(RuntimeError, match="closed"):
        store.list_for_source("unknown")
    with pytest.raises(RuntimeError, match="closed"):
        store.get("unknown")


def test_memory_research_default_admission_capacity_is_sixteen():
    store = MemoryResearchStore()
    try:
        for index in range(16):
            store.submit(f"analysis-{index}", payload(f"source-{index}"))
        with pytest.raises(ResearchCapacityError) as error:
            store.submit("overflow", payload("last-source"))
        assert error.value.code == "RESEARCH_QUEUE_FULL"
    finally:
        store.close()


@pytest.mark.parametrize(
    "options",
    [{"max_queue_size": 0}, {"max_per_source": 33}, {"max_attempts": 0}, {"lease_seconds": 0}],
)
def test_research_configuration_reuses_validated_bounds(options):
    with pytest.raises(ValueError):
        MemoryResearchStore(**options)
