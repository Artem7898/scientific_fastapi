"""Registered checkpoint selection and fenced, bounded continuation on both stores."""

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
from scientific_api.core.task_store import (
    CheckpointStoreError,
    Job,
    MemoryTaskStore,
    QueueFull,
    RedisTaskStore,
)
from scientific_api.domain.checkpoint import (
    checkpoint_model_version,
    checkpoint_parameters_sha256,
)
from scientific_api.domain.experiments import parameters_from_payload

PAYLOAD = {
    "initial_temperature": 1.0,
    "num_particles": 16,
    "num_steps": 10,
    "seed": 2**64 - 1,
}


@pytest.fixture(params=["memory", "redis"])
def backend(request, monkeypatch):
    stores = []
    namespace = "scientific-api-checkpoint-test-" + uuid4().hex
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    clock = [1000.0]
    client = None
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

    yield SimpleNamespace(make=make, advance=advance, kind=request.param)
    for store in stores:
        store.close()
    if client is not None:
        keys = list(client.scan_iter(match=namespace + ":*"))
        if keys:
            client.delete(*keys)
        client.close()


def checkpoint(job: Job, step: int = 2, checkpoint_id: str | None = None):
    checkpoint_id = checkpoint_id or uuid4().hex
    parameters = parameters_from_payload(job.payload)
    prefix = f"{job.task_id}/{job.owner}/checkpoints/{checkpoint_id}"
    return {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "task_id": job.task_id,
        "attempt_id": job.owner,
        "completed_steps": step,
        "total_steps": parameters.num_steps,
        "model_version": checkpoint_model_version(parameters),
        "parameters_sha256": checkpoint_parameters_sha256(parameters),
        "fingerprint_sha256": "a" * 64,
        "manifest": {"path": f"{prefix}/checkpoint.json", "sha256": "b" * 64, "bytes": 10},
        "state": {"path": f"{prefix}/state.npz", "sha256": "c" * 64, "bytes": 20},
        "created_at": "2026-11-01T12:00:00+00:00",
    }


def submit(store, task_id="task", **changes):
    store.submit(task_id, {**PAYLOAD, **changes}, {})
    job = store.claim()
    assert job is not None and job.task_id == task_id
    return job


def finish(store, job, status="failed"):
    record = {
        "task_id": job.task_id,
        "attempt_id": job.owner,
        "status": status,
        "seed": 2**64 - 1,
    }
    assert store.finish(job, record)
    return record


def history(store, task_id="task"):
    return store.timeline(task_id, limit=256)["events"]


def expect_code(code, function, *args, **kwargs):
    with pytest.raises(CheckpointStoreError) as error:
        function(*args, **kwargs)
    assert error.value.code == code


def test_unregistered_files_are_not_checkpoint_registry_entries(backend, tmp_path):
    store = backend.make()
    assert store.list_checkpoints("unknown") is None
    assert store.get_latest_checkpoint("unknown") is None
    assert store.get_checkpoint("unknown", uuid4().hex) is None
    store.submit("task", PAYLOAD, {})
    orphan = tmp_path / "task" / "orphan-owner" / "checkpoints" / uuid4().hex
    orphan.mkdir(parents=True)
    (orphan / "checkpoint.json").write_text("{}")
    (orphan / "state.npz").write_bytes(b"unregistered")
    job = store.claim()
    assert job.checkpoint is None
    assert store.list_checkpoints("task") == []


def test_zero_step_registration_is_idempotent_immutable_and_survives_ttl(backend):
    store = backend.make(ttl_seconds=0.03)
    job = submit(store)
    saved = checkpoint(job, step=0)
    assert store.register_checkpoint(job, saved)
    before = history(store)
    assert store.register_checkpoint(job, copy.deepcopy(saved))
    assert history(store) == before
    assert store.get_checkpoint("task", saved["checkpoint_id"]) == saved
    assert store.list_checkpoints("task") == [saved]
    assert store.get_latest_checkpoint("task") == saved
    copy_of_saved = copy.deepcopy(saved)
    saved["state"]["bytes"] = 999
    assert store.get_latest_checkpoint("task") == copy_of_saved
    finish(store, job)
    backend.advance(0.06)
    assert store.get("task") is None
    assert store.get_latest_checkpoint("task") == copy_of_saved
    assert store.get_task_archive("task")["checkpoints"] == [copy_of_saved]
    assert [item["event_type"] for item in before] == [
        "submitted",
        "attempt_claimed",
        "checkpoint_created",
    ]


def test_steps_increase_strictly_but_older_identical_id_is_still_idempotent(backend):
    store = backend.make()
    job = submit(store)
    first, latest = checkpoint(job, 0), checkpoint(job, 4)
    assert store.register_checkpoint(job, first)
    assert store.register_checkpoint(job, latest)
    before = history(store)
    assert store.register_checkpoint(job, first)
    expect_code("CHECKPOINT_STEP_CONFLICT", store.register_checkpoint, job, checkpoint(job, 4))
    expect_code("CHECKPOINT_STEP_CONFLICT", store.register_checkpoint, job, checkpoint(job, 3))
    changed = copy.deepcopy(first)
    changed["state"]["sha256"] = "d" * 64
    expect_code("CHECKPOINT_CONFLICT", store.register_checkpoint, job, changed)
    assert history(store) == before
    assert store.list_checkpoints("task") == [first, latest]


def test_lease_and_cancel_fences_reject_new_or_retried_checkpoint_registration(backend):
    store = backend.make(lease_seconds=0.03)
    old = submit(store)
    saved = checkpoint(old)
    assert store.register_checkpoint(old, saved)
    backend.advance(0.06)
    before = history(store)
    assert not store.register_checkpoint(old, saved)
    new = store.claim()
    assert new.owner != old.owner
    assert new.checkpoint == saved
    assert not store.register_checkpoint(old, checkpoint(old, 4))
    assert store.request_cancel("task")
    current = history(store)
    assert not store.register_checkpoint(new, checkpoint(new, 4))
    assert history(store) == current
    assert len(before) == 3
    assert store.list_checkpoints("task") == [saved]


@pytest.mark.parametrize("mutation", ["model", "digest", "total", "job-payload"])
def test_checkpoint_matches_normalized_original_physical_parameters(backend, mutation):
    store = backend.make()
    job = submit(store)
    saved = checkpoint(job)
    if mutation == "model":
        saved["model_version"] = "langevin-free-v1"
    elif mutation == "digest":
        saved["parameters_sha256"] = "0" * 64
    elif mutation == "total":
        saved["total_steps"] += 1
    else:
        job = replace(job, payload={**job.payload, "seed": 42})
    before = history(store)
    expect_code("CHECKPOINT_PARAMETERS_MISMATCH", store.register_checkpoint, job, saved)
    assert store.list_checkpoints("task") == []
    assert history(store) == before


def test_missing_model_backend_defaults_are_normalized_without_rewriting_payload(backend):
    store = backend.make()
    payload = {"initial_temperature": 1.0, "num_particles": 16, "seed": 2**64 - 1}
    store.submit("task", payload, {})
    job = store.claim()
    saved = checkpoint(job, 0)
    assert saved["total_steps"] == 100
    assert store.register_checkpoint(job, saved)
    assert store.get_task_archive("task")["payload"] == payload
    finish(store, job)
    store.request_resume("task", saved["checkpoint_id"], {})
    assert store.claim().payload == payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("checkpoint_id", "BAD"),
        ("completed_steps", True),
        ("completed_steps", -1),
        ("completed_steps", 11),
        ("total_steps", 0),
        ("fingerprint_sha256", "bad"),
        ("created_at", "2026-11-01T12:00:00"),
        ("task_id", "other"),
        ("attempt_id", "other"),
    ],
)
def test_invalid_descriptors_never_mutate_the_registry(backend, field, value):
    store = backend.make()
    job = submit(store)
    saved = checkpoint(job)
    saved[field] = value
    before = history(store)
    expect_code("CHECKPOINT_DESCRIPTOR_INVALID", store.register_checkpoint, job, saved)
    assert store.list_checkpoints("task") == []
    assert history(store) == before


@pytest.mark.parametrize("kind", ["path", "bytes", "hash", "extra-field"])
def test_checkpoint_artifact_descriptors_are_strict_and_attempt_scoped(backend, kind):
    store = backend.make()
    job = submit(store)
    saved = checkpoint(job)
    if kind == "path":
        saved["state"]["path"] = "outside/state.npz"
    elif kind == "bytes":
        saved["state"]["bytes"] = True
    elif kind == "hash":
        saved["manifest"]["sha256"] = "INVALID"
    else:
        saved["injected"] = "not in the checkpoint contract"
    expect_code("CHECKPOINT_DESCRIPTOR_INVALID", store.register_checkpoint, job, saved)
    assert store.list_checkpoints("task") == []


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_checkpoint_limits_are_atomic_and_allow_exact_idempotent_retry(backend, monkeypatch, limit):
    if limit == "count":
        monkeypatch.setattr(task_store, "MAX_CHECKPOINTS_PER_TASK", 1)
    else:
        monkeypatch.setattr(task_store, "MAX_CHECKPOINT_TOTAL_BYTES", 50)
    store = backend.make()
    job = submit(store)
    first = checkpoint(job)
    assert store.register_checkpoint(job, first)
    before = store.get_task_archive("task")
    assert store.register_checkpoint(job, first)
    rejected = checkpoint(job, 4)
    expect_code("CHECKPOINT_LIMIT_EXCEEDED", store.register_checkpoint, job, rejected)
    assert store.get_task_archive("task") == before
    assert store.get_checkpoint("task", rejected["checkpoint_id"]) is None


def test_automatic_retry_atomically_selects_latest_registered_checkpoint(backend):
    store = backend.make(lease_seconds=0.03)
    old = submit(store)
    first, latest = checkpoint(old, 0), checkpoint(old, 5)
    assert store.register_checkpoint(old, first)
    assert store.register_checkpoint(old, latest)
    backend.advance(0.06)
    new = store.claim()
    assert new.owner != old.owner
    assert new.payload == old.payload
    assert new.checkpoint == latest
    archive = store.get_task_archive("task")
    assert archive["selected_checkpoint"] == latest
    assert [item["attempt"] for item in archive["attempts"]] == [1, 2]
    assert all(set(item) == {"attempt_id", "attempt", "claimed_at"} for item in archive["attempts"])
    events = history(store)
    assert [item["event_type"] for item in events][-3:] == [
        "retry",
        "attempt_claimed",
        "attempt_resumed",
    ]
    assert events[-1]["details"] == {
        "checkpoint_id": latest["checkpoint_id"],
        "parent_attempt_id": old.owner,
        "completed_steps": 5,
    }
    assert store.events("task")["snapshot"]["completed_steps"] == 5


def test_retry_without_registered_checkpoint_remains_a_seeded_restart(backend):
    store = backend.make(lease_seconds=0.03)
    old = submit(store)
    backend.advance(0.06)
    new = store.claim()
    assert new.checkpoint is None
    assert new.payload == old.payload
    assert "attempt_resumed" not in [item["event_type"] for item in history(store)]


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_manual_resume_rehydrates_expired_terminal_task_and_preserves_outcomes(backend, terminal):
    store = backend.make(ttl_seconds=0.03, max_attempts=1)
    old = submit(store)
    saved = checkpoint(old, 4)
    assert store.register_checkpoint(old, saved)
    original_record = finish(store, old, terminal)
    old_sse = store.events("task")["last_event_id"]
    backend.advance(0.06)
    assert store.get("task") is None
    response = store.request_resume("task", saved["checkpoint_id"], {"traceparent": "new-context"})
    assert response == {
        "task_id": "task",
        "status": "pending",
        "checkpoint_id": saved["checkpoint_id"],
        "deduplicated": False,
    }
    assert store.get("task") == {"task_id": "task", "status": "pending"}
    archive = store.get_task_archive("task")
    assert archive["previous_outcomes"] == [original_record]
    assert archive["resume_count"] == 1
    new = store.claim()
    assert new.owner != old.owner and new.checkpoint == saved
    assert new.payload == old.payload
    assert new.carrier == {"traceparent": "new-context"}
    assert [item["attempt"] for item in store.get_task_archive("task")["attempts"]] == [1, 2]
    assert store.events("task")["last_event_id"] != old_sse
    finish(store, new)


def test_manual_resume_deduplicates_pending_running_and_later_checkpoint_creation(backend):
    store = backend.make()
    old = submit(store)
    saved = checkpoint(old, 2)
    assert store.register_checkpoint(old, saved)
    finish(store, old)
    assert not store.request_resume("task", saved["checkpoint_id"], {"context": "first"})[
        "deduplicated"
    ]
    before = history(store)
    assert store.request_resume("task", saved["checkpoint_id"], {"context": "ignored"})[
        "deduplicated"
    ]
    assert history(store) == before
    new = store.claim()
    later = checkpoint(new, 4)
    assert store.register_checkpoint(new, later)
    assert store.request_resume("task", saved["checkpoint_id"], {}) == {
        "task_id": "task",
        "status": "running",
        "checkpoint_id": saved["checkpoint_id"],
        "deduplicated": True,
    }
    expect_code("RESUME_STATE_CONFLICT", store.request_resume, "task", later["checkpoint_id"], {})
    assert store.get_task_archive("task")["resume_count"] == 1
    assert new.carrier == {"context": "first"}


def test_manual_resume_requires_latest_checkpoint_and_refuses_completed_or_active_task(backend):
    store = backend.make()
    job = submit(store)
    first, latest = checkpoint(job, 0), checkpoint(job, 2)
    assert store.register_checkpoint(job, first)
    assert store.register_checkpoint(job, latest)
    expect_code("RESUME_STATE_CONFLICT", store.request_resume, "task", latest["checkpoint_id"], {})
    finish(store, job)
    expect_code("CHECKPOINT_NOT_LATEST", store.request_resume, "task", first["checkpoint_id"], {})
    expect_code("CHECKPOINT_NOT_FOUND", store.request_resume, "task", uuid4().hex, {})
    expect_code("TASK_NOT_FOUND", store.request_resume, "unknown", uuid4().hex, {})
    store.request_resume("task", latest["checkpoint_id"], {})
    resumed = store.claim()
    finish(store, resumed, "completed")
    expect_code("RESUME_STATE_CONFLICT", store.request_resume, "task", latest["checkpoint_id"], {})


def test_manual_resume_never_restarts_ensemble_children(backend):
    store = backend.make()
    store.submit_ensemble("ensemble", [("child", PAYLOAD)], {}, {})
    job = store.claim()
    saved = checkpoint(job)
    assert store.register_checkpoint(job, saved)
    finish(store, job)
    before = history(store, "child")
    expect_code(
        "RESUME_ENSEMBLE_UNSUPPORTED", store.request_resume, "child", saved["checkpoint_id"], {}
    )
    assert history(store, "child") == before


def test_manual_resume_capacity_rejection_has_no_partial_history_or_state(backend):
    store = backend.make(max_queue_size=1)
    job = submit(store)
    saved = checkpoint(job)
    assert store.register_checkpoint(job, saved)
    finish(store, job)
    before = store.get_task_archive("task")
    store.submit("occupying", PAYLOAD, {})
    with pytest.raises(QueueFull):
        store.request_resume("task", saved["checkpoint_id"], {})
    assert store.get_task_archive("task") == before
    assert store.get("task")["status"] == "failed"


def test_manual_resume_limit_resets_retry_budget_but_keeps_uint64_previous_outcomes(
    backend, monkeypatch
):
    monkeypatch.setattr(task_store, "MAX_MANUAL_RESUMES", 2)
    store = backend.make(max_attempts=1)
    job = submit(store)
    saved = checkpoint(job)
    assert store.register_checkpoint(job, saved)
    outcomes = [finish(store, job)]
    for _ in range(2):
        store.request_resume("task", saved["checkpoint_id"], {})
        job = store.claim()
        outcomes.append(finish(store, job))
    before = store.get_task_archive("task")
    assert before["previous_outcomes"] == outcomes[:2]
    assert before["resume_count"] == 2
    assert [item["attempt"] for item in before["attempts"]] == [1, 2, 3]
    expect_code("RESUME_LIMIT_EXCEEDED", store.request_resume, "task", saved["checkpoint_id"], {})
    assert store.get_task_archive("task") == before


def test_concurrent_manual_resume_requests_enqueue_one_delivery(backend):
    store = backend.make(max_queue_size=1)
    job = submit(store)
    saved = checkpoint(job)
    assert store.register_checkpoint(job, saved)
    finish(store, job)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: store.request_resume("task", saved["checkpoint_id"], {}), range(8))
        )
    assert sum(not item["deduplicated"] for item in responses) == 1
    assert store.get_task_archive("task")["resume_count"] == 1
    assert sum(item["event_type"] == "resume_requested" for item in history(store)) == 1
    assert store.claim() is not None
    assert store.claim() is None


def test_cancel_during_a_manual_resume_is_not_undone_by_idempotent_retry(backend):
    store = backend.make()
    job = submit(store)
    saved = checkpoint(job)
    assert store.register_checkpoint(job, saved)
    finish(store, job)
    store.request_resume("task", saved["checkpoint_id"], {})
    resumed = store.claim()
    assert store.request_cancel("task")
    assert store.request_resume("task", saved["checkpoint_id"], {})["deduplicated"]
    assert store.is_cancel_requested(resumed)
    finish(store, resumed, "completed")
    assert store.get("task")["status"] == "cancelled"


def test_redis_checkpoints_and_manual_resume_survive_client_restart(backend):
    if backend.kind != "redis":
        pytest.skip("Memory checkpoints are explicitly volatile across process restarts")
    first = backend.make(ttl_seconds=0.03)
    job = submit(first)
    saved = checkpoint(job)
    assert first.register_checkpoint(job, saved)
    record = finish(first, job)
    first.close()
    backend.advance(0.06)
    second = backend.make()
    assert second.get("task") is None
    assert second.get_latest_checkpoint("task") == saved
    second.request_resume("task", saved["checkpoint_id"], {})
    second.close()
    third = backend.make()
    resumed = third.claim()
    assert resumed.checkpoint == saved
    assert resumed.payload == PAYLOAD
    assert third.get_task_archive("task")["previous_outcomes"] == [record]
