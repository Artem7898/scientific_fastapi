"""Registered checkpoints support explicit, provenance-preserving continuation."""

import asyncio
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np
import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from scientific_api.api.v1.endpoints import checkpoints as checkpoint_endpoints
from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.domain.experiments import parameters_from_payload
from scientific_api.main import create_app
from scientific_api.models.checkpoint_state import restore_simulator
from scientific_api.schemas.checkpoint import (
    CheckpointDetails,
    CheckpointList,
    CheckpointResumeAccepted,
)
from scientific_api.schemas.langevin import LangevinRequest
from scientific_api.schemas.lennard_jones import LennardJonesRequest
from scientific_api.schemas.simulation import SimulationRequest
from scientific_api.services import checkpoint_service
from scientific_api.services.checkpoint_service import CheckpointControl
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation

from .test_result_envelopes import CASES


class InjectedInterruption(RuntimeError):
    pass


def _settings(root, *, mode="embedded"):
    return Settings(
        _env_file=None,
        artifacts_dir=root,
        execution_mode=mode,
        redis_uri="redis://unused" if mode == "api" else "",
        mlflow_tracking_uri="",
        otel_enabled=False,
        checkpoint_interval_steps=4,
    )


def _seed_interrupted(
    root, monkeypatch, case=CASES[0], *, status="failed", ensemble=False, capacity=100
):
    route, original, _ = case
    payload = {**original, "num_steps": 12}
    request_type = {
        "monte-carlo": SimulationRequest,
        "langevin": LangevinRequest,
        "lennard-jones": LennardJonesRequest,
    }[route]
    queued = request_type.model_validate(payload).model_dump()
    if route != "monte-carlo":
        queued = {"model": case[2], **queued}
    parameters = parameters_from_payload(queued)
    backend = MemoryTaskStore(max_queue_size=capacity)
    task_id = "checkpoint-task"
    if ensemble:
        backend.submit_ensemble("parent-ensemble", [(task_id, queued)], {}, {})
    else:
        backend.submit(task_id, queued, {})
    job = backend.claim()
    assert job is not None

    def register(descriptor):
        assert backend.register_checkpoint(job, descriptor)

    control = CheckpointControl(
        parameters,
        task_id=task_id,
        attempt_id=job.owner,
        artifacts_dir=root,
        interval_steps=4,
        register=register,
    )
    original_save = CheckpointControl.save_if_due

    def interrupt_after_save(self, step, simulator, *, samples, next_sample):
        saved = original_save(self, step, simulator, samples=samples, next_sample=next_sample)
        if step == 4 and saved is not None:
            raise InjectedInterruption("Interrupt only after a complete registered checkpoint")
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(CheckpointControl, "save_if_due", interrupt_after_save)
        with pytest.raises(InjectedInterruption):
            run_simulation(
                parameters,
                task_id=task_id,
                attempt_id=job.owner,
                artifacts_dir=root,
                checkpoint_control=control,
            )
    if status == "cancelled":
        assert backend.request_cancel(task_id)
    assert backend.finish(job, {"status": "failed", "error": "Injected interruption"})
    assert backend.get(task_id)["status"] == status
    latest = backend.get_latest_checkpoint(task_id)
    assert latest is not None
    assert latest["completed_steps"] == 4
    return backend, parameters, job, latest


def _terminal(client, task_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/simulations/status/{task_id}")
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("Continuation did not reach a terminal state")


def _prefix(job):
    return f"/api/v1/simulations/{job.task_id}"


@pytest.mark.parametrize("case", CASES)
def test_resume_all_models_matches_uninterrupted_scientific_arrays_and_records_lineage(
    tmp_path, monkeypatch, case
):
    backend, parameters, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch, case)
    with TestClient(create_app(_settings(tmp_path), backend=backend)) as client:
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 202, response.text
        CheckpointResumeAccepted.model_validate_json(response.content)
        accepted = response.json()
        assert accepted["task_id"] == source.task_id
        assert accepted["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert accepted["deduplicated"] is False
        assert response.headers["location"] == accepted["check_url"]
        assert response.headers["retry-after"] == "1"
        result = _terminal(client, source.task_id)
        assert result["status"] == "completed", result
        assert result["attempt_id"] != source.owner
        reference = run_simulation(
            parameters,
            task_id="reference",
            attempt_id="reference-attempt",
            artifacts_dir=tmp_path,
        )
        for name in ("final_state", "trajectory"):
            if name not in result["artifacts"]:
                continue
            with (
                np.load(tmp_path / result["artifacts"][name], allow_pickle=False) as actual,
                np.load(tmp_path / reference["artifacts"][name], allow_pickle=False) as expected,
            ):
                assert set(actual.files) == set(expected.files)
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
        if "diagnostics" in result:
            assert result["diagnostics"] == reference["diagnostics"]
        manifest = json.loads((tmp_path / result["artifacts"]["manifest"]).read_text())
        assert manifest["outputs"] == result
        envelope = client.get(_prefix(source) + "/envelope").json()
        continuation = envelope["continuation"]
        assert continuation["mode"] == "resumed_run"
        assert continuation["logical_task_id"] == source.task_id
        assert continuation["parent_attempt_id"] == source.owner
        assert continuation["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert continuation["completed_steps_at_start"] == 4
        provenance = client.get(f"/api/v1/provenance/tasks/{source.task_id}")
        assert provenance.status_code == 200, provenance.text
        archived = provenance.json()
        assert archived["continuation"] == continuation
        assert archived["resume_count"] == 1
        assert archived["previous_outcomes"][0]["status"] == "failed"
        assert archived["checkpoint_integrity_status"] == "recorded_not_rechecked"
        graph_response = client.get(f"/api/v1/provenance/tasks/{source.task_id}/graph")
        assert graph_response.status_code == 200, graph_response.text
        graph = graph_response.json()["@graph"]
        checkpoint_urn = (
            f"urn:scientific-api:checkpoint:{source.task_id}:{checkpoint['checkpoint_id']}"
        )
        checkpoint_node = next(item for item in graph if item["@id"] == checkpoint_urn)
        assert checkpoint_node["sf:integrityStatus"] == "recorded_not_rechecked"
        assert checkpoint_node["sf:stateSha256"] == checkpoint["state"]["sha256"]
        resumed = next(item for item in graph if item.get("sf:attemptId") == result["attempt_id"])
        assert {"@id": checkpoint_urn} in resumed["prov:used"]
        assert resumed["sf:executionMode"] == "resumed_run"
        assert resumed["prov:wasInformedBy"] == {
            "@id": f"urn:scientific-api:attempt:{source.task_id}:{source.owner}"
        }
        timeline = client.get(f"/api/v1/tasks/{source.task_id}/timeline").json()
        assert {"checkpoint_created", "resume_requested", "attempt_resumed"} <= {
            event["event_type"] for event in timeline["events"]
        }


def test_checkpoint_listing_is_metadata_only_while_details_and_downloads_recheck_bytes(
    tmp_path, monkeypatch
):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        listed = client.get(_prefix(source) + "/checkpoints")
        assert listed.status_code == 200, listed.text
        CheckpointList.model_validate_json(listed.content)
        assert listed.json()["integrity_status"] == "recorded_not_rechecked"
        assert listed.json()["checkpoints"][-1] == checkpoint
        url = _prefix(source) + f"/checkpoints/{checkpoint['checkpoint_id']}"
        details = client.get(url)
        assert details.status_code == 200, details.text
        CheckpointDetails.model_validate_json(details.content)
        assert details.json()["descriptor"] == checkpoint
        assert details.json()["integrity_status"] == "verified"
        assert details.json()["fingerprint_status"] == "not_checked"
        assert details.json()["authenticity"] == "unsigned"
        for name, mime in (("manifest", "application/json"), ("state", "application/octet-stream")):
            artifact = client.get(url + f"/artifacts/{name}")
            assert artifact.status_code == 200, artifact.text
            assert artifact.headers["content-type"].startswith(mime)
            assert hashlib.sha256(artifact.content).hexdigest() == checkpoint[name]["sha256"]
            assert artifact.headers["cache-control"] == "no-store"
        (tmp_path / checkpoint["state"]["path"]).write_bytes(b"changed")
        assert client.get(_prefix(source) + "/checkpoints").json() == listed.json()
        for path in (url, url + "/artifacts/state", url + "/artifacts/manifest"):
            failure = client.get(path)
            assert failure.status_code == 409, failure.text
            assert failure.json()["detail"]["code"] == "CHECKPOINT_INTEGRITY_FAILED"
        # Provenance explicitly records descriptor metadata, rather than
        # pretending it has verified every retained checkpoint file.
        graph = client.get(f"/api/v1/provenance/tasks/{source.task_id}/graph")
        assert graph.status_code == 200, graph.text


def test_active_resume_is_idempotent_and_api_only_preflight_defers_cpu_compatibility(
    tmp_path, monkeypatch
):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    called = []
    original_load = checkpoint_endpoints.load_checkpoint

    def observe_load(*args, **kwargs):
        called.append(kwargs["validate_fingerprint"])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_endpoints, "load_checkpoint", observe_load)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        first = client.post(_prefix(source) + "/resume", json={})
        assert first.status_code == 202, first.text
        second = client.post(
            _prefix(source) + "/resume", json={"checkpoint_id": checkpoint["checkpoint_id"]}
        )
        assert second.status_code == 202, second.text
        assert first.json()["deduplicated"] is False
        assert second.json()["deduplicated"] is True
        assert first.json()["checkpoint_id"] == second.json()["checkpoint_id"]
        assert called == [False]
        assert backend.get_task_archive(source.task_id)["resume_count"] == 1
        claimed = backend.claim()
        assert claimed is not None
        assert claimed.checkpoint == checkpoint


def test_incompatible_local_fingerprint_refuses_without_fresh_run_fallback(tmp_path, monkeypatch):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    fingerprint = checkpoint_service.checkpoint_fingerprint

    def incompatible(parameters):
        result = fingerprint(parameters)
        result["changed_cpu_test"] = True
        return result

    monkeypatch.setattr(checkpoint_service, "checkpoint_fingerprint", incompatible)
    with TestClient(create_app(_settings(tmp_path), backend=backend)) as client:
        details = client.get(_prefix(source) + f"/checkpoints/{checkpoint['checkpoint_id']}")
        assert details.status_code == 200, details.text
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "REPRODUCIBILITY_CONFLICT"
        assert backend.get(source.task_id)["status"] == "failed"
        assert backend.get_task_archive(source.task_id)["resume_count"] == 0


def test_default_resume_retry_keeps_admitted_checkpoint_when_worker_saves_a_newer_one(
    tmp_path, monkeypatch
):
    backend, parameters, source, original = _seed_interrupted(tmp_path, monkeypatch)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        first = client.post(_prefix(source) + "/resume", json={})
        assert first.status_code == 202, first.text
        assert first.json()["checkpoint_id"] == original["checkpoint_id"]
        resumed = backend.claim()
        assert resumed is not None
        assert resumed.checkpoint == original
        loaded = checkpoint_service.load_checkpoint(
            original, parameters=parameters, artifacts_dir=tmp_path
        )
        simulator = restore_simulator(parameters, loaded.model_metadata, loaded.model_arrays)
        for _ in range(4):
            simulator.advance()

        def register(descriptor):
            assert backend.register_checkpoint(resumed, descriptor)

        control = CheckpointControl(
            parameters,
            task_id=source.task_id,
            attempt_id=resumed.owner,
            artifacts_dir=tmp_path,
            interval_steps=4,
            loaded=loaded,
            register=register,
        )
        newer = control.save_if_due(8, simulator, samples={}, next_sample=0)
        assert newer is not None
        assert newer["checkpoint_id"] != original["checkpoint_id"]
        assert backend.get_latest_checkpoint(source.task_id) == newer

        def forbidden(*args, **kwargs):
            pytest.fail("An active idempotent resume must not rehydrate either checkpoint")

        monkeypatch.setattr(checkpoint_endpoints, "load_checkpoint", forbidden)
        retry = client.post(_prefix(source) + "/resume", json={})
        assert retry.status_code == 202, retry.text
        assert retry.json()["status"] == "running"
        assert retry.json()["deduplicated"] is True
        assert retry.json()["checkpoint_id"] == original["checkpoint_id"]
        different = client.post(
            _prefix(source) + "/resume", json={"checkpoint_id": newer["checkpoint_id"]}
        )
        assert different.status_code == 409, different.text
        assert different.json()["detail"]["code"] == "RESUME_STATE_CONFLICT"
        assert backend.get_task_archive(source.task_id)["resume_count"] == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_resume_remains_available_after_operational_ttl(tmp_path, monkeypatch, status):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch, status=status)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        now = task_store._now()
        monkeypatch.setattr(task_store, "_now", lambda: now + 7200)
        assert client.get(f"/api/v1/simulations/status/{source.task_id}").status_code == 404
        listed = client.get(_prefix(source) + "/checkpoints")
        assert listed.status_code == 200, listed.text
        assert listed.json()["checkpoints"][-1] == checkpoint
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "pending"
        assert client.get(response.json()["check_url"]).json()["status"] == "pending"


def test_only_latest_checkpoint_can_start_a_new_continuation(tmp_path, monkeypatch):
    backend, _, source, latest = _seed_interrupted(tmp_path, monkeypatch)
    earliest = backend.list_checkpoints(source.task_id)[0]
    assert earliest["checkpoint_id"] != latest["checkpoint_id"]
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.post(
            _prefix(source) + "/resume", json={"checkpoint_id": earliest["checkpoint_id"]}
        )
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "CHECKPOINT_NOT_LATEST"
        assert backend.get_task_archive(source.task_id)["resume_count"] == 0


def test_completed_task_cannot_be_resumed(tmp_path, monkeypatch):
    backend, parameters, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    backend.request_resume(source.task_id, checkpoint["checkpoint_id"], {})
    claimed = backend.claim()
    assert claimed is not None
    result = run_simulation(
        parameters, task_id=source.task_id, attempt_id=claimed.owner, artifacts_dir=tmp_path
    )
    envelope = read_result_envelope(result, artifacts_dir=tmp_path)
    assert backend.finish(claimed, result, evidence=build_archive_evidence(envelope))
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "RESUME_STATE_CONFLICT"


def test_ensemble_children_cannot_be_resumed_individually(tmp_path, monkeypatch):
    backend, _, source, _ = _seed_interrupted(tmp_path, monkeypatch, ensemble=True)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "RESUME_ENSEMBLE_UNSUPPORTED"


def test_resume_obeys_queue_capacity_and_preserves_terminal_record_on_rejection(
    tmp_path, monkeypatch
):
    backend, _, source, _ = _seed_interrupted(tmp_path, monkeypatch, capacity=1)
    original = backend.get(source.task_id)
    backend.submit("other-active-task", {"num_steps": 1}, {})
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 429, response.text
        assert response.json()["detail"]["code"] == "TASK_CAPACITY_EXHAUSTED"
        assert response.headers["retry-after"] == "1"
        assert backend.get(source.task_id) == original
        assert backend.get_task_archive(source.task_id)["resume_count"] == 0


@pytest.mark.parametrize("name", ["manifest", "state"])
def test_corrupt_checkpoint_preflight_refuses_before_requeue(tmp_path, monkeypatch, name):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    (tmp_path / checkpoint[name]["path"]).write_bytes(b"private /storage/secret")
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.post(_prefix(source) + "/resume", json={})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "CHECKPOINT_INTEGRITY_FAILED"
        assert "private" not in response.text
        assert "secret" not in response.text
        assert backend.get(source.task_id)["status"] == "failed"


@pytest.mark.parametrize(
    "body",
    [
        {"seed": 42},
        {"checkpoint_id": 1},
        {"checkpoint_id": "invalid"},
        {"checkpoint_id": "A" * 32},
        {"num_steps": 10},
    ],
)
def test_resume_does_not_accept_parameter_changes(tmp_path, body):
    with TestClient(
        create_app(_settings(tmp_path, mode="api"), backend=MemoryTaskStore())
    ) as client:
        response = client.post("/api/v1/simulations/missing/resume", json=body)
        assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/v1/simulations/missing/checkpoints", "get"),
        ("/api/v1/simulations/missing/checkpoints/" + "0" * 32, "get"),
        ("/api/v1/simulations/missing/checkpoints/invalid", "get"),
        ("/api/v1/simulations/missing/checkpoints/" + "0" * 32 + "/artifacts/state", "get"),
        ("/api/v1/simulations/missing/resume", "post"),
    ],
)
def test_unknown_tasks_or_checkpoints_have_safe_404s(tmp_path, path, method):
    with TestClient(
        create_app(_settings(tmp_path, mode="api"), backend=MemoryTaskStore())
    ) as client:
        response = client.post(path, json={}) if method == "post" else client.get(path)
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"]
        assert response.headers["cache-control"] == "no-store"


def test_storage_errors_do_not_reveal_checkpoint_paths(tmp_path, monkeypatch):
    backend = MemoryTaskStore()

    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credentials: secret value")

    monkeypatch.setattr(backend, "list_checkpoints", unavailable)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        response = client.get("/api/v1/simulations/missing/checkpoints")
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "CHECKPOINT_STORAGE_UNAVAILABLE"
        assert "private" not in response.text
        assert "secret" not in response.text


def test_ro_crate_records_checkpoint_lineage_without_including_restart_state_files(
    tmp_path, monkeypatch
):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    with TestClient(create_app(_settings(tmp_path), backend=backend)) as client:
        accepted = client.post(_prefix(source) + "/resume", json={})
        assert accepted.status_code == 202, accepted.text
        result = _terminal(client, source.task_id)
        assert result["status"] == "completed", result
        exported = client.get(f"/api/v1/provenance/tasks/{source.task_id}/export")
        assert exported.status_code == 200, exported.text
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            assert not any("/checkpoints/" in path for path in archive.namelist())
            task_path = next(path for path in archive.namelist() if path.endswith("task.json"))
            task = json.loads(archive.read(task_path))
            assert task["continuation"]["checkpoint_id"] == checkpoint["checkpoint_id"]
            assert task["checkpoints"]


def test_openapi_declares_checkpoint_contracts(tmp_path):
    schema = create_app(_settings(tmp_path)).openapi()
    for path, method, status, response_type in (
        ("/api/v1/simulations/{task_id}/checkpoints", "get", "200", "CheckpointList"),
        (
            "/api/v1/simulations/{task_id}/checkpoints/{checkpoint_id}",
            "get",
            "200",
            "CheckpointDetails",
        ),
        ("/api/v1/simulations/{task_id}/resume", "post", "202", "CheckpointResumeAccepted"),
    ):
        actual = schema["paths"][path][method]["responses"][status]["content"]["application/json"]
        assert actual["schema"]["$ref"] == f"#/components/schemas/{response_type}"


def test_parallel_resume_preflights_have_bounded_capacity_until_actual_completion(
    tmp_path, monkeypatch
):
    backend, _, source, checkpoint = _seed_interrupted(tmp_path, monkeypatch)
    entered, release, lock = Event(), Event(), Lock()
    count = 0
    original = checkpoint_endpoints.load_checkpoint

    def blocked(*args, **kwargs):
        nonlocal count
        with lock:
            count += 1
            if count == 2:
                entered.set()
        assert release.wait(10), "Checkpoint test failed to release its preflights"
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint_endpoints, "load_checkpoint", blocked)
    with TestClient(create_app(_settings(tmp_path, mode="api"), backend=backend)) as client:
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures = [
                callers.submit(client.post, _prefix(source) + "/resume", json={}) for _ in range(2)
            ]
            try:
                assert entered.wait(5), "Both admitted checkpoint reads must start"
                rejected = client.post(_prefix(source) + "/resume", json={})
                assert rejected.status_code == 429, rejected.text
                assert (
                    rejected.json()["detail"]["code"] == "CHECKPOINT_PREFLIGHT_CAPACITY_EXHAUSTED"
                )
                assert rejected.headers["retry-after"] == "1"
                assert backend.get_task_archive(source.task_id)["resume_count"] == 0
                download = client.get(
                    _prefix(source) + f"/checkpoints/{checkpoint['checkpoint_id']}/artifacts/state"
                )
                assert download.status_code == 429, download.text
                assert (
                    download.json()["detail"]["code"] == "CHECKPOINT_PREFLIGHT_CAPACITY_EXHAUSTED"
                )
            finally:
                release.set()
            responses = [future.result(timeout=10) for future in futures]
            assert all(response.status_code == 202 for response in responses)
        assert backend.request_cancel(source.task_id)
        retry = client.post(_prefix(source) + "/resume", json={})
        assert retry.status_code == 202, retry.text
        assert backend.get_task_archive(source.task_id)["resume_count"] == 2


def test_cancelled_http_preflight_keeps_queued_work_and_releases_its_slot(tmp_path):
    # A dedicated one-thread executor makes the scientific read wait in the
    # executor queue. An HTTP cancellation must not cancel that queued job,
    # otherwise its thread-finally never runs and the capacity slot is lost.
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        unhandled = []
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        release, occupied, executed = Event(), Event(), Event()
        request = Request({"type": "http", "app": create_app(_settings(tmp_path))})

        def occupy():
            occupied.set()
            assert release.wait(10)

        def abandoned():
            executed.set()
            raise RuntimeError("A late worker failure must be consumed after HTTP cancellation")

        blocker = loop.run_in_executor(None, occupy)
        deadline = loop.time() + 5
        while not occupied.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.001)
        assert occupied.is_set()
        cancelled = asyncio.create_task(checkpoint_endpoints._bounded_read(request, abandoned))
        second = None
        try:
            await asyncio.sleep(0)
            assert occupied.is_set()
            assert not executed.is_set()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            second = asyncio.create_task(
                checkpoint_endpoints._bounded_read(request, lambda: "second completed")
            )
            await asyncio.sleep(0)
            with pytest.raises(HTTPException) as refused:
                await checkpoint_endpoints._bounded_read(request, lambda: "not admitted")
            assert refused.value.status_code == 429
            assert refused.value.detail["code"] == "CHECKPOINT_PREFLIGHT_CAPACITY_EXHAUSTED"
        finally:
            release.set()
        await blocker
        assert second is not None
        assert await second == "second completed"
        assert executed.is_set()
        assert await checkpoint_endpoints._bounded_read(request, lambda: "reused") == "reused"
        await asyncio.sleep(0)
        assert unhandled == []

    asyncio.run(scenario())
