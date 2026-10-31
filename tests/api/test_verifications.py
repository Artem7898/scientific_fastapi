"""Independent verification is asynchronous, pinned and read-only for its source."""

import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from scientific_api.core import task_store, verification_manager
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.verification_store import MemoryVerificationStore
from scientific_api.main import create_app
from scientific_api.schemas.simulation import SimulationRequest
from scientific_api.schemas.verification import (
    VerificationAccepted,
    VerificationCertificate,
    VerificationList,
    VerificationRecord,
)
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import (
    publish_result_envelope,
    read_result_envelope,
)
from scientific_api.services.simulation_service import run_simulation

from .test_result_envelopes import CASES, _run
from .test_simulations import PAYLOAD, configuration


def _source_url(result):
    return f"/api/v1/simulations/{result['task_id']}/verifications"


def _submit(client, result, *, rerun=False):
    response = client.post(_source_url(result), json={"rerun": rerun})
    assert response.status_code == 202, response.text
    VerificationAccepted.model_validate_json(response.content)
    accepted = response.json()
    assert accepted["source_task_id"] == result["task_id"]
    assert accepted["source_attempt_id"] == result["attempt_id"]
    assert response.headers["location"] == accepted["check_url"]
    assert response.headers["retry-after"] == "1"
    assert response.headers["cache-control"] == "no-store"
    return accepted


def _terminal(client, accepted):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(accepted["check_url"])
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        VerificationRecord.model_validate_json(response.content)
        record = response.json()
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    pytest.fail("Independent verification did not reach its terminal state")


def _certificate(client, accepted):
    response = client.get(accepted["check_url"] + "/certificate")
    assert response.status_code == 200, response.text
    VerificationCertificate.model_validate_json(response.content)
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"].startswith("attachment; filename=")
    assert response.headers["x-content-sha256"] == hashlib.sha256(response.content).hexdigest()
    return response


def _seed_completed(root, backend, task_id="archived-source"):
    request = SimulationRequest.model_validate(PAYLOAD)
    backend.submit(task_id, request.model_dump(), {})
    job = backend.claim()
    assert job is not None
    result = run_simulation(
        request.to_parameters(), task_id=task_id, attempt_id=job.owner, artifacts_dir=root
    )
    envelope = read_result_envelope(result, artifacts_dir=root)
    assert backend.finish(job, result, evidence=build_archive_evidence(envelope))
    return result


def _api_settings(root):
    return Settings(
        _env_file=None,
        artifacts_dir=root,
        execution_mode="api",
        redis_uri="redis://unused",
        mlflow_tracking_uri="",
        otel_enabled=False,
    )


@pytest.mark.parametrize("route,payload,model_version", CASES)
def test_all_models_produce_unsigned_scoped_certificates_without_mutating_source(
    tmp_path, route, payload, model_version
):
    app = create_app(configuration(tmp_path))
    with TestClient(app) as client:
        result = _run(client, route, payload)
        directory = tmp_path / result["task_id"] / result["attempt_id"]
        original_files = {
            path.relative_to(directory): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file()
        }
        archive = app.state.backend.get_task_archive(result["task_id"])
        accepted = _submit(client, result)
        finished = _terminal(client, accepted)
        assert finished["status"] == "completed", finished
        assert finished["verdict"] == "passed", finished
        certificate = _certificate(client, accepted)
        body = certificate.json()
        assert body["verification_id"] == accepted["verification_id"]
        assert body["verdict"] == finished["verdict"]
        assert body["source"]["task_id"] == result["task_id"]
        assert body["source"]["attempt_id"] == result["attempt_id"]
        assert body["source"]["model_version"] == model_version
        assert (
            body["source"]["manifest_sha256"]
            == hashlib.sha256(
                original_files[
                    next(path for path in original_files if str(path) == "manifest.json")
                ]
            ).hexdigest()
        )
        assert body["authenticity"] == "unsigned"
        assert body["scope"] == "recorded_artifacts_at_verification_time"
        assert body["options"] == {"rerun": False}
        assert body["checks"][0]["name"] == "artifact_integrity"
        assert all(check["status"] == "passed" for check in body["checks"])
        assert body["limitations"]
        assert finished["certificate_sha256"] == certificate.headers["x-content-sha256"]
        assert client.get(accepted["check_url"] + "/certificate").content == certificate.content
        listing = client.get(_source_url(result))
        assert listing.status_code == 200, listing.text
        VerificationList.model_validate_json(listing.content)
        assert listing.json()["verifications"] == [finished]
        assert app.state.backend.get_task_archive(result["task_id"]) == archive
        assert {
            path.relative_to(directory): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file()
        } == original_files
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result
        assert (
            client.get(f"/api/v1/simulations/{result['task_id']}/envelope").json()[
                "scientific_verification"
            ]
            == "not_performed"
        )


def test_explicit_small_same_seed_rerun_adds_evidence_without_new_source_attempt(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        accepted = _submit(client, result, rerun=True)
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        assert record["verdict"] == "passed", record
        certificate = _certificate(client, accepted).json()
        assert certificate["options"]["rerun"] is True
        assert any("rerun" in check["name"] for check in certificate["checks"])
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


@pytest.mark.parametrize("corruption", ["state", "manifest", "coherent_rewrite"])
def test_corrupt_source_is_admitted_and_receives_failed_verdict_not_admission_error(
    tmp_path, corruption
):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        manifest_path = tmp_path / result["artifacts"]["manifest"]
        original_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if corruption == "state":
            (tmp_path / result["artifacts"]["final_state"]).write_bytes(b"changed scientific data")
        else:
            manifest = json.loads(manifest_path.read_text())
            manifest["environment"]["processor"] = "changed after archived commit"
            manifest_path.write_text(json.dumps(manifest))
            if corruption == "coherent_rewrite":
                publish_result_envelope(result, artifacts_dir=tmp_path)
        accepted = _submit(client, result)
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        assert record["verdict"] == "failed", record
        certificate = _certificate(client, accepted).json()
        assert certificate["source"]["manifest_sha256"] == original_digest
        assert certificate["checks"][0]["name"] == "artifact_integrity"
        assert certificate["checks"][0]["status"] == "failed"
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


def test_saved_certificate_remains_identical_after_source_changes_and_ttl_expires(
    tmp_path, monkeypatch
):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        accepted = _submit(client, result)
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        original = _certificate(client, accepted)
        (tmp_path / result["artifacts"]["final_state"]).unlink()
        now = task_store._now()
        monkeypatch.setattr(task_store, "_now", lambda: now + 7200)
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").status_code == 404
        later = _certificate(client, accepted)
        assert later.content == original.content
        assert later.headers["x-content-sha256"] == original.headers["x-content-sha256"]
        assert client.get(accepted["check_url"]).json() == record
        assert client.get(_source_url(result)).json()["verifications"] == [record]


def test_new_verification_can_be_requested_after_source_live_ttl(tmp_path, monkeypatch):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        now = task_store._now()
        monkeypatch.setattr(task_store, "_now", lambda: now + 7200)
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").status_code == 404
        accepted = _submit(client, result)
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        assert record["verdict"] == "passed", record
        assert _certificate(client, accepted).json()["source"]["task_id"] == result["task_id"]


def test_api_only_submission_stays_pending_until_a_separate_verifier_consumes_it(tmp_path):
    source = MemoryTaskStore()
    result = _seed_completed(tmp_path, source)
    verification = MemoryVerificationStore()
    with TestClient(
        create_app(_api_settings(tmp_path), backend=source, verification_backend=verification)
    ) as client:
        accepted = _submit(client, result)
        record = client.get(accepted["check_url"])
        assert record.status_code == 200, record.text
        assert record.json()["status"] == "pending"
        assert record.json()["verdict"] is None
        assert record.json()["attempts"] == 0
        certificate = client.get(accepted["check_url"] + "/certificate")
        assert certificate.status_code == 409, certificate.text
        assert certificate.json()["detail"]["code"] == "VERIFICATION_CERTIFICATE_NOT_READY"
        queued = verification.claim()
        assert queued is not None
        assert queued.task_id == accepted["verification_id"]
        assert queued.payload["source"]["attempt_id"] == result["attempt_id"]


def test_retained_source_limit_has_no_retry_after_or_partial_admission(tmp_path):
    source = MemoryTaskStore()
    result = _seed_completed(tmp_path, source)
    verification = MemoryVerificationStore(max_queue_size=10, max_per_source=2)
    with TestClient(
        create_app(_api_settings(tmp_path), backend=source, verification_backend=verification)
    ) as client:
        accepted = [_submit(client, result), _submit(client, result)]
        response = client.post(_source_url(result), json={})
        assert response.status_code == 429, response.text
        assert response.json()["detail"]["code"] == "VERIFICATION_SOURCE_LIMIT_EXCEEDED"
        assert "retry-after" not in response.headers
        listing = client.get(_source_url(result)).json()["verifications"]
        assert [record["verification_id"] for record in listing] == [
            record["verification_id"] for record in accepted
        ]


def test_global_queue_limit_has_retry_after_and_does_not_allocate_a_source_slot(tmp_path):
    source = MemoryTaskStore()
    first = _seed_completed(tmp_path, source, "first-source")
    second = _seed_completed(tmp_path, source, "second-source")
    verification = MemoryVerificationStore(max_queue_size=1)
    with TestClient(
        create_app(_api_settings(tmp_path), backend=source, verification_backend=verification)
    ) as client:
        _submit(client, first)
        response = client.post(_source_url(second), json={})
        assert response.status_code == 429, response.text
        assert response.json()["detail"]["code"] == "VERIFICATION_QUEUE_FULL"
        assert response.headers["retry-after"] == "1"
        assert client.get(_source_url(second)).json()["verifications"] == []


@pytest.mark.parametrize("status", ["pending", "running", "failed", "cancelled", "completed"])
def test_unavailable_sources_are_refused_without_reading_untrusted_files(tmp_path, status):
    source = MemoryTaskStore()
    source.submit("unavailable", PAYLOAD, {})
    if status != "pending":
        job = source.claim()
        assert job is not None
        if status != "running":
            assert source.finish(job, {"status": status, "error": "No scientific evidence"})
    with TestClient(create_app(_api_settings(tmp_path), backend=source)) as client:
        response = client.post("/api/v1/simulations/unavailable/verifications", json={})
        assert response.status_code == 409, response.text
        expected = (
            "VERIFICATION_SOURCE_EVIDENCE_UNAVAILABLE"
            if status == "completed"
            else "VERIFICATION_SOURCE_NOT_COMPLETED"
        )
        assert response.json()["detail"]["code"] == expected
        assert (
            client.get("/api/v1/simulations/unavailable/verifications").json()["verifications"]
            == []
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"rerun": "true"},
        {"rerun": 1},
        {"rerun": None},
        {"seed": 42},
        {"rerun": False, "backend": "jax"},
    ],
)
def test_verification_requests_are_strict_and_do_not_change_source_parameters(tmp_path, payload):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post("/api/v1/simulations/missing/verifications", json=payload)
        assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/simulations/missing/verifications"),
        ("get", "/api/v1/simulations/missing/verifications"),
        ("get", "/api/v1/verifications/" + "0" * 32),
        ("get", "/api/v1/verifications/" + "0" * 32 + "/certificate"),
        ("get", "/api/v1/verifications/invalid"),
        ("get", "/api/v1/verifications/invalid/certificate"),
    ],
)
def test_unknown_sources_and_verifications_have_safe_404s(tmp_path, method, path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post(path, json={}) if method == "post" else client.get(path)
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"]
        assert response.headers["cache-control"] == "no-store"


def test_infrastructure_failure_is_not_a_failed_scientific_verdict(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credential-file: secret value")

    monkeypatch.setattr(verification_manager, "verify_result", unavailable)
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        accepted = _submit(client, result)
        record = _terminal(client, accepted)
        assert record["status"] == "failed", record
        assert record["verdict"] is None
        assert record["certificate_sha256"] is None
        assert record["error_code"] == "VERIFICATION_EXECUTION_FAILED"
        assert "private" not in record["error"]
        assert "secret" not in record["error"]
        assert client.get(accepted["check_url"] + "/certificate").status_code == 409


def test_changed_certificate_is_rejected_against_its_committed_sha256(tmp_path, monkeypatch):
    app = create_app(configuration(tmp_path))
    with TestClient(app) as client:
        result = _run(client)
        accepted = _submit(client, result)
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        _certificate(client, accepted)
        store = app.state.verification_manager.backend
        original = store.get

        def changed(identifier):
            stored = original(identifier)
            if stored is not None and identifier == accepted["verification_id"]:
                stored["certificate"]["limitations"].append("Changed after terminal commit")
            return stored

        monkeypatch.setattr(store, "get", changed)
        response = client.get(accepted["check_url"] + "/certificate")
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "VERIFICATION_CERTIFICATE_INTEGRITY_FAILED"


def test_api_storage_failures_do_not_reveal_internal_details(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credential-file: secret value")

    monkeypatch.setattr(verification_manager.VerificationManager, "get", unavailable)
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get("/api/v1/verifications/" + "0" * 32)
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "VERIFICATION_STORAGE_UNAVAILABLE"
        assert "private" not in response.text
        assert "secret" not in response.text


def test_openapi_distinguishes_verification_status_and_downloaded_certificate(tmp_path):
    schema = create_app(configuration(tmp_path)).openapi()
    for path, method, status, response_type in (
        ("/api/v1/simulations/{task_id}/verifications", "post", "202", "VerificationAccepted"),
        ("/api/v1/simulations/{task_id}/verifications", "get", "200", "VerificationList"),
        ("/api/v1/verifications/{verification_id}", "get", "200", "VerificationRecord"),
        (
            "/api/v1/verifications/{verification_id}/certificate",
            "get",
            "200",
            "VerificationCertificate",
        ),
    ):
        actual = schema["paths"][path][method]["responses"][status]["content"]["application/json"]
        assert actual["schema"]["$ref"] == f"#/components/schemas/{response_type}"
