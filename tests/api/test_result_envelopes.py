"""Additive result evidence endpoints verify the bytes currently on disk."""

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientific_api.api.v1.endpoints import result_envelopes
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app
from scientific_api.schemas.result_envelope import ResultEnvelope

from .test_simulations import PAYLOAD, configuration

CASES = (
    ("monte-carlo", PAYLOAD, "ideal-gas-ou-v1"),
    (
        "langevin",
        {
            "initial_temperature": 1.0,
            "bath_temperature": 1.0,
            "num_particles": 16,
            "time_step": 0.1,
            "num_steps": 10,
            "sample_stride": 2,
            "seed": 42,
        },
        "langevin-free-v1",
    ),
    (
        "lennard-jones",
        {
            "num_particles": 16,
            "num_steps": 4,
            "sample_stride": 2,
            "seed": 42,
        },
        "lennard-jones-2d-v1",
    ),
)


def _run(client: TestClient, route: str = "monte-carlo", payload: dict | None = None) -> dict:
    admitted = client.post(f"/api/v1/simulations/{route}", json=payload or PAYLOAD)
    assert admitted.status_code == 202, admitted.text
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(admitted.json()["check_url"])
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            assert result["status"] == "completed", result
            return result
        time.sleep(0.01)
    pytest.fail("Result envelope workflow did not complete")


def _url(result: dict) -> str:
    return f"/api/v1/simulations/{result['task_id']}/envelope"


def _envelope_file(root: Path, result: dict) -> Path:
    return root / result["task_id"] / result["attempt_id"] / "envelope.json"


@pytest.mark.parametrize("route,payload,model_version", CASES)
def test_each_model_has_verified_envelope_without_changing_legacy_status(
    tmp_path, route, payload, model_version
):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client, route, payload)
        raw_manifest = (tmp_path / result["artifacts"]["manifest"]).read_bytes()
        response = client.get(_url(result))
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        envelope = response.json()
        ResultEnvelope.model_validate(envelope)
        assert envelope["kind"] == "simulation"
        assert envelope["task_id"] == result["task_id"]
        assert envelope["attempt_id"] == result["attempt_id"]
        assert envelope["result"] == result
        assert envelope["result"]["model_version"] == model_version
        assert envelope["integrity"]["status"] == "verified"
        assert envelope["manifest"]["sha256"] == hashlib.sha256(raw_manifest).hexdigest()
        assert json.loads(raw_manifest)["outputs"] == result
        assert _envelope_file(tmp_path, result).is_file()
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result
        for record in envelope["artifacts"]:
            assert (
                record["sha256"]
                == hashlib.sha256((tmp_path / record["path"]).read_bytes()).hexdigest()
            )


@pytest.mark.parametrize("target", ["manifest", "final_state", "envelope"])
def test_envelope_rechecks_integrity_on_every_request(tmp_path, target):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        assert client.get(_url(result)).status_code == 200
        path = (
            _envelope_file(tmp_path, result)
            if target == "envelope"
            else tmp_path / result["artifacts"][target]
        )
        path.write_bytes(b"corrupted artifact with private /server/secret location")
        response = client.get(_url(result))
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"]
        assert "/server/secret" not in response.text
        assert "private" not in response.text
        # Evidence failures do not rewrite the already committed scientific result.
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


@pytest.mark.parametrize("target", ["manifest", "final_state", "envelope"])
def test_marked_result_cannot_silently_lose_an_artifact(tmp_path, target):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        path = (
            _envelope_file(tmp_path, result)
            if target == "envelope"
            else tmp_path / result["artifacts"][target]
        )
        path.unlink()
        response = client.get(_url(result))
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"]


def test_unmarked_historical_result_is_explicitly_not_verified(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        manifest_path = tmp_path / result["artifacts"]["manifest"]
        manifest = json.loads(manifest_path.read_text())
        del manifest["result_envelope"]
        manifest_path.write_text(json.dumps(manifest))
        _envelope_file(tmp_path, result).unlink()
        response = client.get(_url(result))
        assert response.status_code == 200, response.text
        envelope = response.json()
        ResultEnvelope.model_validate(envelope)
        assert envelope["result"] == result
        assert envelope["integrity"]["status"] == "not_verified"
        assert envelope["integrity"]["reason"]
        assert envelope["artifacts"] == []
        assert envelope["manifest"] is None


def test_analysis_has_separate_envelope_and_detects_changed_analysis(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        parent_envelope = client.get(_url(result)).json()
        created = client.post(
            f"/api/v1/simulations/{result['task_id']}/analyses", json={"bins": 12}
        )
        assert created.status_code == 201, created.text
        analysis = created.json()
        url = f"/api/v1/simulations/{result['task_id']}/analyses/{analysis['analysis_id']}/envelope"
        response = client.get(url)
        assert response.status_code == 200, response.text
        envelope = response.json()
        ResultEnvelope.model_validate(envelope)
        assert envelope["kind"] == "velocity_analysis"
        assert envelope["analysis_id"] == analysis["analysis_id"]
        assert envelope["integrity"]["status"] == "verified"
        for name in ("analysis_id", "task_id", "attempt_id", "status", "analysis_version"):
            assert envelope["result"][name] == analysis[name]
        raw_manifest = client.get(analysis["artifacts"]["manifest"]).content
        assert envelope["manifest"]["sha256"] == hashlib.sha256(raw_manifest).hexdigest()
        analysis_path = (
            tmp_path
            / result["task_id"]
            / result["attempt_id"]
            / "analyses"
            / analysis["analysis_id"]
            / "analysis.json"
        )
        analysis_path.write_text("{}")
        failure = client.get(url)
        assert failure.status_code == 409, failure.text
        assert failure.json()["detail"]["code"]
        assert client.get(_url(result)).json() == parent_envelope


def test_missing_tasks_and_unknown_analysis_are_not_found(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        assert client.get("/api/v1/simulations/missing/envelope").status_code == 404
        result = _run(client)
        for analysis_id in ("invalid", "0" * 32, "a" * 129):
            response = client.get(
                f"/api/v1/simulations/{result['task_id']}/analyses/{analysis_id}/envelope"
            )
            assert response.status_code == 404, response.text


@pytest.mark.parametrize("terminal", [False, True])
def test_noncompleted_tasks_have_structured_refusal(tmp_path, terminal):
    backend = MemoryTaskStore()
    backend.submit("waiting", PAYLOAD, {})
    if terminal:
        job = backend.claim()
        assert job is not None
        backend.finish(job, {"status": "failed", "error": "Test failure"})
    settings = Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )
    with TestClient(create_app(settings, backend=backend)) as client:
        response = client.get("/api/v1/simulations/waiting/envelope")
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "RESULT_NOT_COMPLETED"


def test_storage_failure_is_safe_and_does_not_change_scientific_status(tmp_path, monkeypatch):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)

        def unavailable(*args, **kwargs):
            raise RuntimeError("/private/credential-file: secret value")

        monkeypatch.setattr(result_envelopes, "read_result_envelope", unavailable)
        response = client.get(_url(result))
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "EVIDENCE_STORAGE_UNAVAILABLE"
        assert "private" not in response.text
        assert "secret" not in response.text
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


def test_envelopes_are_declared_as_typed_openapi_responses(tmp_path):
    schema = create_app(configuration(tmp_path)).openapi()
    for path in (
        "/api/v1/simulations/{task_id}/envelope",
        "/api/v1/simulations/{task_id}/analyses/{analysis_id}/envelope",
    ):
        response_schema = schema["paths"][path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert response_schema == {"$ref": "#/components/schemas/ResultEnvelope"}
