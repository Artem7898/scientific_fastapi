"""Archival provenance survives live TTL and verifies exact committed evidence."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientific_api.api.v1.endpoints import analyses as analysis_endpoints
from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app
from scientific_api.schemas.provenance import (
    ProvenanceAnalysis,
    ProvenanceArtifact,
    ProvenanceGraph,
    ProvenanceTask,
    ProvenanceTimeline,
)
from scientific_api.services import analysis_service
from scientific_api.services.provenance_service import ProvenanceError, ProvenanceService
from scientific_api.services.result_envelope_service import publish_result_envelope

from .test_result_envelopes import CASES, _run
from .test_simulations import PAYLOAD, configuration


def _task_url(result: dict) -> str:
    return f"/api/v1/provenance/tasks/{result['task_id']}"


def _timeline_url(result: dict) -> str:
    return f"/api/v1/tasks/{result['task_id']}/timeline"


def _document(client, url, model):
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    model.model_validate_json(response.content)
    return response.json()


def _analysis(client, result):
    response = client.post(f"/api/v1/simulations/{result['task_id']}/analyses", json={"bins": 12})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("route,payload,model_version", CASES)
def test_completed_models_have_archived_metadata_and_queryable_prov_graph(
    tmp_path, route, payload, model_version
):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client, route, payload)
        task = _document(client, _task_url(result), ProvenanceTask)
        assert task["task_id"] == result["task_id"]
        assert task["status"] == "completed"
        assert task["coverage"] == "complete"
        assert task["storage"] == "memory"
        assert task["model_version"] == model_version
        assert task["record"] == result
        assert task["evidence_status"] == "verified"
        manifest = (tmp_path / result["artifacts"]["manifest"]).read_bytes()
        assert task["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
        assert task["parameters"]["seed"] == payload.get("seed", 0)
        assert task["rng"]["bit_generator"] == "PCG64"
        assert task["execution"]["backend"] == "numpy"
        assert task["environment"]["source_sha256"]
        assert task["attempts"][-1]["attempt_id"] == result["attempt_id"]
        assert {record["name"] for record in task["artifacts"]} >= set(result["artifacts"])
        graph = _document(client, _task_url(result) + "/graph", ProvenanceGraph)
        assert graph["task_id"] == result["task_id"]
        assert graph["evidence_status"] == "verified"
        assert graph["@context"]["prov"] == "http://www.w3.org/ns/prov#"
        assert graph["@graph"]
        types = []
        for node in graph["@graph"]:
            value = node.get("@type", [])
            types.extend([value] if isinstance(value, str) else value)
        assert "prov:Entity" in types
        assert "prov:Activity" in types
        assert "prov:Person" not in types
        timeline = _document(client, _timeline_url(result), ProvenanceTimeline)
        event_types = [event["event_type"] for event in timeline["events"]]
        assert event_types[0] == "submitted"
        assert "attempt_claimed" in event_types
        assert event_types[-1] == "completed"
        assert "heartbeat" not in event_types
        assert "progress" not in event_types


def test_indexed_artifact_id_identifies_origin_separately_from_identical_bytes(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        origins = []
        for _ in range(2):
            result = _run(client)
            task = _document(client, _task_url(result), ProvenanceTask)
            state = next(record for record in task["artifacts"] if record["name"] == "final_state")
            artifact = _document(
                client,
                f"/api/v1/provenance/artifacts/{state['artifact_id']}",
                ProvenanceArtifact,
            )
            assert artifact == state
            assert artifact["task_id"] == result["task_id"]
            assert artifact["attempt_id"] == result["attempt_id"]
            assert (
                artifact["sha256"]
                == hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest()
            )
            graph = _document(client, _task_url(result) + "/graph", ProvenanceGraph)
            assert any(
                node.get("@id") == f"urn:scientific-api:artifact:{artifact['artifact_id']}"
                for node in graph["@graph"]
            )
            origins.append(artifact)
        assert origins[0]["sha256"] == origins[1]["sha256"]
        assert origins[0]["artifact_id"] != origins[1]["artifact_id"]


def test_analysis_is_indexed_and_linked_to_exact_parent_attempt(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        analysis = _analysis(client, result)
        record = _document(
            client,
            f"/api/v1/provenance/analyses/{analysis['analysis_id']}",
            ProvenanceAnalysis,
        )
        assert record["analysis_id"] == analysis["analysis_id"]
        assert record["task_id"] == result["task_id"]
        assert record["attempt_id"] == result["attempt_id"]
        assert record["evidence_status"] == "verified"
        assert (
            record["source"]["manifest_sha256"]
            == hashlib.sha256((tmp_path / result["artifacts"]["manifest"]).read_bytes()).hexdigest()
        )
        parent = _document(client, _task_url(result), ProvenanceTask)
        assert parent["analyses"] == [record]
        for artifact in record["artifacts"]:
            indexed = _document(
                client,
                f"/api/v1/provenance/artifacts/{artifact['artifact_id']}",
                ProvenanceArtifact,
            )
            assert indexed == artifact
        timeline = _document(client, _timeline_url(result), ProvenanceTimeline)
        assert "analysis_completed" in [event["event_type"] for event in timeline["events"]]


def test_provenance_and_analysis_survive_expired_live_status(tmp_path, monkeypatch):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        analysis = _analysis(client, result)
        before = _document(client, _task_url(result), ProvenanceTask)
        current = task_store._now()
        monkeypatch.setattr(task_store, "_now", lambda: current + 7200)
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").status_code == 404
        assert client.get(f"/api/v1/simulations/{result['task_id']}/envelope").status_code == 404
        after = _document(client, _task_url(result), ProvenanceTask)
        assert after == before
        _document(client, _task_url(result) + "/graph", ProvenanceGraph)
        _document(client, _timeline_url(result), ProvenanceTimeline)
        _document(
            client,
            f"/api/v1/provenance/analyses/{analysis['analysis_id']}",
            ProvenanceAnalysis,
        )
        for artifact in after["artifacts"]:
            _document(
                client,
                f"/api/v1/provenance/artifacts/{artifact['artifact_id']}",
                ProvenanceArtifact,
            )
        assert client.get(_task_url(result) + "/export").status_code == 200


def test_timeline_pages_without_duplicate_or_missing_lifecycle_events(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        whole = _document(client, _timeline_url(result), ProvenanceTimeline)
        events = []
        query = {"limit": 1}
        for _ in range(20):
            response = client.get(_timeline_url(result), params=query)
            assert response.status_code == 200, response.text
            page = response.json()
            ProvenanceTimeline.model_validate_json(response.content)
            assert len(page["events"]) <= 1
            events.extend(page["events"])
            if not page["has_more"]:
                break
            assert page["next_cursor"]
            query["after_id"] = page["next_cursor"]
        else:
            pytest.fail("Timeline cursor did not advance")
        assert events == whole["events"]
        assert len({event["event_id"] for event in events}) == len(events)
        final = client.get(
            _timeline_url(result), params={"after_id": events[-1]["event_id"], "limit": 1}
        ).json()
        assert final["events"] == []
        assert final["has_more"] is False


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 0},
        {"limit": 257},
        {"limit": "true"},
        {"limit": 1.5},
        {"after_id": "invalid"},
        {"after_id": "1-2-3"},
        {"after_id": "1" * 64 + "-0"},
    ],
)
def test_timeline_queries_are_bounded(tmp_path, query):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get("/api/v1/tasks/missing/timeline", params=query)
        assert response.status_code == 422, response.text


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled"])
def test_noncompleted_tasks_retain_observed_history_without_fabricated_evidence(tmp_path, status):
    backend = MemoryTaskStore()
    backend.submit("archived-task", PAYLOAD, {})
    if status == "failed":
        job = backend.claim()
        assert job is not None
        backend.finish(job, {"status": "failed", "error": "Test scientific failure"})
    elif status == "cancelled":
        backend.request_cancel("archived-task")
    settings = Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )
    with TestClient(create_app(settings, backend=backend)) as client:
        result = {"task_id": "archived-task"}
        archived = _document(client, _task_url(result), ProvenanceTask)
        assert archived["status"] == status
        assert archived["evidence_status"] == "unavailable"
        assert archived["artifacts"] == []
        assert archived["manifest_sha256"] is None
        events = _document(client, _timeline_url(result), ProvenanceTimeline)["events"]
        assert events[0]["event_type"] == "submitted"
        assert all(event["event_type"] != "completed" for event in events)
        if status != "pending":
            assert events[-1]["event_type"] == status


@pytest.mark.parametrize("corruption", ["state", "manifest", "envelope", "coherent_rewrite"])
def test_queries_and_export_refuse_changed_committed_evidence(tmp_path, corruption):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        task = _document(client, _task_url(result), ProvenanceTask)
        artifact = task["artifacts"][0]
        if corruption == "state":
            (tmp_path / result["artifacts"]["final_state"]).write_bytes(b"corrupt")
        elif corruption == "envelope":
            (tmp_path / result["task_id"] / result["attempt_id"] / "envelope.json").unlink()
        else:
            manifest_path = tmp_path / result["artifacts"]["manifest"]
            manifest = json.loads(manifest_path.read_text())
            manifest["environment"]["processor"] = "changed-after-commit"
            manifest_path.write_text(json.dumps(manifest))
            if corruption == "coherent_rewrite":
                # All local sidecar digests can agree while the durable archive
                # still identifies a different manifest from terminal commit.
                publish_result_envelope(result, artifacts_dir=tmp_path)
                assert (
                    client.get(f"/api/v1/simulations/{result['task_id']}/envelope").status_code
                    == 200
                )
        for url in (
            _task_url(result),
            _task_url(result) + "/graph",
            _task_url(result) + "/export",
            f"/api/v1/provenance/artifacts/{artifact['artifact_id']}",
        ):
            failure = client.get(url)
            assert failure.status_code == 409, failure.text
            assert failure.json()["detail"]["code"] == "PROVENANCE_INTEGRITY_FAILED"
            assert str(tmp_path) not in failure.text


def test_analysis_corruption_is_not_hidden_by_parent_graph_or_export(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        analysis = _analysis(client, result)
        analysis_path = (
            tmp_path
            / result["task_id"]
            / result["attempt_id"]
            / "analyses"
            / analysis["analysis_id"]
            / "analysis.json"
        )
        analysis_path.write_text("{}")
        for url in (
            f"/api/v1/provenance/analyses/{analysis['analysis_id']}",
            _task_url(result) + "/graph",
            _task_url(result) + "/export",
        ):
            response = client.get(url)
            assert response.status_code == 409, response.text


def test_ro_crate_export_contains_verified_scientific_files_and_metadata(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        analysis = _analysis(client, result)
        response = client.get(_task_url(result) + "/export", params={"format": "ro-crate"})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/zip"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-disposition"].startswith("attachment; filename=")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            assert "ro-crate-metadata.json" in names
            assert len(set(names)) == len(names)
            assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
            metadata = json.loads(archive.read("ro-crate-metadata.json"))
            assert metadata["@context"]
            dataset = next(record for record in metadata["@graph"] if record.get("@id") == "./")
            assert dataset["@type"] == "Dataset"
            assert dataset["license"] == {"@id": "#data-license-unavailable"}
            data_license = next(
                record
                for record in metadata["@graph"]
                if record.get("@id") == "#data-license-unavailable"
            )
            assert data_license["@type"] == "CreativeWork"
            assert "not supplied" in data_license["name"]
            for artifact_name in ("final_state.npz", "analysis.json"):
                path = next(name for name in names if name.endswith("/" + artifact_name))
                if artifact_name == "final_state.npz":
                    source = tmp_path / result["artifacts"]["final_state"]
                else:
                    source = (
                        tmp_path
                        / result["task_id"]
                        / result["attempt_id"]
                        / "analyses"
                        / analysis["analysis_id"]
                        / "analysis.json"
                    )
                assert archive.read(path) == source.read_bytes()
        unsupported = client.get(_task_url(result) + "/export", params={"format": "workflow-run"})
        assert unsupported.status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/provenance/tasks/missing",
        "/api/v1/provenance/tasks/missing/graph",
        "/api/v1/provenance/tasks/missing/export",
        "/api/v1/tasks/missing/timeline",
        "/api/v1/provenance/artifacts/" + "0" * 64,
        "/api/v1/provenance/artifacts/invalid",
        "/api/v1/provenance/analyses/" + "0" * 32,
        "/api/v1/provenance/analyses/invalid",
    ],
)
def test_unknown_or_invalid_archival_resources_are_not_found(tmp_path, path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get(path)
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"]
        assert response.headers["cache-control"] == "no-store"


def test_archival_storage_failure_is_safe(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credential-file secret-value")

    monkeypatch.setattr(ProvenanceService, "task", unavailable)
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get("/api/v1/provenance/tasks/missing")
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "PROVENANCE_STORAGE_UNAVAILABLE"
        assert "private" not in response.text
        assert "secret" not in response.text


def test_lost_analysis_archive_acknowledgement_retains_registered_files(tmp_path, monkeypatch):
    app = create_app(configuration(tmp_path))
    with TestClient(app) as client:
        result = _run(client)
        backend = app.state.backend
        register = backend.register_analysis
        committed = {}

        def register_then_timeout(task_id, attempt_id, analysis_id, record, *, evidence):
            register(task_id, attempt_id, analysis_id, record, evidence=evidence)
            directory = tmp_path / task_id / attempt_id / "analyses" / analysis_id
            committed["analysis_id"] = analysis_id
            committed["files"] = {
                path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
            }
            raise TimeoutError("/private/redis-credentials: acknowledgement lost")

        monkeypatch.setattr(backend, "register_analysis", register_then_timeout)
        response = client.post(
            f"/api/v1/simulations/{result['task_id']}/analyses", json={"bins": 12}
        )
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "ANALYSIS_ARCHIVE_UNAVAILABLE"
        assert "private" not in response.text
        assert "credentials" not in response.text
        assert committed["files"]
        archived = _document(client, _task_url(result), ProvenanceTask)
        assert len(archived["analyses"]) == 1
        assert archived["analyses"][0]["analysis_id"] == committed["analysis_id"]
        directory = (
            tmp_path
            / result["task_id"]
            / result["attempt_id"]
            / "analyses"
            / committed["analysis_id"]
        )
        assert {
            path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
        } == committed["files"]
        analysis = _document(
            client,
            f"/api/v1/provenance/analyses/{committed['analysis_id']}",
            ProvenanceAnalysis,
        )
        assert analysis["evidence_status"] == "verified"


def test_archive_capacity_rejects_new_analysis_before_computation_or_file_writes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(task_store, "MAX_ARCHIVED_ANALYSES", 2)
    monkeypatch.setattr(analysis_endpoints, "MAX_ARCHIVED_ANALYSES", 2)
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = _run(client)
        records = [_analysis(client, result), _analysis(client, result)]
        before = {
            path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

        def forbidden(*args, **kwargs):
            pytest.fail("Archived analysis capacity must be checked before analysis starts")

        monkeypatch.setattr(analysis_service, "analyze_velocities", forbidden)
        response = client.post(
            f"/api/v1/simulations/{result['task_id']}/analyses", json={"bins": 12}
        )
        assert response.status_code == 429, response.text
        assert response.json()["detail"]["code"] == "ANALYSIS_ARCHIVE_LIMIT_EXCEEDED"
        after = {
            path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        assert after == before
        archived = _document(client, _task_url(result), ProvenanceTask)
        assert {record["analysis_id"] for record in archived["analyses"]} == {
            record["analysis_id"] for record in records
        }


def test_export_limit_is_a_structured_413(tmp_path, monkeypatch):
    from scientific_api.services import provenance_export

    def over_limit(*args, **kwargs):
        raise ProvenanceError("PROVENANCE_LIMIT_EXCEEDED", status_code=413)

    monkeypatch.setattr(provenance_export, "export_ro_crate", over_limit)
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get("/api/v1/provenance/tasks/known/export")
        assert response.status_code == 413, response.text
        assert response.json()["detail"]["code"] == "PROVENANCE_LIMIT_EXCEEDED"


def test_openapi_declares_archival_types_and_binary_export(tmp_path):
    schema = create_app(configuration(tmp_path)).openapi()
    for path, model in (
        ("/api/v1/provenance/tasks/{task_id}", "ProvenanceTask"),
        ("/api/v1/provenance/tasks/{task_id}/graph", "ProvenanceGraph"),
        ("/api/v1/provenance/artifacts/{artifact_id}", "ProvenanceArtifact"),
        ("/api/v1/provenance/analyses/{analysis_id}", "ProvenanceAnalysis"),
        ("/api/v1/tasks/{task_id}/timeline", "ProvenanceTimeline"),
    ):
        response = schema["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]
        assert response["schema"]["$ref"] == f"#/components/schemas/{model}"
    exported = schema["paths"]["/api/v1/provenance/tasks/{task_id}/export"]["get"]["responses"][
        "200"
    ]
    assert exported["content"]["application/zip"]["schema"]["format"] == "binary"
