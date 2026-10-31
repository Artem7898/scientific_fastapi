"""Asynchronous LJ research reports are pinned, bounded and source-preserving."""

import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from scientific_api.core import research_manager, task_store
from scientific_api.core.config import Settings
from scientific_api.core.research_store import MemoryResearchStore
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app
from scientific_api.schemas.lennard_jones import LennardJonesRequest
from scientific_api.schemas.lj_research import (
    LJResearchAccepted,
    LJResearchList,
    LJResearchRecord,
    LJResearchReport,
)
from scientific_api.schemas.simulation import SimulationRequest
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import (
    publish_result_envelope,
    read_result_envelope,
)
from scientific_api.services.simulation_service import run_simulation

from .test_simulations import PAYLOAD, configuration


def _seed_lj(root, backend, task_id, *, seed=42, cutoff_radius=2.5):
    request = LennardJonesRequest(
        num_particles=16,
        box_size=(8.0, 8.0),
        num_steps=40,
        sample_stride=1,
        seed=seed,
        cutoff_radius=cutoff_radius,
    )
    backend.submit(task_id, {"model": "lennard-jones-2d-v1", **request.model_dump()}, {})
    job = backend.claim()
    assert job is not None
    result = run_simulation(
        request.to_parameters(), task_id=task_id, attempt_id=job.owner, artifacts_dir=root
    )
    envelope = read_result_envelope(result, artifacts_dir=root)
    assert backend.finish(job, result, evidence=build_archive_evidence(envelope))
    return result


def _sources(root, backend, kind):
    groups = {
        "timeseries": [(1, 2.5)],
        "replicas": [(1, 2.5), (2, 2.5)],
        "sensitivity": [(1, 2.0), (2, 2.0), (1, 2.5), (2, 2.5)],
    }[kind]
    return [
        _seed_lj(root, backend, f"lj-source-{index}", seed=seed, cutoff_radius=cutoff)
        for index, (seed, cutoff) in enumerate(groups)
    ]


def _payload(results, kind="timeseries"):
    request = {
        "kind": kind,
        "task_ids": [result["task_id"] for result in results],
        "observables": ["pressure"],
        "bootstrap_samples": 200,
        "bootstrap_seed": 2**64 - 1,
    }
    if kind == "sensitivity":
        request["varied_parameter"] = "cutoff_radius"
    return request


def _submit(client, payload):
    response = client.post("/api/v1/research/lj/analyses", json=payload)
    assert response.status_code == 202, response.text
    LJResearchAccepted.model_validate_json(response.content)
    accepted = response.json()
    assert accepted["kind"] == payload["kind"]
    assert accepted["source_task_ids"] == payload["task_ids"]
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
        LJResearchRecord.model_validate_json(response.content)
        record = response.json()
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    pytest.fail("LJ research analysis did not reach its terminal state")


def _report(client, accepted):
    response = client.get(accepted["check_url"] + "/report")
    assert response.status_code == 200, response.text
    LJResearchReport.model_validate_json(response.content)
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"].startswith("attachment; filename=")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-sha256"] == hashlib.sha256(response.content).hexdigest()
    return response


def _api_settings(root):
    return Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused",
        artifacts_dir=root,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )


def _listing_url(result):
    return f"/api/v1/simulations/{result['task_id']}/research-analyses"


@pytest.mark.parametrize("kind", ["timeseries", "replicas", "sensitivity"])
def test_research_kinds_produce_pinned_reports_without_changing_sources(tmp_path, kind):
    backend = MemoryTaskStore()
    results = _sources(tmp_path, backend, kind)
    source_files = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    archives = {
        result["task_id"]: backend.get_task_archive(result["task_id"]) for result in results
    }
    app = create_app(configuration(tmp_path), backend=backend)
    with TestClient(app) as client:
        accepted = _submit(client, _payload(results, kind))
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        report = _report(client, accepted)
        body = report.json()
        assert body["analysis_id"] == accepted["analysis_id"]
        assert body["kind"] == kind
        assert isinstance(body["statistics"], dict)
        assert body["availability"] == record["availability"]
        if kind == "timeseries":
            # Successful execution does not turn a short, unqualified time
            # series into a confidence interval or claim stationarity.
            metric = body["statistics"]["metrics"]["pressure"]
            assert record["availability"] == "unavailable"
            assert metric["status"] == "unavailable"
            assert metric["confidence_interval"] is None
            assert metric["reason"]
        assert body["source_integrity"] == "verified_against_committed_archive"
        assert body["authenticity"] == "unsigned"
        assert body["limitations"]
        assert body["options"]["bootstrap_seed"] == 2**64 - 1
        pins = {source["identity"]["task_id"]: source for source in body["sources"]}
        assert list(pins) == [result["task_id"] for result in results]
        for result in results:
            pin = pins[result["task_id"]]["identity"]
            assert pin["attempt_id"] == result["attempt_id"]
            assert (
                pin["manifest_sha256"] == archives[result["task_id"]]["evidence"]["manifest_sha256"]
            )
            assert backend.get_task_archive(result["task_id"]) == archives[result["task_id"]]
            assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result
        assert client.get(accepted["check_url"] + "/report").content == report.content
        assert record["report_sha256"] == report.headers["x-content-sha256"]
        listing = client.get(_listing_url(results[0]))
        assert listing.status_code == 200, listing.text
        LJResearchList.model_validate_json(listing.content)
        assert listing.json()["analyses"] == [record]
        for secondary in results[1:]:
            assert client.get(_listing_url(secondary)).json()["analyses"] == []
        assert {
            path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        } == source_files


def test_api_only_analysis_stays_pending_and_report_is_not_ready(tmp_path):
    backend = MemoryTaskStore()
    results = _sources(tmp_path, backend, "timeseries")
    research = MemoryResearchStore()
    with TestClient(
        create_app(_api_settings(tmp_path), backend=backend, research_backend=research)
    ) as client:
        accepted = _submit(client, _payload(results))
        record = client.get(accepted["check_url"])
        assert record.status_code == 200, record.text
        assert record.json()["status"] == "pending"
        assert record.json()["attempts"] == 0
        assert record.json()["report_sha256"] is None
        report = client.get(accepted["check_url"] + "/report")
        assert report.status_code == 409, report.text
        assert report.json()["detail"]["code"] == "RESEARCH_REPORT_NOT_READY"
        queued = research.claim()
        assert queued is not None
        assert queued.task_id == accepted["analysis_id"]


def test_primary_anchor_limit_and_transient_queue_limit_have_distinct_retry_headers(tmp_path):
    backend = MemoryTaskStore()
    result = _seed_lj(tmp_path, backend, "primary-source")
    research = MemoryResearchStore(max_queue_size=10, max_per_source=2)
    with TestClient(
        create_app(_api_settings(tmp_path), backend=backend, research_backend=research)
    ) as client:
        accepted = [_submit(client, _payload([result])), _submit(client, _payload([result]))]
        rejected = client.post("/api/v1/research/lj/analyses", json=_payload([result]))
        assert rejected.status_code == 429, rejected.text
        assert rejected.json()["detail"]["code"] == "RESEARCH_SOURCE_LIMIT_EXCEEDED"
        assert "retry-after" not in rejected.headers
        assert [
            record["analysis_id"] for record in client.get(_listing_url(result)).json()["analyses"]
        ] == [record["analysis_id"] for record in accepted]
    # A separate app/store keeps the queue-limited case independent of the
    # permanent retained-analysis count above.
    backend = MemoryTaskStore()
    first = _seed_lj(tmp_path / "queue", backend, "first-source")
    second = _seed_lj(tmp_path / "queue", backend, "second-source", seed=43)
    research = MemoryResearchStore(max_queue_size=1)
    with TestClient(
        create_app(_api_settings(tmp_path / "queue"), backend=backend, research_backend=research)
    ) as client:
        _submit(client, _payload([first]))
        rejected = client.post("/api/v1/research/lj/analyses", json=_payload([second]))
        assert rejected.status_code == 429, rejected.text
        assert rejected.json()["detail"]["code"] == "RESEARCH_QUEUE_FULL"
        assert rejected.headers["retry-after"] == "1"
        assert client.get(_listing_url(second)).json()["analyses"] == []


@pytest.mark.parametrize("corruption", ["state", "manifest", "coherent_rewrite"])
def test_source_corruption_is_an_execution_failure_not_an_uncertainty_estimate(
    tmp_path, corruption
):
    backend = MemoryTaskStore()
    result = _seed_lj(tmp_path, backend, "corrupted-source")
    manifest_path = tmp_path / result["artifacts"]["manifest"]
    if corruption == "state":
        (tmp_path / result["artifacts"]["final_state"]).write_bytes(b"private /storage/secret")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["environment"]["processor"] = "changed after archive commit"
        manifest_path.write_text(json.dumps(manifest))
        if corruption == "coherent_rewrite":
            publish_result_envelope(result, artifacts_dir=tmp_path)
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        accepted = _submit(client, _payload([result]))
        record = _terminal(client, accepted)
        assert record["status"] == "failed", record
        assert record["report_sha256"] is None
        assert record["error_code"]
        assert "private" not in record["error"]
        assert "secret" not in record["error"]
        assert client.get(accepted["check_url"] + "/report").status_code == 409


def test_reports_remain_immutable_after_source_ttl_and_later_corruption(tmp_path, monkeypatch):
    backend = MemoryTaskStore()
    result = _seed_lj(tmp_path, backend, "retained-source")
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        now = task_store._now()
        monkeypatch.setattr(task_store, "_now", lambda: now + 7200)
        assert client.get(f"/api/v1/simulations/status/{result['task_id']}").status_code == 404
        accepted = _submit(client, _payload([result]))
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        original = _report(client, accepted)
        (tmp_path / result["artifacts"]["trajectory"]).unlink()
        later = _report(client, accepted)
        assert later.content == original.content
        assert later.headers["x-content-sha256"] == original.headers["x-content-sha256"]
        assert client.get(_listing_url(result)).json()["analyses"] == [record]


@pytest.mark.parametrize("status", ["pending", "running", "failed", "cancelled", "completed"])
def test_unavailable_sources_are_refused_before_admission(tmp_path, status):
    backend = MemoryTaskStore()
    backend.submit("unavailable-source", {"model": "lennard-jones-2d-v1", "num_steps": 4}, {})
    if status != "pending":
        job = backend.claim()
        assert job is not None
        if status != "running":
            assert backend.finish(job, {"status": status, "error": "No scientific evidence"})
    with TestClient(create_app(_api_settings(tmp_path), backend=backend)) as client:
        rejected = client.post(
            "/api/v1/research/lj/analyses", json={"task_ids": ["unavailable-source"]}
        )
        assert rejected.status_code == 409, rejected.text
        expected = (
            "RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE"
            if status == "completed"
            else "RESEARCH_SOURCE_NOT_COMPLETED"
        )
        assert rejected.json()["detail"]["code"] == expected


def test_non_lj_source_cannot_be_interpreted_as_interacting_particle_research(tmp_path):
    backend = MemoryTaskStore()
    request = SimulationRequest.model_validate(PAYLOAD)
    backend.submit("ideal-gas-source", request.model_dump(), {})
    job = backend.claim()
    result = run_simulation(
        request.to_parameters(), task_id=job.task_id, attempt_id=job.owner, artifacts_dir=tmp_path
    )
    assert backend.finish(
        job,
        result,
        evidence=build_archive_evidence(read_result_envelope(result, artifacts_dir=tmp_path)),
    )
    with TestClient(create_app(_api_settings(tmp_path), backend=backend)) as client:
        response = client.post("/api/v1/research/lj/analyses", json={"task_ids": [job.task_id]})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "RESEARCH_SOURCE_MODEL_UNSUPPORTED"


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "unknown"},
        {"task_ids": []},
        {"task_ids": ["same", "same"]},
        {"task_ids": ["missing-" + str(index) for index in range(51)]},
        {"kind": "replicas"},
        {"kind": "sensitivity"},
        {"observables": ["temperature"]},
        {"observables": ["pressure", "pressure"]},
        {"bootstrap_samples": 199},
        {"bootstrap_samples": 5001},
        {"bootstrap_samples": True},
        {"bootstrap_seed": -1},
        {"bootstrap_seed": 2**64},
        {"confidence_level": 0.8},
        {"block_length": 0},
        {"block_length": 1002},
        {"assume_stationary": "true"},
        {"burn_in_time": -1.0},
        {"varied_parameter": "cutoff_radius"},
        {"unknown": 1},
    ],
)
def test_research_request_contracts_reject_invalid_or_unused_options(tmp_path, change):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post(
            "/api/v1/research/lj/analyses", json={"task_ids": ["missing"], **change}
        )
        assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/research/lj/analyses"),
        ("get", "/api/v1/research/analyses/" + "0" * 32),
        ("get", "/api/v1/research/analyses/" + "0" * 32 + "/report"),
        ("get", "/api/v1/research/analyses/invalid"),
        ("get", "/api/v1/research/analyses/invalid/report"),
        ("get", "/api/v1/simulations/missing/research-analyses"),
    ],
)
def test_unknown_sources_or_research_ids_have_safe_404s(tmp_path, method, path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = (
            client.post(path, json={"task_ids": ["missing"]})
            if method == "post"
            else client.get(path)
        )
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["code"]
        assert response.headers["cache-control"] == "no-store"


def test_worker_failure_does_not_fabricate_an_uncertainty_report(tmp_path, monkeypatch):
    backend = MemoryTaskStore()
    result = _seed_lj(tmp_path, backend, "execution-source")

    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credential-file secret-value")

    monkeypatch.setattr(research_manager, "compute_lj_research", unavailable)
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        accepted = _submit(client, _payload([result]))
        record = _terminal(client, accepted)
        assert record["status"] == "failed", record
        assert record["report_sha256"] is None
        assert "private" not in record["error"]
        assert "secret" not in record["error"]


def test_report_integrity_is_checked_against_its_own_committed_hash(tmp_path, monkeypatch):
    backend = MemoryTaskStore()
    result = _seed_lj(tmp_path, backend, "report-source")
    app = create_app(configuration(tmp_path), backend=backend)
    with TestClient(app) as client:
        accepted = _submit(client, _payload([result]))
        record = _terminal(client, accepted)
        assert record["status"] == "completed", record
        _report(client, accepted)
        store = app.state.research_manager.backend
        original = store.get

        def changed(identifier):
            saved = original(identifier)
            if saved is not None and identifier == accepted["analysis_id"]:
                saved["report"]["statistics"]["changed_after_commit"] = True
            return saved

        monkeypatch.setattr(store, "get", changed)
        response = client.get(accepted["check_url"] + "/report")
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "RESEARCH_REPORT_INTEGRITY_FAILED"


def test_storage_failures_do_not_leak_research_paths(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/credential-file secret-value")

    monkeypatch.setattr(research_manager.ResearchManager, "get", unavailable)
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.get("/api/v1/research/analyses/" + "0" * 32)
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "RESEARCH_STORAGE_UNAVAILABLE"
        assert "private" not in response.text
        assert "secret" not in response.text


def test_openapi_exposes_research_contracts_and_canonical_report(tmp_path):
    schema = create_app(configuration(tmp_path)).openapi()
    for path, method, status, response_type in (
        ("/api/v1/research/lj/analyses", "post", "202", "LJResearchAccepted"),
        ("/api/v1/research/analyses/{analysis_id}", "get", "200", "LJResearchRecord"),
        ("/api/v1/research/analyses/{analysis_id}/report", "get", "200", "LJResearchReport"),
        ("/api/v1/simulations/{task_id}/research-analyses", "get", "200", "LJResearchList"),
    ):
        actual = schema["paths"][path][method]["responses"][status]["content"]["application/json"]
        assert actual["schema"]["$ref"] == f"#/components/schemas/{response_type}"
