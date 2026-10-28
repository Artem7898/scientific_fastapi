"""Saved-state analysis, provenance, downloads and capacity integration."""

import builtins
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from fastapi.testclient import TestClient

from scientific_api.core import analysis_manager
from scientific_api.main import create_app
from scientific_api.services import analysis_service

from .test_simulations import PAYLOAD, completed, configuration


@pytest.fixture
def saved_run(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
        result = completed(client, accepted["check_url"])
        yield client, result, tmp_path


def endpoint(result):
    return f"/api/v1/simulations/{result['task_id']}/analyses"


def test_json_analysis_uses_only_saved_state_and_preserves_simulation(saved_run, monkeypatch):
    client, result, root = saved_run
    originals = {key: (root / path).read_bytes() for key, path in result["artifacts"].items()}

    def forbidden(*args, **kwargs):
        pytest.fail("Analysis must not execute a simulation")

    monkeypatch.setattr(
        "scientific_api.models.physics_engine.IdealGasSimulator.__init__", forbidden
    )
    response = client.post(endpoint(result), json={"bins": 12})
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["task_id"] == result["task_id"]
    assert record["attempt_id"] == result["attempt_id"]
    assert set(record["artifacts"]) == {"manifest", "analysis"}
    assert response.headers["location"] == record["artifacts"]["manifest"]
    summary_response = client.get(record["artifacts"]["analysis"])
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["sample_size"] == PAYLOAD["num_particles"]
    assert summary["dimensions"] == 2
    assert summary["source"]["state_sha256"] == hashlib.sha256(originals["final_state"]).hexdigest()
    assert summary["source"]["manifest_sha256"] == hashlib.sha256(originals["manifest"]).hexdigest()
    manifest = client.get(record["artifacts"]["manifest"]).json()
    assert manifest["parameters"] == {"bins": 12, "formats": []}
    assert manifest["environment"]["matplotlib"] is None
    for artifact in manifest["artifacts"]:
        data = client.get(record["artifacts"][artifact["name"]]).content
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    for key, path in result["artifacts"].items():
        assert (root / path).read_bytes() == originals[key]
    assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


def test_png_pdf_downloads_and_repeated_analysis_are_isolated(saved_run):
    pytest.importorskip("matplotlib")
    client, result, _ = saved_run
    records = []
    for formats in (["png", "pdf"], []):
        response = client.post(endpoint(result), json={"formats": formats})
        assert response.status_code == 201, response.text
        records.append(response.json())
    assert records[0]["analysis_id"] != records[1]["analysis_id"]
    for name, signature, mime in (
        ("png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("pdf", b"%PDF-", "application/pdf"),
    ):
        response = client.get(records[0]["artifacts"][name])
        assert response.status_code == 200
        assert response.headers["content-type"] == mime
        assert response.content.startswith(signature)
        assert len(response.content) > 1000
    manifest = client.get(records[0]["artifacts"]["manifest"]).json()
    for artifact in manifest["artifacts"]:
        response = client.get(records[0]["artifacts"][artifact["name"]])
        assert hashlib.sha256(response.content).hexdigest() == artifact["sha256"]
    assert (
        client.get(records[1]["artifacts"]["analysis"].removesuffix("analysis") + "pdf").status_code
        == 404
    )


def test_missing_research_does_not_disable_json_analysis(saved_run, monkeypatch):
    client, result, root = saved_run
    original_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("research extra absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    failed = client.post(endpoint(result), json={"formats": ["png"]})
    assert failed.status_code == 503, failed.text
    assert "research" in failed.json()["detail"]
    parent = root / result["task_id"] / result["attempt_id"] / "analyses"
    assert list(parent.iterdir()) == []
    assert client.post(endpoint(result), json={}).status_code == 201


@pytest.mark.parametrize(
    "payload",
    [
        {"bins": True},
        {"bins": "12"},
        {"bins": 1},
        {"bins": 257},
        {"formats": ["svg"]},
        {"formats": ["png", "png"]},
        {"unknown": 1},
    ],
)
def test_analysis_inputs_are_strict(saved_run, payload):
    client, result, _ = saved_run
    assert client.post(endpoint(result), json=payload).status_code == 422


@pytest.mark.parametrize(
    "corruption", ["hash", "identity", "model", "malformed_model", "temperature", "seed", "symlink"]
)
def test_corrupt_or_untrusted_sources_cannot_be_analyzed(saved_run, corruption):
    client, result, root = saved_run
    path = root / result["artifacts"]["manifest"]
    manifest = json.loads(path.read_text())
    if corruption == "hash":
        manifest["artifacts"][0]["sha256"] = "0" * 64
    elif corruption == "identity":
        manifest["attempt_id"] = "other-attempt"
    elif corruption == "model":
        manifest["model"]["dimensions"] = 3
    elif corruption == "malformed_model":
        manifest["model"] = []
    elif corruption == "temperature":
        manifest["parameters"]["initial_temperature"] *= 2
        manifest["model"]["target_temperature"] *= 2
    elif corruption == "seed":
        manifest["parameters"]["seed"] += 1
    else:
        original = root / result["artifacts"]["final_state"]
        outside = root.parent / "outside-state.npz"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
    path.write_text(json.dumps(manifest))
    response = client.post(endpoint(result), json={})
    assert response.status_code == 409, response.text
    assert not (path.parent / "analyses").exists()


def test_render_failure_never_publishes_a_partial_analysis(saved_run, monkeypatch):
    client, result, root = saved_run

    def fail(analysis, *, output_dir, formats):
        (output_dir / "velocity_distribution.png").write_bytes(b"partial png")
        raise RuntimeError("PDF failed")

    monkeypatch.setattr(analysis_service, "render_velocity_report", fail)
    assert client.post(endpoint(result), json={"formats": ["png", "pdf"]}).status_code == 500
    assert list((root / result["task_id"] / result["attempt_id"] / "analyses").iterdir()) == []
    assert client.get(f"/api/v1/simulations/status/{result['task_id']}").json() == result


def test_tampered_analysis_downloads_are_rejected(saved_run):
    client, result, root = saved_run
    record = client.post(endpoint(result), json={}).json()
    directory = root / result["task_id"] / result["attempt_id"] / "analyses" / record["analysis_id"]
    (directory / "analysis.json").write_text('{"changed": true}')
    assert client.get(record["artifacts"]["analysis"]).status_code == 404
    assert client.get(endpoint(result) + "/invalid-id/artifacts/analysis").status_code == 404


@pytest.mark.parametrize("field, value", [("source", None), ("artifacts", ["invalid"])])
def test_malformed_analysis_manifest_returns_not_found(saved_run, field, value):
    client, result, root = saved_run
    record = client.post(endpoint(result), json={}).json()
    path = (
        root
        / result["task_id"]
        / result["attempt_id"]
        / "analyses"
        / record["analysis_id"]
        / "manifest.json"
    )
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    assert client.get(record["artifacts"]["analysis"]).status_code == 404


def test_analysis_capacity_rejects_excess_concurrent_requests(saved_run, monkeypatch):
    client, result, _ = saved_run
    original = analysis_manager.create_velocity_analysis
    entered, release = Event(), Event()
    lock = Lock()
    running = 0

    def blocked(*args, **kwargs):
        nonlocal running
        with lock:
            running += 1
            if running == 2:
                entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis_manager, "create_velocity_analysis", blocked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, endpoint(result), json={}) for _ in range(2)]
        try:
            assert entered.wait(3)
            response = client.post(endpoint(result), json={})
            assert response.status_code == 429
            assert response.headers["retry-after"] == "1"
            assert client.get("/health/live").status_code == 200
        finally:
            release.set()
        assert all(future.result().status_code == 201 for future in futures)


def test_incomplete_simulation_cannot_be_analyzed(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    entered, release = Event(), Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", blocked)
    with TestClient(create_app(configuration(tmp_path))) as client:
        result = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
        try:
            assert entered.wait(2)
            assert client.post(endpoint(result), json={}).status_code == 409
        finally:
            release.set()
        completed(client, result["check_url"])


def test_missing_task_has_no_analysis_endpoint(saved_run):
    client, _, _ = saved_run
    assert client.post("/api/v1/simulations/missing/analyses", json={}).status_code == 404
