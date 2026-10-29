"""API contracts, resource lifecycle and reproducible artifact integration."""

import hashlib
import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app

PAYLOAD = {
    "initial_temperature": 300.0,
    "num_particles": 128,
    "time_step": 0.01,
    "num_steps": 10,
    "seed": 42,
}


def configuration(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        artifacts_dir=tmp_path,
        redis_uri="",
        mlflow_tracking_uri="",
        otel_enabled=False,
        **overrides,
    )


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(create_app(configuration(tmp_path))) as connection:
        yield connection


def completed(client: TestClient, url: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] in {"pending", "running", "completed"}, result
        if result["status"] == "completed":
            return result
        time.sleep(0.01)
    pytest.fail("Task did not complete before its deadline")


def test_simulation_async_workflow(client, tmp_path):
    response = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD)
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "pending"
    assert response.headers["location"] == accepted["check_url"]
    result = completed(client, accepted["check_url"])
    assert result["task_id"] == accepted["task_id"]
    assert result["model_version"] == "ideal-gas-ou-v1"
    assert result["seed"] == PAYLOAD["seed"]
    assert all(
        np.isfinite(result[name])
        for name in ("final_temperature", "total_energy", "temperature_standard_error")
    )
    manifest_response = client.get(f"/api/v1/simulations/{accepted['task_id']}/artifacts/manifest")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["parameters"] == {**PAYLOAD, "backend": "numpy"}
    assert manifest["outputs"] == result
    state_response = client.get(f"/api/v1/simulations/{accepted['task_id']}/artifacts/final_state")
    assert state_response.status_code == 200
    assert hashlib.sha256(state_response.content).hexdigest() == manifest["artifacts"][0]["sha256"]
    with np.load(tmp_path / result["artifacts"]["final_state"]) as state:
        energy = float(0.5 * np.sum(state["velocities"] ** 2))
    assert energy == pytest.approx(result["total_energy"])


def test_same_seed_repeats_numerical_result(client):
    results = []
    for _ in range(2):
        accepted = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
        results.append(completed(client, accepted["check_url"]))
    assert results[0]["task_id"] != results[1]["task_id"]
    for key in ("final_temperature", "total_energy", "temperature_standard_error"):
        assert results[0][key] == results[1][key]


@pytest.mark.parametrize(
    "changed",
    [
        {"initial_temperature": "300"},
        {"initial_temperature": -10.0},
        {"initial_temperature": True},
        {"num_particles": True},
        {"num_particles": 0},
        {"time_step": 0.0},
        {"num_steps": 0},
        {"num_steps": 10_001},
        {"num_steps": 1000, "num_particles": 1_000_000},
        {"seed": -1},
        {"seed": True},
        {"temprature": 300.0},
    ],
)
def test_invalid_inputs_rejected(client, changed):
    response = client.post("/api/v1/simulations/monte-carlo", json={**PAYLOAD, **changed})
    assert response.status_code == 422, response.text


def test_json_overflow_is_rejected_with_json_error(client):
    raw = json.dumps(PAYLOAD).replace('"time_step": 0.01', '"time_step": 1e309')
    response = client.post(
        "/api/v1/simulations/monte-carlo",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_repeated_lifespan_creates_fresh_workers(tmp_path):
    application = create_app(configuration(tmp_path))
    for _ in range(2):
        with TestClient(application) as client:
            assert client.get("/health/ready").status_code == 200
            accepted = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD)
            assert accepted.status_code == 202
            completed(client, accepted.json()["check_url"])


def test_health_and_missing_artifacts(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    assert client.get("/api/v1/simulations/status/missing").status_code == 404
    assert client.get("/api/v1/simulations/missing/artifacts/manifest").status_code == 404


def test_backpressure_and_active_artifact_conflict(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    entered, release = Event(), Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3), "Test must release its worker"
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", blocked)
    with TestClient(
        create_app(configuration(tmp_path, max_queue_size=1, worker_count=1))
    ) as client:
        try:
            response = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD)
            accepted = response.json()
            assert entered.wait(2)
            assert client.get(accepted["check_url"]).json()["status"] == "running"
            assert (
                client.get(
                    f"/api/v1/simulations/{accepted['task_id']}/artifacts/manifest"
                ).status_code
                == 409
            )
            full = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD)
            assert full.status_code == 429
            assert full.headers["retry-after"] == "1"
        finally:
            release.set()
        completed(client, accepted["check_url"])


def test_artifact_symlink_cannot_escape_root(client, tmp_path):
    accepted = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
    result = completed(client, accepted["check_url"])
    artifact = tmp_path / result["artifacts"]["manifest"]
    secret = tmp_path.parent / "outside-manifest.json"
    secret.write_text("private")
    artifact.unlink()
    artifact.symlink_to(secret)
    response = client.get(f"/api/v1/simulations/{accepted['task_id']}/artifacts/manifest")
    assert response.status_code == 404


def test_failed_calculation_does_not_report_completed(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    def fail(*args, **kwargs):
        raise FloatingPointError("Numerical instability")

    monkeypatch.setattr(queue_manager, "run_simulation", fail)
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = client.get(accepted["check_url"])
            assert response.status_code == 200
            result = response.json()
            if result["status"] == "failed":
                assert result["task_id"] == accepted["task_id"]
                assert "nonfinite" in result["error"]
                return
            time.sleep(0.01)
        pytest.fail("Numerical failure was not persisted")


def test_backend_unavailable_is_not_ready(tmp_path):
    backend = MemoryTaskStore()
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        backend.close()
        assert client.get("/health/ready").status_code == 503
        assert client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).status_code == 503


def test_configured_work_limit_enforced(tmp_path):
    with TestClient(
        create_app(
            configuration(
                tmp_path,
                max_simulation_iterations=2,
                max_particle_steps=128,
            )
        )
    ) as client:
        assert client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).status_code == 422
