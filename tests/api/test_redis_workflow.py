"""Redis-backed API state survives cooperative worker restarts."""

import os
import time
import uuid
from threading import Event

import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.main import create_app

URI = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
pytestmark = pytest.mark.skipif(not URI, reason="SCIENTIFIC_TEST_REDIS_URI is not configured")
PAYLOAD = {
    "initial_temperature": 300.0,
    "num_particles": 64,
    "num_steps": 10,
    "time_step": 0.01,
    "seed": 17,
}


@pytest.fixture
def settings(tmp_path):
    redis = pytest.importorskip("redis")
    namespace = "scientific-api-integration-" + uuid.uuid4().hex
    configuration = Settings(
        _env_file=None,
        redis_uri=URI,
        redis_namespace=namespace,
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
        worker_count=1,
    )
    yield configuration
    connection = redis.Redis.from_url(URI, decode_responses=True)
    try:
        keys = list(connection.scan_iter(match=namespace + ":*"))
        if keys:
            connection.delete(*keys)
    finally:
        connection.close()


def wait_completed(client, url):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] == "completed":
            return record
        assert record["status"] in {"pending", "running"}
        time.sleep(0.02)
    pytest.fail("Redis task did not complete")


@pytest.mark.parametrize("model", ["ideal", "langevin", "lj"])
def test_redis_task_requeued_on_shutdown_finishes_in_new_app(settings, monkeypatch, model):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    entered = Event()

    def interrupted(*args, **kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(3), "Lifespan must request cancellation"
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", interrupted)
    payload = dict(PAYLOAD)
    endpoint = "/api/v1/simulations/monte-carlo"
    if model == "langevin":
        payload.update(bath_temperature=1.0, sample_stride=2, acceleration=[0.5, -0.25])
        endpoint = "/api/v1/simulations/langevin"
    elif model == "lj":
        payload = {
            "num_particles": 36,
            "box_size": [8.0, 8.0],
            "initial_temperature": 0.5,
            "time_step": 0.002,
            "num_steps": 12,
            "sample_stride": 3,
            "seed": PAYLOAD["seed"],
        }
        endpoint = "/api/v1/simulations/lennard-jones"
    with TestClient(create_app(settings)) as first:
        accepted = first.post(endpoint, json=payload).json()
        assert entered.wait(2)
        assert first.get(accepted["check_url"]).json()["status"] == "running"
    monkeypatch.setattr(queue_manager, "run_simulation", original)
    with TestClient(create_app(settings)) as second:
        result = wait_completed(second, accepted["check_url"])
        assert result["task_id"] == accepted["task_id"]
        manifest = second.get(f"/api/v1/simulations/{accepted['task_id']}/artifacts/manifest")
        assert manifest.status_code == 200
        assert manifest.json()["parameters"]["seed"] == PAYLOAD["seed"]
        if model == "langevin":
            assert result["model_version"] == "langevin-free-v1"
            transport = second.get(f"/api/v1/simulations/{accepted['task_id']}/artifacts/transport")
            assert transport.status_code == 200
            assert transport.json()["diffusion"] == result["diffusion"]
        elif model == "lj":
            assert result["model_version"] == "lennard-jones-2d-v1"
            observables = second.get(
                f"/api/v1/simulations/{accepted['task_id']}/artifacts/observables"
            )
            assert observables.status_code == 200
            assert (
                observables.json()["series"]["potential_energy"][-1] == result["potential_energy"]
            )


def test_polling_across_two_app_instances(settings):
    with TestClient(create_app(settings)) as first, TestClient(create_app(settings)) as second:
        accepted = first.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).json()
        result = wait_completed(second, accepted["check_url"])
        assert result["task_id"] == accepted["task_id"]
        assert first.get(accepted["check_url"]).json() == result
