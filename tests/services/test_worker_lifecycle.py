"""Verify API-only admission and independently supervised worker lifecycles."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scientific_api import worker
from scientific_api.core.config import Settings
from scientific_api.core.experiment_tracker import NullTracker
from scientific_api.core.queue_manager import QueueManager
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app

PAYLOAD = {
    "initial_temperature": 1.0,
    "num_particles": 16,
    "num_steps": 5,
    "time_step": 0.01,
    "seed": 42,
}


def configuration(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        **{
            "artifacts_dir": tmp_path,
            "redis_uri": "redis://unused.invalid:6379/0",
            "mlflow_tracking_uri": "",
            "otel_enabled": False,
            "worker_count": 1,
            **overrides,
        },
    )


@pytest.mark.parametrize("mode", ["api", "worker"])
def test_external_roles_require_shared_redis_state(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValidationError, match="requires REDIS_URI"):
        configuration(tmp_path, execution_mode=mode, redis_uri="")
    assert (
        configuration(tmp_path, execution_mode="embedded", redis_uri="").execution_mode
        == "embedded"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"event_poll_interval_seconds": 0},
        {"event_poll_interval_seconds": 5.1},
        {"sse_max_connections": 0},
        {"sse_max_connections": 1025},
    ],
)
def test_event_resource_limits_are_validated(tmp_path: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        configuration(tmp_path, **options)


def test_api_role_accepts_without_claiming_and_readiness_checks_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scientific_api.main as main_module

    def no_unused_tracking(tracking_uri: str, experiment: str) -> NullTracker:
        assert tracking_uri == "", "API-only execution must not initialize an unused MLflow adapter"
        return NullTracker()

    monkeypatch.setattr(main_module, "build_tracker", no_unused_tracking)

    class CountingStore(MemoryTaskStore):
        claims = 0

        def claim(self):
            self.claims += 1
            return super().claim()

    backend = CountingStore()
    app = create_app(
        configuration(tmp_path, execution_mode="api", mlflow_tracking_uri="http://unused.invalid"),
        backend=backend,
    )
    with TestClient(app) as client:
        assert not app.state.queue_manager.ready
        assert client.get("/health/ready").status_code == 200
        response = client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD)
        assert response.status_code == 202
        url = response.json()["check_url"]
        assert client.get(url).json()["status"] == "pending"
        time.sleep(0.1)
        assert client.get(url).json()["status"] == "pending"
        assert backend.claims == 0
        backend.close()
        assert client.get("/health/ready").status_code == 503


@pytest.mark.asyncio
async def test_worker_stops_gracefully_without_opening_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = MemoryTaskStore()
    applications = []

    def factory(settings: Settings):
        assert settings.execution_mode == "worker"
        app = create_app(settings, backend=backend)
        applications.append(app)
        return app

    def no_http_bind(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A standalone worker must not bind a listening socket")

    monkeypatch.setattr(worker, "_create_app", factory)
    monkeypatch.setattr(socket.socket, "bind", no_http_bind)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_worker(configuration(tmp_path), stop_event=stop))
    try:
        async with asyncio.timeout(3):
            while not applications or not applications[0].state.queue_manager.ready:
                await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=3)
        assert not backend.health()
        assert not applications[0].state.queue_manager.ready
    finally:
        stop.set()
        if not task.done():
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_unexpected_worker_loop_exit_fails_process_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = MemoryTaskStore()

    async def broken_loop(self: QueueManager) -> None:
        return

    monkeypatch.setattr(QueueManager, "_worker", broken_loop)
    monkeypatch.setattr(
        worker, "_create_app", lambda settings: create_app(settings, backend=backend)
    )
    with pytest.raises(RuntimeError, match="worker loop stopped unexpectedly"):
        await asyncio.wait_for(
            worker.run_worker(configuration(tmp_path), stop_event=asyncio.Event()), timeout=3
        )
    assert not backend.health()


@pytest.mark.asyncio
async def test_worker_requires_redis_before_creating_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_resources(settings: Settings) -> None:
        raise AssertionError("Invalid external execution must fail before creating resources")

    monkeypatch.setattr(worker, "_create_app", no_resources)
    with pytest.raises(ValueError, match="require REDIS_URI"):
        await worker.run_worker(configuration(tmp_path, redis_uri=""), stop_event=asyncio.Event())


def test_cli_help_is_available_before_loading_environment_settings(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update({"EXECUTION_MODE": "worker", "REDIS_URI": ""})
    result = subprocess.run(
        [sys.executable, "-m", "scientific_api.worker", "--help"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no HTTP" in result.stdout


def test_cli_configuration_error_does_not_echo_credentials(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_MODE": "worker",
            "REDIS_URI": "",
            "MLFLOW_TRACKING_URI": "http://private-credential-must-not-appear/",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "scientific_api.worker"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "REDIS_URI" in result.stderr
    assert "private-credential-must-not-appear" not in result.stdout + result.stderr


@pytest.mark.skipif(
    not os.environ.get("SCIENTIFIC_TEST_REDIS_URI"),
    reason="SCIENTIFIC_TEST_REDIS_URI is not configured",
)
@pytest.mark.parametrize("backend_name", ["numpy", "numba"])
@pytest.mark.parametrize("model_version", ["ideal-gas-ou-v1", "langevin-free-v1"])
def test_standalone_worker_process_consumes_api_jobs_and_handles_sigterm(
    tmp_path: Path, backend_name: str, model_version: str
) -> None:
    if backend_name == "numba":
        pytest.importorskip("numba")
    payload = {**PAYLOAD, "backend": backend_name}
    endpoint = "/api/v1/simulations/monte-carlo"
    if model_version == "langevin-free-v1":
        payload.update(
            bath_temperature=1.0,
            sample_stride=1,
            acceleration=[0.5, -0.25],
            boundary="periodic",
            box_size=[1.0, 1.0],
        )
        endpoint = "/api/v1/simulations/langevin"
    redis = pytest.importorskip("redis")
    uri = os.environ["SCIENTIFIC_TEST_REDIS_URI"]
    namespace = "scientific-api-external-worker-" + uuid4().hex
    settings = configuration(
        tmp_path, execution_mode="api", redis_uri=uri, redis_namespace=namespace
    )
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_MODE": "worker",
            "REDIS_URI": uri,
            "REDIS_NAMESPACE": namespace,
            "ARTIFACTS_DIR": str(tmp_path),
            "WORKER_COUNT": "1",
            "MLFLOW_TRACKING_URI": "",
            "OTEL_ENABLED": "false",
        }
    )
    process = None
    try:
        with TestClient(create_app(settings)) as client:
            assert not client.app.state.queue_manager.ready
            submitted = client.post(endpoint, json=payload)
            assert submitted.status_code == 202, submitted.text
            accepted = submitted.json()
            assert client.get(accepted["check_url"]).json()["status"] == "pending"
            pending = client.app.state.backend.events(accepted["task_id"])
            assert pending["snapshot"]["phase"] == "pending"
            pending_cursor = pending["last_event_id"]
            process = subprocess.Popen(
                [sys.executable, "-m", "scientific_api.worker"],
                env=environment,
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None, process.communicate(timeout=2)[0]
                response = client.get(accepted["check_url"])
                assert response.status_code == 200, response.text
                result = response.json()
                if result["status"] == "completed":
                    break
                assert result["status"] in {"pending", "running"}, result
                time.sleep(0.02)
            else:
                pytest.fail("Standalone worker did not complete its queued API task")
            assert result["task_id"] == accepted["task_id"]
            artifact = client.get(f"/api/v1/simulations/{result['task_id']}/artifacts/manifest")
            assert artifact.status_code == 200
            assert artifact.json()["outputs"] == result
            assert artifact.json()["parameters"]["backend"] == backend_name
            assert artifact.json()["execution"]["backend"] == backend_name
            assert artifact.json()["model"]["version"] == model_version
            if model_version == "langevin-free-v1":
                transport = client.get(
                    f"/api/v1/simulations/{result['task_id']}/artifacts/transport"
                )
                assert transport.status_code == 200
                assert transport.json()["centered_msd"][-1] == result["final_centered_msd"]

            # A fresh API instance reads durable events and results produced by
            # the separate worker process, rather than relying on local state.
            with TestClient(create_app(settings)) as second:
                replay = second.get(
                    f"/api/v1/simulations/{result['task_id']}/events",
                    headers={"Last-Event-ID": pending_cursor},
                )
                assert replay.status_code == 200
                assert replay.headers["content-type"].startswith("text/event-stream")
                events = [
                    json.loads(line.removeprefix("data: "))
                    for line in replay.text.splitlines()
                    if line.startswith("data: ")
                ]
                phases = [event["phase"] for event in events]
                assert {"computing", "saving_artifacts", "completed"} <= set(phases)
                assert phases[-1] == "completed"
                assert second.get(accepted["check_url"]).json() == result

                submission = second.post(
                    "/api/v1/ensembles",
                    json={
                        "model": model_version,
                        "parameters": {
                            key: value for key, value in payload.items() if key != "seed"
                        },
                        "replicas": 3,
                        "base_seed": 42,
                    },
                )
                assert submission.status_code == 202, submission.text
                ensemble = submission.json()
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    response = second.get(ensemble["check_url"])
                    assert response.status_code == 200, response.text
                    aggregate = response.json()
                    if aggregate["status"] == "completed":
                        break
                    assert aggregate["status"] in {"pending", "running"}, aggregate
                    time.sleep(0.02)
                else:
                    pytest.fail("Standalone worker did not finish the independent replicas")
                assert aggregate["counts"]["completed"] == 3
                manifest = second.get(f"/api/v1/ensembles/{ensemble['ensemble_id']}/manifest")
                assert manifest.status_code == 200
                assert manifest.json()["outputs"] == aggregate
                contributions = [child["result"] for child in aggregate["children"]]
                assert len({child["attempt_id"] for child in contributions}) == 3
                for child in contributions:
                    committed = second.get(
                        f"/api/v1/simulations/{child['task_id']}/artifacts/manifest"
                    )
                    assert committed.status_code == 200
                    assert committed.json()["outputs"] == child
                    assert committed.json()["parameters"]["backend"] == backend_name
                    assert committed.json()["execution"]["backend"] == backend_name
                    assert committed.json()["model"]["version"] == model_version
            process.terminate()
            output, _ = process.communicate(timeout=10)
            assert process.returncode == 0, output
            assert "Stopping standalone scientific workers" in output
            # Closing the independent Redis client in the worker must not close
            # the API's client or start an API-local consumer.
            assert client.get("/health/ready").status_code == 200
            assert not client.app.state.queue_manager.ready
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        connection = redis.Redis.from_url(uri)
        try:
            keys = list(connection.scan_iter(match=namespace + ":*"))
            if keys:
                connection.delete(*keys)
        finally:
            connection.close()
