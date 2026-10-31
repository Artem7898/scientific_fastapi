"""The verification CLI consumes its own queue and opens no HTTP listener."""

import asyncio
import os
import socket
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scientific_api import verification_worker
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.verification_manager import VerificationManager
from scientific_api.core.verification_store import MemoryVerificationStore
from scientific_api.main import create_app


def settings(root, **kwargs):
    return Settings(
        _env_file=None,
        artifacts_dir=root,
        redis_uri="redis://unused.invalid:6379/0",
        mlflow_tracking_uri="",
        otel_enabled=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_verification_cli_does_not_start_simulation_consumers_or_http(tmp_path, monkeypatch):
    source = MemoryTaskStore()
    source.submit("simulation", {"num_particles": 8, "initial_temperature": 1.0}, {})
    backend = MemoryVerificationStore()
    applications = []

    def factory(configuration):
        assert configuration.execution_mode == "verification-worker"
        app = create_app(configuration, backend=source, verification_backend=backend)
        applications.append(app)
        return app

    def no_http(*args, **kwargs):
        raise AssertionError("A verification worker must not bind an HTTP listener")

    monkeypatch.setattr(verification_worker, "_create_app", factory)
    monkeypatch.setattr(socket.socket, "bind", no_http)
    stop = asyncio.Event()
    process = asyncio.create_task(
        verification_worker.run_worker(settings(tmp_path), stop_event=stop)
    )
    try:
        async with asyncio.timeout(3):
            while not applications or not applications[0].state.verification_manager.ready:
                await asyncio.sleep(0.01)
        assert not applications[0].state.queue_manager.ready
        assert source.get("simulation")["status"] == "pending"
        stop.set()
        await asyncio.wait_for(process, timeout=3)
        assert not source.health() and not backend.health()
    finally:
        stop.set()
        if not process.done():
            await asyncio.wait_for(process, timeout=3)


@pytest.mark.asyncio
async def test_unexpected_verification_consumer_exit_fails_supervisor(tmp_path, monkeypatch):
    source = MemoryTaskStore()
    backend = MemoryVerificationStore()

    async def broken(self):
        return

    monkeypatch.setattr(VerificationManager, "_worker", broken)
    monkeypatch.setattr(
        verification_worker,
        "_create_app",
        lambda configuration: create_app(
            configuration, backend=source, verification_backend=backend
        ),
    )
    with pytest.raises(RuntimeError, match="verification worker loop stopped unexpectedly"):
        await asyncio.wait_for(
            verification_worker.run_worker(settings(tmp_path), stop_event=asyncio.Event()),
            timeout=3,
        )
    assert not source.health() and not backend.health()


def test_readiness_checks_verification_pool_in_embedded_mode(tmp_path):
    source = MemoryTaskStore()
    backend = MemoryVerificationStore()
    app = create_app(settings(tmp_path), backend=source, verification_backend=backend)
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        backend.close()
        assert client.get("/health/ready").status_code == 503


def test_verification_role_requires_shared_storage():
    with pytest.raises(ValidationError, match="requires REDIS_URI"):
        Settings(_env_file=None, execution_mode="verification-worker", redis_uri="")


def test_help_does_not_initialize_settings_or_optional_packages(tmp_path):
    environment = {**os.environ, "EXECUTION_MODE": "verification-worker", "REDIS_URI": ""}
    result = subprocess.run(
        [sys.executable, "-m", "scientific_api.verification_worker", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "verification" in result.stdout
