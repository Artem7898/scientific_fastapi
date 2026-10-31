"""Research workers supervise only their own pool and open no HTTP listener."""

import asyncio
import os
import socket
import subprocess
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scientific_api import research_worker
from scientific_api.core.config import Settings
from scientific_api.core.research_manager import ResearchManager
from scientific_api.core.research_store import MemoryResearchStore
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.verification_store import MemoryVerificationStore
from scientific_api.main import create_app


def settings(root, **kwargs):
    values: dict[str, Any] = {
        "_env_file": None,
        "artifacts_dir": root,
        "redis_uri": "redis://unused.invalid:6379/0",
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
        **kwargs,
    }
    return Settings(**values)


@pytest.mark.asyncio
async def test_research_cli_starts_no_simulation_verification_consumers_or_http(
    tmp_path, monkeypatch
):
    source = MemoryTaskStore()
    source.submit("simulation", {"initial_temperature": 1.0, "num_particles": 8}, {})
    verification = MemoryVerificationStore()
    verification.submit("verification", {"source_id": "simulation"}, {})
    research = MemoryResearchStore()
    applications = []

    def factory(configuration):
        assert configuration.execution_mode == "research-worker"
        app = create_app(
            configuration,
            backend=source,
            verification_backend=verification,
            research_backend=research,
        )
        applications.append(app)
        return app

    def no_http(*args, **kwargs):
        raise AssertionError("A research worker must not bind an HTTP listener")

    monkeypatch.setattr(research_worker, "_create_app", factory)
    monkeypatch.setattr(socket.socket, "bind", no_http)
    stop = asyncio.Event()
    process = asyncio.create_task(research_worker.run_worker(settings(tmp_path), stop_event=stop))
    try:
        async with asyncio.timeout(3):
            while not applications or not applications[0].state.research_manager.ready:
                await asyncio.sleep(0.01)
        assert not applications[0].state.queue_manager.ready
        assert not applications[0].state.verification_manager.ready
        assert source.get("simulation")["status"] == "pending"
        assert verification.get("verification")["status"] == "pending"
        assert verification.get("verification")["attempts"] == 0
        stop.set()
        await asyncio.wait_for(process, timeout=3)
        assert not source.health()
        assert not verification.health()
        assert not research.health()
    finally:
        stop.set()
        if not process.done():
            await asyncio.wait_for(process, timeout=3)


@pytest.mark.asyncio
async def test_unexpected_research_consumer_exit_fails_cli_supervisor(tmp_path, monkeypatch):
    source = MemoryTaskStore()
    verification = MemoryVerificationStore()
    research = MemoryResearchStore()

    async def broken(self):
        return

    monkeypatch.setattr(ResearchManager, "_worker", broken)
    monkeypatch.setattr(
        research_worker,
        "_create_app",
        lambda configuration: create_app(
            configuration,
            backend=source,
            verification_backend=verification,
            research_backend=research,
        ),
    )
    with pytest.raises(RuntimeError, match="research.*worker.*stopped unexpectedly"):
        await asyncio.wait_for(
            research_worker.run_worker(settings(tmp_path), stop_event=asyncio.Event()), timeout=3
        )
    assert not source.health() and not verification.health() and not research.health()


def test_readiness_tracks_research_pool_storage_in_embedded_role(tmp_path):
    source = MemoryTaskStore()
    verification = MemoryVerificationStore()
    research = MemoryResearchStore()
    application = create_app(
        settings(tmp_path),
        backend=source,
        verification_backend=verification,
        research_backend=research,
    )
    with TestClient(application) as client:
        assert client.get("/health/ready").status_code == 200
        research.close()
        assert client.get("/health/ready").status_code == 503


def test_research_worker_role_requires_shared_storage():
    values: dict[str, Any] = {
        "_env_file": None,
        "execution_mode": "research-worker",
        "redis_uri": "",
    }
    with pytest.raises(ValidationError, match="requires REDIS_URI"):
        Settings(**values)


def test_help_does_not_initialize_settings_or_optional_packages(tmp_path):
    environment = {**os.environ, "EXECUTION_MODE": "research-worker", "REDIS_URI": ""}
    result = subprocess.run(
        [sys.executable, "-m", "scientific_api.research_worker", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "research" in result.stdout
