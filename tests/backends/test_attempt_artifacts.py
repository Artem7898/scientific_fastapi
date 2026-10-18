"""A reclaimed worker cannot replace the committed attempt's scientific record."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.main import create_app
from scientific_api.services.simulation_service import run_simulation


def test_late_obsolete_worker_cannot_overwrite_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_000.0]
    monkeypatch.setattr(task_store, "_now", lambda: clock[0])
    backend = MemoryTaskStore(lease_seconds=1)
    task_id = "attempt-artifact-regression"
    parameters = SimulationParameters(
        initial_temperature=300.0,
        num_particles=32,
        num_steps=2,
        seed=9,
    )
    backend.submit(task_id, {"seed": 9}, {})
    obsolete = backend.claim()
    assert obsolete is not None
    clock[0] += 2
    winner = backend.claim()
    assert winner is not None and winner.owner != obsolete.owner

    committed = run_simulation(
        parameters,
        task_id=task_id,
        attempt_id=winner.owner,
        artifacts_dir=tmp_path,
    )
    assert backend.finish(winner, committed)
    paths = {name: tmp_path / relative for name, relative in committed["artifacts"].items()}
    published = {name: path.read_bytes() for name, path in paths.items()}

    # Simulate a process resuming after losing its lease. It may still publish
    # its own files, but neither the winner's directory nor selected result.
    late = run_simulation(
        parameters,
        task_id=task_id,
        attempt_id=obsolete.owner,
        artifacts_dir=tmp_path,
    )
    assert not backend.finish(obsolete, late)
    assert backend.get(task_id) == committed
    assert late["artifacts"] != committed["artifacts"]
    assert all(path.read_bytes() == published[name] for name, path in paths.items())
    manifest = json.loads(published["manifest"])
    assert manifest["outputs"]["attempt_id"] == winner.owner

    settings_values: dict[str, Any] = {
        "_env_file": None,
        "artifacts_dir": tmp_path,
        "redis_uri": "",
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
    }
    settings = Settings(**settings_values)
    with TestClient(create_app(settings, backend=backend)) as client:
        response = client.get(f"/api/v1/simulations/status/{task_id}")
        assert response.status_code == 200
        assert response.json() == committed
        for name, content in published.items():
            download = client.get(f"/api/v1/simulations/{task_id}/artifacts/{name}")
            assert download.status_code == 200
            assert download.content == content
