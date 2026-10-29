"""Backend admission is explicit, durable and independent of optional imports."""

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore, RedisTaskStore
from scientific_api.domain.experiments import parameters_from_payload
from scientific_api.main import create_app
from scientific_api.models import computation
from scientific_api.services.simulation_service import run_simulation

from .test_simulations import PAYLOAD, configuration


def api_configuration(tmp_path: Path, mode: str) -> Settings:
    return Settings(
        _env_file=None,
        execution_mode=mode,
        redis_uri="redis://unused:6379/0" if mode == "api" else "",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )


def wait_for_terminal(client: TestClient, url: str) -> dict:
    # The initial optional-backend compilation belongs to the first worker
    # invocation and may exceed the regular small-NumPy-run deadline.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("Backend run did not reach a terminal state")


@pytest.mark.parametrize("mode", ["embedded", "api"])
@pytest.mark.parametrize("ensemble", [False, True])
def test_missing_backend_is_rejected_before_admission(tmp_path, monkeypatch, mode, ensemble):
    monkeypatch.setattr(computation, "find_spec", lambda name: None)
    backend = MemoryTaskStore()
    application = create_app(api_configuration(tmp_path, mode), backend=backend)
    with TestClient(application) as client:
        if ensemble:
            parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
            response = client.post(
                "/api/v1/ensembles",
                json={"parameters": {**parameters, "backend": "numba"}, "replicas": 2},
            )
        else:
            response = client.post(
                "/api/v1/simulations/monte-carlo", json={**PAYLOAD, "backend": "numba"}
            )
        assert response.status_code == 503, response.text
        assert "scientific-api[numba]" in response.json()["detail"]
        assert backend.claim() is None


@pytest.mark.parametrize("backend_name", ["jax", "auto", "Numba", True, None])
@pytest.mark.parametrize("ensemble", [False, True])
def test_unknown_backend_is_rejected_by_the_transport_contract(tmp_path, backend_name, ensemble):
    with TestClient(create_app(configuration(tmp_path))) as client:
        if ensemble:
            parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
            response = client.post(
                "/api/v1/ensembles",
                json={"parameters": {**parameters, "backend": backend_name}, "replicas": 2},
            )
        else:
            response = client.post(
                "/api/v1/simulations/monte-carlo",
                json={**PAYLOAD, "backend": backend_name},
            )
        assert response.status_code == 422, response.text


def test_api_role_persists_selected_backend_and_seeds_without_importing_numba(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(computation, "find_spec", lambda name: object())
    backend = MemoryTaskStore()
    application = create_app(api_configuration(tmp_path, "api"), backend=backend)
    with TestClient(application) as client:
        accepted = client.post(
            "/api/v1/simulations/monte-carlo",
            json={**PAYLOAD, "seed": 2**64 - 1, "backend": "numba"},
        )
        assert accepted.status_code == 202, accepted.text
        job = backend.claim()
        assert job is not None
        assert job.payload["backend"] == "numba"
        assert job.payload["seed"] == 2**64 - 1

        parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
        payload = {"parameters": {**parameters, "backend": "numba"}, "replicas": 2}
        headers = {"Idempotency-Key": "selected-backend"}
        ensemble = client.post("/api/v1/ensembles", json=payload, headers=headers)
        assert ensemble.status_code == 202, ensemble.text
        repeated = client.post("/api/v1/ensembles", json=payload, headers=headers)
        assert repeated.json() == ensemble.json()
        changed = client.post(
            "/api/v1/ensembles",
            json={**payload, "parameters": {**parameters, "backend": "numpy"}},
            headers=headers,
        )
        assert changed.status_code == 409
        identifier = ensemble.json()["ensemble_id"]
        snapshot = backend.get_ensemble(identifier)
        assert snapshot["metadata"]["request"]["parameters"]["backend"] == "numba"
        children = [backend.claim(), backend.claim()]
        assert all(child is not None for child in children)
        assert [child.payload["seed"] for child in children] == snapshot["metadata"]["seeds"]
        assert all(child.payload["backend"] == "numba" for child in children)


def test_api_import_does_not_load_optional_computational_libraries():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scientific_api.main import app; "
            "assert app is not None; assert 'numba' not in sys.modules; "
            "assert 'jax' not in sys.modules; assert 'llvmlite' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_numba_http_run_repeats_and_saved_state_can_be_analysed(tmp_path):
    pytest.importorskip("numba")
    values = []
    with TestClient(create_app(configuration(tmp_path))) as client:
        for backend_name in ("numpy", "numba", "numba"):
            accepted = client.post(
                "/api/v1/simulations/monte-carlo",
                json={**PAYLOAD, "seed": 2**64 - 1, "backend": backend_name},
            )
            assert accepted.status_code == 202, accepted.text
            result = wait_for_terminal(client, accepted.json()["check_url"])
            assert result["status"] == "completed", result
            manifest = client.get(
                f"/api/v1/simulations/{result['task_id']}/artifacts/manifest"
            ).json()
            assert manifest["parameters"]["backend"] == backend_name
            assert manifest["parameters"]["seed"] == 2**64 - 1
            assert manifest["execution"]["backend"] == backend_name
            assert manifest["rng"]["bit_generator"] == "PCG64"
            with np.load(tmp_path / result["artifacts"]["final_state"]) as state:
                values.append((result, state["velocities"].copy()))
        np.testing.assert_allclose(values[0][1], values[1][1], rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(values[1][1], values[2][1])
        assert values[1][0]["final_temperature"] == values[2][0]["final_temperature"]
        result = values[1][0]
        analysis = client.post(
            f"/api/v1/simulations/{result['task_id']}/analyses", json={"bins": 12}
        )
        assert analysis.status_code == 201, analysis.text
        summary = client.get(analysis.json()["artifacts"]["analysis"]).json()
        assert summary["sample_size"] == PAYLOAD["num_particles"]


def test_numba_ensemble_uses_persisted_backend_for_every_replica(tmp_path):
    pytest.importorskip("numba")
    parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post(
            "/api/v1/ensembles",
            json={
                "parameters": {**parameters, "backend": "numba"},
                "replicas": 3,
                "base_seed": 2**64 - 1,
            },
        )
        assert accepted.status_code == 202, accepted.text
        result = wait_for_terminal(client, accepted.json()["check_url"])
        assert result["status"] == "completed", result
        ensemble_manifest = client.get(
            f"/api/v1/ensembles/{accepted.json()['ensemble_id']}/manifest"
        ).json()
        assert ensemble_manifest["request"]["parameters"]["backend"] == "numba"
        assert ensemble_manifest["seeds"] == [child["seed"] for child in result["children"]]
        temperatures = []
        for child in result["children"]:
            manifest = client.get(
                f"/api/v1/simulations/{child['task_id']}/artifacts/manifest"
            ).json()
            assert manifest["parameters"]["backend"] == "numba"
            assert manifest["parameters"]["seed"] == child["seed"]
            assert manifest["execution"]["backend"] == "numba"
            temperatures.append(child["result"]["final_temperature"])
        assert result["statistics"]["final_temperature"]["mean"] == math.fsum(temperatures) / 3


@pytest.mark.parametrize("storage", ["memory", "redis"])
@pytest.mark.parametrize("backend_name", ["numpy", "numba"])
@pytest.mark.parametrize("model", ["ideal", "lj"])
def test_retry_preserves_backend_seed_and_fences_obsolete_results(
    tmp_path, monkeypatch, storage, backend_name, model
):
    if backend_name == "numba":
        pytest.importorskip("numba")
    connection = None
    clock = [1000.0]
    namespace = "scientific-compute-retry-" + uuid4().hex
    if storage == "redis":
        uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
        if not uri:
            pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
        redis = pytest.importorskip("redis")
        connection = redis.Redis.from_url(uri)
        store = RedisTaskStore(uri, namespace=namespace)
    else:
        monkeypatch.setattr(task_store, "_now", lambda: clock[0])
        store = MemoryTaskStore()
    try:
        payload = {**PAYLOAD, "seed": 2**64 - 1, "backend": backend_name}
        if model == "lj":
            payload = {
                "model": "lennard-jones-2d-v1",
                "num_particles": 36,
                "box_size": [8.0, 8.0],
                "initial_temperature": 0.5,
                "num_steps": 6,
                "sample_stride": 3,
                "seed": 2**64 - 1,
                "backend": backend_name,
            }
        store.submit("retry", payload, {})
        obsolete = store.claim()
        assert obsolete is not None
        if storage == "memory":
            # Memory release intentionally cancels during process shutdown.
            # Lease expiry is its recoverable retry path.
            clock[0] += 31.0
        else:
            assert store.release(obsolete)
        current = store.claim()
        assert current is not None
        assert current.owner != obsolete.owner
        assert current.payload == obsolete.payload == payload
        assert current.payload["seed"] == 2**64 - 1
        # A previous thread can finish after losing its lease. Its
        # artifacts remain isolated and cannot replace the committed attempt.
        stale_result = run_simulation(
            parameters_from_payload(obsolete.payload),
            task_id=obsolete.task_id,
            attempt_id=obsolete.owner,
            artifacts_dir=tmp_path,
        )
        retried_result = run_simulation(
            parameters_from_payload(current.payload),
            task_id=current.task_id,
            attempt_id=current.owner,
            artifacts_dir=tmp_path,
        )
        assert not store.finish(obsolete, stale_result)
        assert store.finish(current, retried_result)
        assert store.get("retry") == retried_result
        with (
            np.load(tmp_path / stale_result["artifacts"]["final_state"]) as first,
            np.load(tmp_path / retried_result["artifacts"]["final_state"]) as second,
        ):
            np.testing.assert_array_equal(first["velocities"], second["velocities"])
    finally:
        store.close()
        if connection is not None:
            try:
                keys = list(connection.scan_iter(match=namespace + ":*"))
                if keys:
                    connection.delete(*keys)
            finally:
                connection.close()
