"""Strict queued Langevin contracts and auditable transport artifacts."""

import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.main import create_app
from scientific_api.models import computation

from .test_simulations import PAYLOAD as IDEAL_PAYLOAD
from .test_simulations import configuration

PAYLOAD = {
    "initial_temperature": 2.0,
    "bath_temperature": 1.0,
    "num_particles": 128,
    "mass": 2.0,
    "friction": 1.0,
    "time_step": 0.1,
    "num_steps": 100,
    "sample_stride": 10,
    "seed": 2**64 - 1,
    "acceleration": [1.0, -0.5],
    "initial_mean_velocity": [0.2, -0.3],
    "initial_position": [0.8, 0.9],
    "boundary": "periodic",
    "box_size": [1.0, 1.0],
}


def terminal(client: TestClient, url: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("Langevin workflow did not reach a terminal state")


def submit(client: TestClient, payload: dict) -> tuple[dict, dict]:
    response = client.post("/api/v1/simulations/langevin", json=payload)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert response.headers["location"] == accepted["check_url"]
    result = terminal(client, accepted["check_url"])
    assert result["status"] == "completed", result
    assert result["model_version"] == "langevin-free-v1"
    return accepted, result


def test_langevin_artifacts_keep_unwrapped_trajectory_and_drift_centered_msd(tmp_path: Path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted, result = submit(client, PAYLOAD)
        assert result["seed"] == PAYLOAD["seed"]
        assert "final_temperature" not in result
        assert "total_energy" not in result
        prefix = f"/api/v1/simulations/{accepted['task_id']}/artifacts/"
        downloads = {name: client.get(prefix + name) for name in result["artifacts"]}
        assert set(downloads) == {"manifest", "final_state", "trajectory", "transport"}
        assert all(response.status_code == 200 for response in downloads.values())
        manifest = downloads["manifest"].json()
        assert manifest["outputs"] == result
        assert manifest["parameters"]["seed"] == PAYLOAD["seed"]
        assert manifest["execution"]["backend"] == "numpy"
        records = {item["path"]: item for item in manifest["artifacts"]}
        for name in ("final_state", "trajectory", "transport"):
            assert (
                hashlib.sha256(downloads[name].content).hexdigest()
                == records[result["artifacts"][name]]["sha256"]
            )
        assert downloads["transport"].headers["content-type"].startswith("application/json")
        with (
            np.load(io.BytesIO(downloads["final_state"].content), allow_pickle=False) as final,
            np.load(io.BytesIO(downloads["trajectory"].content), allow_pickle=False) as trajectory,
        ):
            assert set(final.files) == {"positions", "unwrapped_positions", "velocities"}
            assert set(trajectory.files) == {"positions", "sample_steps", "times"}
            np.testing.assert_array_equal(trajectory["sample_steps"], np.arange(0, 101, 10))
            np.testing.assert_array_equal(trajectory["positions"][-1], final["unwrapped_positions"])
            np.testing.assert_array_equal(
                final["positions"], np.mod(final["unwrapped_positions"], 1.0)
            )
            displacement = final["unwrapped_positions"] - trajectory["positions"][0]
            mean = displacement.mean(axis=0)
            np.testing.assert_allclose(result["mean_displacement"], mean)
            raw = float(np.mean(np.sum(displacement**2, axis=1)))
            centered = float(np.sum((displacement - mean) ** 2)) / (PAYLOAD["num_particles"] - 1)
            assert result["final_msd"] == pytest.approx(raw)
            assert result["final_centered_msd"] == pytest.approx(centered)
            assert raw == pytest.approx(
                (PAYLOAD["num_particles"] - 1) / PAYLOAD["num_particles"] * centered
                + float(mean @ mean)
            )
        transport = downloads["transport"].json()
        assert result["diffusion"] == transport["diffusion"]
        assert result["diffusion"]["status"] == "available"
        assert result["diffusion"]["theoretical"] == 0.5
        assert "particle" in result["diffusion"]["independent_units"]
        events = client.get(f"/api/v1/simulations/{accepted['task_id']}/events")
        assert events.status_code == 200
        assert '"status":"completed"' in events.text
        # The equilibrium-only Maxwellian analysis must not interpret a
        # drifted, potentially transient Langevin snapshot as its old model.
        assert (
            client.post(
                f"/api/v1/simulations/{accepted['task_id']}/analyses", json={"bins": 12}
            ).status_code
            == 409
        )


def test_selected_numba_langevin_repeats_and_matches_numpy(tmp_path):
    pytest.importorskip("numba")
    states = []
    with TestClient(create_app(configuration(tmp_path))) as client:
        for backend in ("numpy", "numba", "numba"):
            _, result = submit(client, {**PAYLOAD, "backend": backend})
            manifest = json.loads((tmp_path / result["artifacts"]["manifest"]).read_text())
            assert manifest["execution"]["backend"] == backend
            with np.load(
                tmp_path / result["artifacts"]["final_state"], allow_pickle=False
            ) as state:
                states.append({key: state[key].copy() for key in state.files})
        for key in states[0]:
            np.testing.assert_allclose(states[0][key], states[1][key], rtol=1e-11, atol=1e-11)
            np.testing.assert_array_equal(states[1][key], states[2][key])


def test_short_langevin_run_retains_diagnostics_without_inventing_diffusion(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        _, result = submit(client, {**PAYLOAD, "num_steps": 4, "sample_stride": 2})
        assert result["final_centered_msd"] >= 0
        assert result["diffusion"]["status"] == "unavailable"
        assert result["diffusion"]["estimate"] is None
        assert result["diffusion"]["standard_error"] is None
        assert result["diffusion"]["theoretical"] == 0.5


@pytest.mark.parametrize(
    "patch",
    [
        {"bath_temperature": "1.0"},
        {"bath_temperature": True},
        {"num_particles": 2},
        {"num_particles": True},
        {"mass": 0.0},
        {"friction": 0.0},
        {"time_step": 20.0},
        {"seed": 2**64},
        {"backend": "jax"},
        {"acceleration": [0.0]},
        {"acceleration": [True, 0.0]},
        {"acceleration": [1001.0, 0.0]},
        {"initial_mean_velocity": [0.0, "0"]},
        {"initial_position": [1e7, 0.0]},
        {"boundary": "reflecting"},
        {"box_size": None},
        {"box_size": [0.0, 1.0]},
        {"boundary": "unbounded", "box_size": [1.0, 1.0]},
        {"sample_stride": 0},
        {"num_steps": 10000, "num_particles": 3, "sample_stride": 1},
        {"num_particles": 100_000, "sample_stride": 1},
        {"num_particles": 1_000_000, "num_steps": 101},
        {"diffusion_fit_start_step": 100},
        {"diffusion_fit_start_step": 50, "diffusion_fit_end_step": 25},
        {"unrecognized_parameter": 0},
    ],
)
def test_invalid_langevin_inputs_are_rejected_before_admission(tmp_path, patch):
    backend = MemoryTaskStore()
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        response = client.post("/api/v1/simulations/langevin", json={**PAYLOAD, **patch})
        assert response.status_code == 422, response.text
        assert backend.claim() is None


def test_langevin_admission_limits_missing_dependency_and_queue_capacity(tmp_path, monkeypatch):
    with TestClient(create_app(configuration(tmp_path, max_particle_steps=100))) as client:
        assert client.post("/api/v1/simulations/langevin", json=PAYLOAD).status_code == 422
    monkeypatch.setattr(computation, "find_spec", lambda name: None)
    backend = MemoryTaskStore()
    settings = Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused:6379/0",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )
    with TestClient(create_app(settings, backend=backend)) as client:
        response = client.post("/api/v1/simulations/langevin", json={**PAYLOAD, "backend": "numba"})
        assert response.status_code == 503
        assert "scientific-api[numba]" in response.json()["detail"]
        assert backend.claim() is None


def test_api_role_preserves_new_payload_and_cancels_it_without_a_worker(tmp_path):
    backend = MemoryTaskStore()
    settings = Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused:6379/0",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )
    with TestClient(create_app(settings, backend=backend)) as client:
        response = client.post("/api/v1/simulations/langevin", json=PAYLOAD)
        assert response.status_code == 202
        accepted = response.json()
        assert client.get(accepted["check_url"]).json()["status"] == "pending"
        job = backend.claim()
        assert job.payload["model"] == "langevin-free-v1"
        assert job.payload["seed"] == 2**64 - 1
        assert list(job.payload["acceleration"]) == PAYLOAD["acceleration"]
        assert backend.events(job.task_id)["snapshot"]["total_steps"] == PAYLOAD["num_steps"]
        # A separate queued task can be cancelled before any process claims it.
        second = client.post("/api/v1/simulations/langevin", json=PAYLOAD).json()
        cancelled = client.post(f"/api/v1/simulations/{second['task_id']}/cancel")
        assert cancelled.status_code == 202
        assert client.get(second["check_url"]).json()["status"] == "cancelled"
        assert (
            '"status":"cancelled"'
            in client.get(f"/api/v1/simulations/{second['task_id']}/events").text
        )


def test_new_model_ensemble_materializes_each_backend_seed_and_statistics(tmp_path):
    parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
    payload = {
        "model": "langevin-free-v1",
        "parameters": parameters,
        "replicas": 3,
        "base_seed": 42,
    }
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post(
            "/api/v1/ensembles", json=payload, headers={"Idempotency-Key": "transport"}
        )
        assert response.status_code == 202, response.text
        accepted = response.json()
        repeated = client.post(
            "/api/v1/ensembles", json=payload, headers={"Idempotency-Key": "transport"}
        )
        assert repeated.json() == accepted
        changed = {**payload, "parameters": {**parameters, "acceleration": [0.0, 0.0]}}
        assert (
            client.post(
                "/api/v1/ensembles", json=changed, headers={"Idempotency-Key": "transport"}
            ).status_code
            == 409
        )
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "completed", result
        assert result["model"] == "langevin-free-v1"
        assert result["counts"]["completed"] == 3
        assert set(result["statistics"]) == {
            "thermal_temperature",
            "kinetic_energy",
            "final_centered_msd",
            "diffusion_estimate",
        }
        temperatures = [child["result"]["thermal_temperature"] for child in result["children"]]
        assert result["statistics"]["thermal_temperature"]["mean"] == math.fsum(temperatures) / 3
        expected_temperature = 1.0 + math.exp(-20.0)
        assert result["theoretical_temperature_standard_error"] == pytest.approx(
            expected_temperature / math.sqrt(127 * 3)
        )
        manifest = client.get(f"/api/v1/ensembles/{accepted['ensemble_id']}/manifest")
        assert manifest.status_code == 200
        assert manifest.json()["outputs"] == result
        for child in result["children"]:
            run = client.get(f"/api/v1/simulations/{child['task_id']}/artifacts/manifest").json()
            assert run["model"]["version"] == "langevin-free-v1"
            assert run["parameters"]["seed"] == child["seed"]


def test_legacy_endpoint_does_not_accept_langevin_parameters_or_artifacts(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        assert client.post("/api/v1/simulations/monte-carlo", json=PAYLOAD).status_code == 422
        accepted = client.post("/api/v1/simulations/monte-carlo", json=IDEAL_PAYLOAD).json()
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "completed"
        assert (
            client.get(f"/api/v1/simulations/{result['task_id']}/artifacts/trajectory").status_code
            == 404
        )


def test_langevin_ensemble_total_trajectory_budget_is_rejected_before_admission(tmp_path):
    backend = MemoryTaskStore()
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        response = client.post(
            "/api/v1/ensembles",
            json={
                "model": "langevin-free-v1",
                "parameters": {
                    "initial_temperature": 1.0,
                    "bath_temperature": 1.0,
                    "num_particles": 524_289,
                    "num_steps": 3,
                    "sample_stride": 1,
                },
                "replicas": 16,
            },
        )
        assert response.status_code == 422, response.text
        assert "512 MiB" in response.text
        assert backend.claim() is None
