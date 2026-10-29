"""Queued periodic LJ experiments keep signed observables and model provenance."""

import hashlib
import io
import math
import time

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
    "num_particles": 64,
    "box_size": [10.0, 10.0],
    "initial_temperature": 0.1,
    "time_step": 0.001,
    "num_steps": 12,
    "sample_stride": 3,
    "rdf_bins": 16,
    "seed": 2**64 - 1,
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
    pytest.fail("LJ task did not reach a terminal state")


def submit(client: TestClient, payload: dict) -> tuple[dict, dict]:
    response = client.post("/api/v1/simulations/lennard-jones", json=payload)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert response.headers["location"] == accepted["check_url"]
    result = terminal(client, accepted["check_url"])
    assert result["status"] == "completed", result
    assert result["model_version"] == "lennard-jones-2d-v1"
    return accepted, result


def test_lj_http_preserves_signed_energy_pressure_and_four_verified_artifacts(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted, result = submit(client, PAYLOAD)
        assert result["seed"] == PAYLOAD["seed"]
        assert result["ensemble"] == "nve"
        assert result["potential_energy"] < 0
        assert result["total_energy"] < 0
        assert result["pressure"] < 0
        assert result["kinetic_energy"] >= 0
        assert result["thermal_temperature"] >= 0
        prefix = f"/api/v1/simulations/{accepted['task_id']}/artifacts/"
        downloads = {name: client.get(prefix + name) for name in result["artifacts"]}
        assert set(downloads) == {"manifest", "final_state", "trajectory", "observables"}
        assert all(response.status_code == 200 for response in downloads.values())
        manifest = downloads["manifest"].json()
        assert manifest["outputs"] == result
        assert manifest["parameters"]["ensemble"] == "nve"
        assert manifest["parameters"]["bath_temperature"] is None
        assert manifest["parameters"]["friction"] is None
        assert manifest["parameters"]["seed"] == 2**64 - 1
        assert manifest["execution"]["backend"] == "numpy"
        assert manifest["model"]["version"] == result["model_version"]
        records = {item["path"]: item for item in manifest["artifacts"]}
        for name in ("final_state", "trajectory", "observables"):
            assert (
                hashlib.sha256(downloads[name].content).hexdigest()
                == records[result["artifacts"][name]]["sha256"]
            )
        assert downloads["observables"].headers["content-type"].startswith("application/json")
        observed = downloads["observables"].json()
        assert observed["series"]["total_energy"][-1] == result["total_energy"]
        assert observed["series"]["pressure"][-1] == result["pressure"]
        assert observed["rdf"]["dimensions"] == 2
        assert len(observed["rdf"]["g_r"]) == PAYLOAD["rdf_bins"]
        assert observed["energy_drift"]["applicable"] is True
        with (
            np.load(io.BytesIO(downloads["final_state"].content), allow_pickle=False) as state,
            np.load(io.BytesIO(downloads["trajectory"].content), allow_pickle=False) as trajectory,
        ):
            assert set(state.files) == {"positions", "unwrapped_positions", "velocities"}
            assert set(trajectory.files) == {"positions", "sample_steps", "times"}
            np.testing.assert_array_equal(trajectory["sample_steps"], [0, 3, 6, 9, 12])
            np.testing.assert_array_equal(trajectory["positions"][-1], state["unwrapped_positions"])
            assert np.all((state["positions"] >= 0) & (state["positions"] < 10))
        assert client.get(prefix + "transport").status_code == 404
        assert (
            client.post(f"/api/v1/simulations/{accepted['task_id']}/analyses", json={}).status_code
            == 409
        )
        events = client.get(f"/api/v1/simulations/{accepted['task_id']}/events")
        assert events.status_code == 200
        assert '"status":"completed"' in events.text


def test_lj_nvt_records_thermostat_and_does_not_claim_energy_conservation(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        _, result = submit(client, {**PAYLOAD, "ensemble": "nvt", "bath_temperature": 0.7})
        manifest = client.get(f"/api/v1/simulations/{result['task_id']}/artifacts/manifest").json()
        assert manifest["parameters"]["friction"] == 1.0
        assert manifest["parameters"]["bath_temperature"] == 0.7
        assert result["ensemble"] == "nvt"
        observed = client.get(
            f"/api/v1/simulations/{result['task_id']}/artifacts/observables"
        ).json()
        assert observed["energy_drift"]["applicable"] is False


def test_lj_backends_and_pair_paths_match_short_seeded_trajectories(tmp_path):
    pytest.importorskip("numba")
    states = []
    with TestClient(create_app(configuration(tmp_path))) as client:
        for backend, interaction in (
            ("numpy", "direct"),
            ("numpy", "verlet"),
            ("numba", "direct"),
            ("numba", "verlet"),
            ("numba", "verlet"),
        ):
            _, result = submit(
                client, {**PAYLOAD, "backend": backend, "interaction_backend": interaction}
            )
            manifest = client.get(
                f"/api/v1/simulations/{result['task_id']}/artifacts/manifest"
            ).json()
            assert manifest["execution"]["backend"] == backend
            assert manifest["parameters"]["interaction_backend"] == interaction
            with np.load(
                tmp_path / result["artifacts"]["final_state"], allow_pickle=False
            ) as state:
                states.append({name: state[name].copy() for name in state.files})
        for observed in states[1:]:
            for name in states[0]:
                np.testing.assert_allclose(observed[name], states[0][name], rtol=1e-10, atol=1e-10)
        for name in states[-1]:
            np.testing.assert_array_equal(states[-1][name], states[-2][name])


@pytest.mark.parametrize(
    "patch",
    [
        {"num_particles": 2},
        {"num_particles": True},
        {"num_particles": 4097},
        {"initial_temperature": "0.1"},
        {"initial_temperature": True},
        {"initial_temperature": 101.0},
        {"epsilon": 0.0},
        {"sigma": 0.01},
        {"mass": 0.0},
        {"time_step": 0.1},
        {"num_steps": 0},
        {"seed": 2**64},
        {"seed": True},
        {"box_size": [4.0, 4.0]},
        {"box_size": [True, 10.0]},
        {"box_size": [10.0]},
        {"num_particles": 256},  # Dense initial lattice would overlap.
        {"cutoff_radius": 1.5},
        {"cutoff_radius": 5.0},
        {"neighbor_skin": 0.001},
        {"sample_stride": 0},
        {"ensemble": "nvt"},
        {"ensemble": "nve", "bath_temperature": 1.0},
        {"ensemble": "nve", "friction": 1.0},
        {"ensemble": "nvt", "bath_temperature": 1.0, "friction": True},
        {"rdf_bins": 7},
        {"equilibration_steps": 12},
        {"interaction_backend": "tree"},
        {"backend": "jax"},
        {"acceleration": [0.1, 0.0]},
        {"boundary": "unbounded"},
        {"num_steps": 10000, "sample_stride": 1},
        {
            "num_particles": 512,
            "box_size": [32.0, 32.0],
            "num_steps": 1000,
            "interaction_backend": "direct",
        },
        {
            "num_particles": 4096,
            "box_size": [72.0, 72.0],
            "num_steps": 10000,
            "sample_stride": 1000,
        },
    ],
)
def test_invalid_lj_inputs_are_rejected_before_admission(tmp_path, patch):
    backend = MemoryTaskStore()
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        response = client.post("/api/v1/simulations/lennard-jones", json={**PAYLOAD, **patch})
        assert response.status_code == 422, response.text
        assert backend.claim() is None


def api_settings(tmp_path):
    return Settings(
        _env_file=None,
        execution_mode="api",
        redis_uri="redis://unused:6379/0",
        artifacts_dir=tmp_path,
        mlflow_tracking_uri="",
        otel_enabled=False,
    )


def test_lj_admission_checks_dependency_and_preserves_normalized_payload_before_claim(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(computation, "find_spec", lambda name: None)
    backend = MemoryTaskStore()
    with TestClient(create_app(api_settings(tmp_path), backend=backend)) as client:
        missing = client.post(
            "/api/v1/simulations/lennard-jones", json={**PAYLOAD, "backend": "numba"}
        )
        assert missing.status_code == 503
        assert backend.claim() is None
        response = client.post(
            "/api/v1/simulations/lennard-jones",
            json={**PAYLOAD, "ensemble": "nvt", "bath_temperature": 0.7},
        )
        assert response.status_code == 202, response.text
        accepted = response.json()
        assert client.get(accepted["check_url"]).json()["status"] == "pending"
        job = backend.claim()
        assert job.payload["model"] == "lennard-jones-2d-v1"
        assert job.payload["friction"] == 1.0
        assert job.payload["seed"] == 2**64 - 1
        assert backend.events(job.task_id)["snapshot"]["total_steps"] == PAYLOAD["num_steps"]
        second = client.post("/api/v1/simulations/lennard-jones", json=PAYLOAD).json()
        assert client.post(f"/api/v1/simulations/{second['task_id']}/cancel").status_code == 202
        assert client.get(second["check_url"]).json()["status"] == "cancelled"


def test_lj_ensemble_aggregates_signed_observables_with_empirical_replica_error(tmp_path):
    parameters = {key: value for key, value in PAYLOAD.items() if key != "seed"}
    payload = {
        "model": "lennard-jones-2d-v1",
        "parameters": parameters,
        "replicas": 3,
        "base_seed": 2**64 - 1,
    }
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post(
            "/api/v1/ensembles", json=payload, headers={"Idempotency-Key": "lj-ensemble"}
        )
        assert response.status_code == 202, response.text
        accepted = response.json()
        assert (
            client.post(
                "/api/v1/ensembles", json=payload, headers={"Idempotency-Key": "lj-ensemble"}
            ).json()
            == accepted
        )
        changed = {**payload, "parameters": {**parameters, "initial_temperature": 0.2}}
        assert (
            client.post(
                "/api/v1/ensembles", json=changed, headers={"Idempotency-Key": "lj-ensemble"}
            ).status_code
            == 409
        )
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "completed", result
        assert result["model"] == "lennard-jones-2d-v1"
        assert result["counts"]["completed"] == 3
        assert result["theoretical_temperature_standard_error"] is None
        assert set(result["statistics"]) == {
            "thermal_temperature",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
            "pressure",
        }
        values = [child["result"]["potential_energy"] for child in result["children"]]
        assert result["statistics"]["potential_energy"]["mean"] == math.fsum(values) / 3
        assert result["statistics"]["total_energy"]["mean"] < 0
        manifest = client.get(f"/api/v1/ensembles/{accepted['ensemble_id']}/manifest").json()
        assert manifest["outputs"] == result
        assert (
            "interacting" in manifest["statistics_method"]["theoretical_temperature_standard_error"]
        )
        for child in result["children"]:
            record = client.get(f"/api/v1/simulations/{child['task_id']}/artifacts/manifest").json()
            assert record["model"]["version"] == result["model"]
            assert record["parameters"]["seed"] == child["seed"]


def test_lj_collective_budgets_and_model_mismatch_are_rejected_before_admission(tmp_path):
    invalid = [
        {"parameters": {}, "model": "ideal-gas-ou-v1"},
        {
            "parameters": {"initial_temperature": 1.0, "bath_temperature": 1.0, "num_particles": 8},
            "model": "lennard-jones-2d-v1",
        },
        {
            "parameters": {
                "num_particles": 512,
                "box_size": [32.0, 32.0],
                "num_steps": 700,
                "interaction_backend": "direct",
            },
            "replicas": 2,
        },
        {
            "parameters": {
                "num_particles": 4096,
                "box_size": [72.0, 72.0],
                "num_steps": 1000,
                "sample_stride": 1000,
            },
            "replicas": 6,
        },
        {"model": []},
    ]
    backend = MemoryTaskStore()
    with TestClient(create_app(configuration(tmp_path), backend=backend)) as client:
        for patch in invalid:
            response = client.post(
                "/api/v1/ensembles",
                json={
                    "model": "lennard-jones-2d-v1",
                    "parameters": {},
                    "replicas": 2,
                    **patch,
                },
            )
            assert response.status_code == 422, response.text
            assert backend.claim() is None


def test_old_model_does_not_expose_lj_observables_artifact(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post("/api/v1/simulations/monte-carlo", json=IDEAL_PAYLOAD).json()
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "completed"
        assert (
            client.get(f"/api/v1/simulations/{result['task_id']}/artifacts/observables").status_code
            == 404
        )
