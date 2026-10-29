"""Run the optional MLflow adapter against a real, temporary SQLite store."""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from scientific_api.core.config import Settings
from scientific_api.main import create_app


async def _terminal_result(client: AsyncClient, url: str) -> dict:
    deadline = asyncio.get_running_loop().time() + 15
    while True:
        response = await client.get(url)
        assert response.status_code == 200
        result = response.json()
        if result["status"] in {"completed", "failed"}:
            return result
        assert asyncio.get_running_loop().time() < deadline, result
        await asyncio.sleep(0.02)


async def test_parallel_runs_keep_provenance_artifacts_and_failure_status(
    tmp_path, monkeypatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from scientific_api.domain.simulation import MODEL_VERSION

    # MLflow's default artifact root is relative to cwd; keep all outputs local
    # to this fixture rather than writing a new mlruns directory in the checkout.
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    original_global_uri = mlflow.get_tracking_uri()
    artifact_root = tmp_path / "runs"
    app = create_app(
        Settings(
            _env_file=None,
            artifacts_dir=artifact_root,
            worker_count=2,
            redis_uri="",
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment_name="concurrent-science",
            otel_enabled=False,
        )
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/api/v1/simulations/monte-carlo",
                        json={
                            "num_particles": 12,
                            "initial_temperature": 1.0,
                            "time_step": 0.01,
                            "num_steps": 3,
                            "seed": seed,
                        },
                    )
                    for seed in (101, 202)
                )
            )
            assert all(response.status_code == 202 for response in responses)
            results = await asyncio.gather(
                *(_terminal_result(client, response.json()["check_url"]) for response in responses)
            )
            assert all(result["status"] == "completed" for result in results)

            def invalid_numerical_step(self):
                raise FloatingPointError("test numerical failure")

            monkeypatch.setattr(
                "scientific_api.models.physics_engine.IdealGasSimulator.advance",
                invalid_numerical_step,
            )
            failed_response = await client.post(
                "/api/v1/simulations/monte-carlo",
                json={
                    "num_particles": 12,
                    "initial_temperature": 1.0,
                    "num_steps": 1,
                    "seed": 303,
                },
            )
            assert failed_response.status_code == 202
            failed = await _terminal_result(client, failed_response.json()["check_url"])
            assert failed["status"] == "failed"

    tracking = MlflowClient(tracking_uri=tracking_uri)
    experiment = tracking.get_experiment_by_name("concurrent-science")
    runs = tracking.search_runs([experiment.experiment_id])
    assert len(runs) == 3
    assert len({run.info.run_id for run in runs}) == 3
    runs_by_task = {run.data.params["task_id"]: run for run in runs}
    assert runs_by_task[failed["task_id"]].info.status == "FAILED"
    assert runs_by_task[failed["task_id"]].data.params["seed"] == "303"

    for result, seed in zip(results, (101, 202), strict=True):
        task_id = result["task_id"]
        run = runs_by_task[task_id]
        assert run.info.status == "FINISHED"
        assert run.data.params["seed"] == str(seed)
        assert run.data.params["num_steps"] == "3"
        assert run.data.params["model_version"] == MODEL_VERSION
        for metric in ("final_temperature", "total_energy", "temperature_standard_error"):
            assert run.data.metrics[metric] == pytest.approx(result[metric])
        artifacts = {artifact.path for artifact in tracking.list_artifacts(run.info.run_id)}
        assert artifacts == {"manifest.json", "final_state.npz"}

        download_dir = tmp_path / "downloaded" / task_id
        download_dir.mkdir(parents=True)
        source_artifacts = {
            "manifest.json": artifact_root / result["artifacts"]["manifest"],
            "final_state.npz": artifact_root / result["artifacts"]["final_state"],
        }
        for artifact in artifacts:
            downloaded = tracking.download_artifacts(run.info.run_id, artifact, str(download_dir))
            with open(downloaded, "rb") as stream:
                assert stream.read() == source_artifacts[artifact].read_bytes()
        manifest = json.loads((download_dir / "manifest.json").read_text())
        assert manifest["tracking"]["run_id"] == run.info.run_id
        assert manifest["parameters"]["seed"] == seed

    assert mlflow.get_tracking_uri() == original_global_uri
    assert mlflow.active_run() is None
