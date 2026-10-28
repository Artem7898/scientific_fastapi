"""Real ensemble admission, execution, aggregation and failure HTTP contracts."""

import copy
import math
import os
import time
import uuid
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.main import create_app

PAYLOAD = {
    "model": "ideal-gas-ou-v1",
    "parameters": {
        "initial_temperature": 300.0,
        "num_particles": 32,
        "time_step": 0.1,
        "num_steps": 3,
    },
    "replicas": 3,
    "base_seed": 42,
    "failure_policy": "require_all",
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


def terminal(client: TestClient, url: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("Ensemble did not finish before its deadline")


def test_ensemble_http_workflow_and_reproducible_order_across_worker_counts(tmp_path):
    results = []
    for workers in (1, 2):
        with TestClient(
            create_app(configuration(tmp_path / str(workers), worker_count=workers))
        ) as client:
            response = client.post("/api/v1/ensembles", json=PAYLOAD)
            assert response.status_code == 202, response.text
            accepted = response.json()
            assert response.headers["location"] == accepted["check_url"]
            result = terminal(client, accepted["check_url"])
            assert result["status"] == "completed"
            assert result["counts"] == {
                "requested": 3,
                "pending": 0,
                "running": 0,
                "completed": 3,
                "failed": 0,
                "cancelled": 0,
            }
            assert [child["replica_index"] for child in result["children"]] == [0, 1, 2]
            values = [child["result"]["final_temperature"] for child in result["children"]]
            average = math.fsum(values) / 3
            sample_sd = math.sqrt(math.fsum((value - average) ** 2 for value in values) / 2)
            statistic = result["statistics"]["final_temperature"]
            assert statistic["mean"] == average
            assert statistic["sample_standard_deviation"] == sample_sd
            assert statistic["standard_error"] == sample_sd / math.sqrt(3)
            manifest = client.get(f"/api/v1/ensembles/{accepted['ensemble_id']}/manifest")
            assert manifest.status_code == 200
            assert manifest.json()["outputs"] == result
            assert manifest.json()["seed_policy"]["name"] == "numpy.SeedSequence.spawn"
            assert manifest.json()["seeds"] == [child["seed"] for child in result["children"]]
            for child in result["children"]:
                child_status = client.get(f"/api/v1/simulations/status/{child['task_id']}")
                assert child_status.json() == child["result"]
            results.append((values, result["statistics"], manifest.json()["seeds"]))
    assert results[0] == results[1]


def test_ensemble_idempotency_returns_original_and_conflicts_on_changed_request(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        headers = {"Idempotency-Key": "same-experiment"}
        first = client.post("/api/v1/ensembles", json=PAYLOAD, headers=headers)
        assert first.status_code == 202, first.text
        repeated = client.post("/api/v1/ensembles", json=PAYLOAD, headers=headers)
        assert repeated.status_code == 202
        assert repeated.json() == first.json()
        changed = client.post(
            "/api/v1/ensembles", json={**PAYLOAD, "base_seed": 43}, headers=headers
        )
        assert changed.status_code == 409
        assert (
            client.post(
                "/api/v1/ensembles", json=PAYLOAD, headers={"Idempotency-Key": "x" * 129}
            ).status_code
            == 422
        )


def test_ensemble_enqueue_propagates_one_producer_trace_to_every_child(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from scientific_api.core.telemetry import Telemetry

    exporter = InMemorySpanExporter()
    telemetry = Telemetry(enabled=True, span_exporter=exporter)
    application = create_app(configuration(tmp_path), telemetry=telemetry)
    trace_id = "0af7651916cd43dd8448eb211c80319c"
    parent_id = "b7ad6b7169203331"
    with TestClient(application) as client:
        accepted = client.post(
            "/api/v1/ensembles",
            json=PAYLOAD,
            headers={"traceparent": f"00-{trace_id}-{parent_id}-01"},
        )
        assert accepted.status_code == 202
        assert terminal(client, accepted.json()["check_url"])["status"] == "completed"
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            spans = exporter.get_finished_spans()
            executions = [span for span in spans if span.name == "simulation.execute"]
            if len(executions) == 3:
                break
            time.sleep(0.01)
        else:
            pytest.fail("All replica execution spans must finish")
        enqueue = next(span for span in spans if span.name == "ensemble.enqueue")
        http = next(span for span in spans if span.name == "HTTP POST")
        assert enqueue.kind.name == "PRODUCER"
        assert enqueue.parent.span_id == http.context.span_id
        assert enqueue.attributes["ensemble.replicas"] == 3
        for execution in executions:
            assert execution.context.trace_id == int(trace_id, 16)
            assert execution.parent.span_id == enqueue.context.span_id


def test_fifty_replica_ensemble_uses_the_existing_bounded_workers(tmp_path):
    payload = {
        **PAYLOAD,
        "replicas": 50,
        "parameters": {**PAYLOAD["parameters"], "num_particles": 4, "num_steps": 1},
    }
    application = create_app(configuration(tmp_path, worker_count=2))
    with TestClient(application) as client:
        accepted = client.post("/api/v1/ensembles", json=payload)
        assert accepted.status_code == 202, accepted.text
        result = terminal(client, accepted.json()["check_url"])
        assert result["status"] == "completed"
        assert result["counts"]["completed"] == 50
        assert len(result["children"]) == 50
        assert len({child["seed"] for child in result["children"]}) == 50
        assert result["statistics"]["final_temperature"]["n"] == 50
        assert len(application.state.queue_manager._workers) == 2


def test_redis_ensemble_and_idempotency_survive_api_restart(tmp_path):
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    if not uri:
        pytest.skip("SCIENTIFIC_TEST_REDIS_URI is not configured")
    redis = pytest.importorskip("redis")
    namespace = "scientific-ensemble-api-" + uuid.uuid4().hex
    settings = Settings(
        _env_file=None,
        artifacts_dir=tmp_path,
        redis_uri=uri,
        redis_namespace=namespace,
        mlflow_tracking_uri="",
        otel_enabled=False,
        worker_count=1,
    )
    try:
        headers = {"Idempotency-Key": "persistent-ensemble"}
        with TestClient(create_app(settings)) as first:
            accepted = first.post("/api/v1/ensembles", json=PAYLOAD, headers=headers)
            assert accepted.status_code == 202, accepted.text
            result = terminal(first, accepted.json()["check_url"])
            assert result["status"] == "completed"
        with TestClient(create_app(settings)) as second:
            assert second.get(accepted.json()["check_url"]).json() == result
            retried = second.post("/api/v1/ensembles", json=PAYLOAD, headers=headers)
            assert retried.status_code == 202, retried.text
            assert retried.json() == accepted.json()
            identifier = accepted.json()["ensemble_id"]
            manifest = second.get(f"/api/v1/ensembles/{identifier}/manifest")
            assert manifest.status_code == 200
            assert manifest.json()["outputs"] == result
    finally:
        connection = redis.Redis.from_url(uri)
        try:
            keys = list(connection.scan_iter(match=f"{namespace}:*"))
            if keys:
                connection.delete(*keys)
        finally:
            connection.close()


def test_ensemble_all_or_nothing_capacity_and_configured_total_work(tmp_path):
    with TestClient(create_app(configuration(tmp_path / "capacity", max_queue_size=2))) as client:
        response = client.post("/api/v1/ensembles", json=PAYLOAD)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"
        # Refusal does not enqueue the first two children and fill the queue.
        accepted = client.post("/api/v1/ensembles", json={**PAYLOAD, "replicas": 2})
        assert accepted.status_code == 202, accepted.text
        assert terminal(client, accepted.json()["check_url"])["status"] == "completed"
    with TestClient(
        create_app(configuration(tmp_path / "budget", max_particle_steps=200))
    ) as client:
        assert client.post("/api/v1/ensembles", json=PAYLOAD).status_code == 422
    with TestClient(
        create_app(configuration(tmp_path / "steps", max_simulation_iterations=2))
    ) as client:
        assert client.post("/api/v1/ensembles", json=PAYLOAD).status_code == 422


def test_ensemble_failure_is_explicit_without_survivor_aggregate(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation

    def fail_middle(*args, **kwargs):
        if kwargs["task_id"].endswith("_r001"):
            raise FloatingPointError("Numerical instability")
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", fail_middle)
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post("/api/v1/ensembles", json=PAYLOAD).json()
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "failed"
        assert result["counts"]["completed"] == 2
        assert result["counts"]["failed"] == 1
        assert result["statistics"] is None
        assert result["children"][1]["result"]["status"] == "failed"
        manifest = client.get(f"/api/v1/ensembles/{accepted['ensemble_id']}/manifest")
        assert manifest.status_code == 200
        assert manifest.json()["outputs"]["statistics"] is None


def test_ensemble_manifest_waits_and_cancellation_covers_pending_children(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    entered, release = Event(), Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3), "The test must release its worker"
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", blocked)
    with TestClient(create_app(configuration(tmp_path, worker_count=1))) as client:
        try:
            accepted = client.post("/api/v1/ensembles", json=PAYLOAD).json()
            assert entered.wait(2)
            prefix = f"/api/v1/ensembles/{accepted['ensemble_id']}"
            assert client.get(f"{prefix}/manifest").status_code == 409
            cancellation = client.post(f"{prefix}/cancel")
            assert cancellation.status_code == 202
            assert cancellation.json()["status"] == "cancel_requested"
            assert client.get(prefix).json()["cancel_requested"] is True
        finally:
            release.set()
        result = terminal(client, accepted["check_url"])
        assert result["status"] == "cancelled"
        assert result["counts"]["cancelled"] >= 2
        assert result["statistics"] is None
        assert client.post(f"{prefix}/cancel").status_code == 409


@pytest.mark.parametrize(
    "patch",
    [
        {"replicas": 0},
        {"replicas": 51},
        {"replicas": True},
        {"base_seed": -1},
        {"base_seed": True},
        {"parameters": {**PAYLOAD["parameters"], "seed": 42}},
        {"parameters": {**PAYLOAD["parameters"], "initial_temperature": "300"}},
        {"failure_policy": "drop_failed"},
    ],
)
def test_ensemble_strict_validation_and_missing_records(tmp_path, patch):
    with TestClient(create_app(configuration(tmp_path))) as client:
        assert (
            client.post("/api/v1/ensembles", json={**copy.deepcopy(PAYLOAD), **patch}).status_code
            == 422
        )
        assert client.get("/api/v1/ensembles/missing").status_code == 404
        assert client.get("/api/v1/ensembles/missing/manifest").status_code == 404
        assert client.post("/api/v1/ensembles/missing/cancel").status_code == 404
