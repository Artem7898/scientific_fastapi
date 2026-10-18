"""Exercise trace propagation across HTTP, persistent job data and a worker thread."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from scientific_api.core.config import Settings
from scientific_api.core.telemetry import Telemetry
from scientific_api.main import create_app


async def test_http_enqueue_worker_and_numeric_thread_share_trace(tmp_path, monkeypatch) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from scientific_api.core import queue_manager

    global_provider = trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    telemetry = Telemetry(enabled=True, span_exporter=exporter, metric_readers=[reader])
    original_simulation = queue_manager.run_simulation

    def traced_numeric_work(*args, **kwargs):
        with telemetry.span("simulation.numeric"):
            return original_simulation(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", traced_numeric_work)
    app = create_app(
        Settings(artifacts_dir=tmp_path, worker_count=1, redis_uri="", mlflow_tracking_uri=""),
        telemetry=telemetry,
    )
    incoming_trace_id = "0af7651916cd43dd8448eb211c80319c"
    incoming_span_id = "b7ad6b7169203331"
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            accepted = await client.post(
                "/api/v1/simulations/monte-carlo",
                json={
                    "num_particles": 10,
                    "initial_temperature": 1.0,
                    "time_step": 0.01,
                    "num_steps": 5,
                    "seed": 42,
                },
                headers={
                    "traceparent": f"00-{incoming_trace_id}-{incoming_span_id}-01",
                    "tracestate": "science=seeded",
                },
            )
            assert accepted.status_code == 202
            task_id = accepted.json()["task_id"]
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                result = await client.get(accepted.json()["check_url"])
                assert result.status_code == 200
                finished = exporter.get_finished_spans()
                if result.json()["status"] == "completed" and any(
                    span.name == "simulation.execute" for span in finished
                ):
                    break
                assert asyncio.get_running_loop().time() < deadline, result.json()
                await asyncio.sleep(0.01)
            assert result.json()["task_id"] == task_id
            assert (await client.get(f"/unknown/{task_id}")).status_code == 404

        spans = {span.name: span for span in exporter.get_finished_spans()}
        request = spans["HTTP POST"]
        enqueue = spans["simulation.enqueue"]
        execute = spans["simulation.execute"]
        numeric = spans["simulation.numeric"]
        assert request.parent.span_id == int(incoming_span_id, 16)
        assert enqueue.parent.span_id == request.context.span_id
        assert execute.parent.span_id == enqueue.context.span_id
        assert numeric.parent.span_id == execute.context.span_id
        assert request.kind.name == "SERVER"
        assert enqueue.kind.name == "PRODUCER"
        assert execute.kind.name == "CONSUMER"
        for span in (request, enqueue, execute, numeric):
            assert span.context.trace_id == int(incoming_trace_id, 16)
            assert span.context.trace_state.get("science") == "seeded"

        data = reader.get_metrics_data()
        metrics = {
            metric.name: metric
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert metrics["simulation.queue.wait"].data.data_points[0].count == 1
        assert metrics["simulation.execution.duration"].data.data_points[0].count == 1
        completed = metrics["simulation.tasks"].data.data_points[0]
        assert completed.attributes["status"] == "completed"
        assert completed.value == 1
        request_metrics = metrics["http.server.request.duration"].data.data_points
        assert {point.attributes["http.route"] for point in request_metrics} == {
            "/api/v1/simulations/monte-carlo",
            "/api/v1/simulations/status/{task_id}",
            "unmatched",
        }
        assert all(task_id not in str(point.attributes) for point in request_metrics)

    # Shutdown belongs to this app, never to a global SDK singleton.
    assert trace.get_tracer_provider() is global_provider
    finished_count = len(exporter.get_finished_spans())
    with telemetry.span("after-shutdown"):
        assert telemetry.inject() == {}
    assert len(exporter.get_finished_spans()) == finished_count
