"""Verify propagation and exported measurements with the real SDK."""

import json

import pytest

from scientific_api.core.telemetry import Telemetry


@pytest.fixture
def sdk():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    return InMemorySpanExporter, InMemoryMetricReader


def test_request_enqueue_worker_share_trace_after_json_roundtrip(sdk) -> None:
    exporter = sdk[0]()
    telemetry = Telemetry(enabled=True, span_exporter=exporter)
    try:
        with telemetry.span("http.request", kind="server"):
            with telemetry.span("simulation.enqueue", kind="producer"):
                queued_context = json.loads(json.dumps(telemetry.inject()))
        with telemetry.span("simulation.execute", carrier=queued_context, kind="consumer"):
            pass

        spans = {span.name: span for span in exporter.get_finished_spans()}
        request = spans["http.request"]
        enqueue = spans["simulation.enqueue"]
        worker = spans["simulation.execute"]
        assert enqueue.parent.span_id == request.context.span_id
        assert worker.parent.span_id == enqueue.context.span_id
        assert request.context.trace_id == worker.context.trace_id
        assert set(queued_context) == {"traceparent"}
        assert worker.parent.is_remote
        assert request.kind.name == "SERVER"
        assert enqueue.kind.name == "PRODUCER"
        assert worker.kind.name == "CONSUMER"
    finally:
        telemetry.shutdown()


def test_metrics_export_durations_and_bounded_outcomes(sdk) -> None:
    reader = sdk[1]()
    telemetry = Telemetry(enabled=True, metric_readers=[reader])
    try:
        telemetry.observe_queue_wait(0.5)
        telemetry.observe_execution(0.25)
        telemetry.observe_execution(0.75)
        telemetry.record_task("completed")
        telemetry.record_task("completed")
        telemetry.record_task("failed")
        telemetry.record_task("unexpected-arbitrary-value")
        telemetry.record_http("GET", "/simulations/{task_id}", 200, 0.125)
        data = reader.get_metrics_data()
        metrics = {
            metric.name: metric
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        wait = metrics["simulation.queue.wait"]
        assert wait.unit == "s"
        assert wait.data.data_points[0].sum == pytest.approx(0.5)
        execution = metrics["simulation.execution.duration"].data.data_points[0]
        assert execution.count == 2
        assert execution.sum == pytest.approx(1.0)
        outcomes = {
            point.attributes["status"]: point.value
            for point in metrics["simulation.tasks"].data.data_points
        }
        assert outcomes == {"completed": 2, "failed": 1, "other": 1}
        http = metrics["http.server.request.duration"].data.data_points[0]
        assert http.sum == pytest.approx(0.125)
        assert http.attributes == {
            "http.request.method": "GET",
            "http.route": "/simulations/{task_id}",
            "http.response.status_code": 200,
        }
    finally:
        telemetry.shutdown()


def test_repeated_app_lifespans_have_independent_providers(sdk) -> None:
    first_exporter, second_exporter = sdk[0](), sdk[0]()
    first = Telemetry(enabled=True, service_name="first", span_exporter=first_exporter)
    second = Telemetry(enabled=True, service_name="second", span_exporter=second_exporter)
    first.shutdown()
    first.shutdown()
    try:
        with second.span("still-running"):
            pass
        spans = second_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].resource.attributes["service.name"] == "second"
        assert first_exporter.get_finished_spans() == ()
    finally:
        second.shutdown()


def test_span_records_failures(sdk) -> None:
    exporter = sdk[0]()
    telemetry = Telemetry(enabled=True, span_exporter=exporter)
    try:
        with pytest.raises(ValueError, match="invalid numerical result"):
            with telemetry.span("simulation.execute"):
                raise ValueError("invalid numerical result")
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code.name == "ERROR"
        assert span.events[0].name == "exception"
    finally:
        telemetry.shutdown()


@pytest.mark.parametrize(
    "endpoint", ["ftp://collector", "collector:4318", "https://", "https://collector?q=1"]
)
def test_exporter_rejects_invalid_endpoint(endpoint) -> None:
    with pytest.raises(ValueError, match="OpenTelemetry endpoint"):
        Telemetry(enabled=True, endpoint=endpoint)


def test_enabled_without_exporter_does_not_print_or_mutate_global_provider(sdk, capsys) -> None:
    from opentelemetry import trace

    global_provider = trace.get_tracer_provider()
    telemetry = Telemetry(enabled=True)
    with telemetry.span("local-only"):
        assert telemetry.inject()["traceparent"]
    telemetry.shutdown()
    assert trace.get_tracer_provider() is global_provider
    assert capsys.readouterr().out == ""
