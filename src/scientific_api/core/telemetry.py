"""Optional, application-scoped tracing and metrics.

The default facade does not import OpenTelemetry. Providers belong to an app
lifespan rather than the SDK globals, so repeated starts and tests stay isolated.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit


class Telemetry:
    """Trace requests and jobs, with opt-in OTLP/console export.

    ``endpoint`` is the OTLP HTTP base URL. With no exporter configured spans
    remain local, which is useful when a caller supplies in-memory exporters.
    ``span_exporter`` and ``metric_readers`` accept OpenTelemetry SDK objects.
    """

    def __init__(
        self,
        enabled: bool = False,
        service_name: str = "scientific-api",
        endpoint: str = "",
        console: bool = False,
        *,
        span_exporter: Any = None,
        metric_readers: Sequence[Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.tracer_provider: Any = None
        self.meter_provider: Any = None
        self._tracer: Any = None
        self._propagator: Any = None
        self._span_kinds: dict[str, Any] = {}
        self._queue_wait: Any = None
        self._execution: Any = None
        self._tasks: Any = None
        self._http: Any = None
        self._closed = False
        if not enabled:
            return
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("OpenTelemetry endpoint must be an HTTP(S) base URL")
            if parsed.query or parsed.fragment:
                raise ValueError("OpenTelemetry endpoint cannot contain a query or fragment")
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )
        except ImportError as exc:
            raise RuntimeError(
                "OpenTelemetry is enabled but unavailable; install scientific-api[observability]"
            ) from exc

        resource = Resource.create({"service.name": service_name})
        self.tracer_provider = TracerProvider(resource=resource)
        readers = list(metric_readers or ())
        if span_exporter is not None:
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        if console:
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError as exc:
                self.tracer_provider.shutdown()
                raise RuntimeError(
                    "OTLP export is unavailable; install scientific-api[observability]"
                ) from exc
            base = endpoint.rstrip("/")
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces"))
            )
            readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"))
            )
        self.meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        self._tracer = self.tracer_provider.get_tracer("scientific_api")
        meter = self.meter_provider.get_meter("scientific_api")
        self._propagator = TraceContextTextMapPropagator()
        self._span_kinds = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        self._queue_wait = meter.create_histogram(
            "simulation.queue.wait", unit="s", description="Time waiting for a worker"
        )
        self._execution = meter.create_histogram(
            "simulation.execution.duration", unit="s", description="Simulation run duration"
        )
        self._tasks = meter.create_counter(
            "simulation.tasks", unit="{task}", description="Task outcomes"
        )
        self._http = meter.create_histogram(
            "http.server.request.duration", unit="s", description="HTTP request duration"
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        carrier: Mapping[str, str] | None = None,
        attributes: dict[str, Any] | None = None,
        kind: str = "internal",
    ) -> Iterator[Any]:
        """Start a child span, optionally restoring a queued W3C carrier."""
        if not self.enabled or self._closed:
            yield None
            return
        if kind not in self._span_kinds:
            raise ValueError(f"Unsupported OpenTelemetry span kind: {kind}")
        context = self._propagator.extract(carrier) if carrier is not None else None
        with self._tracer.start_as_current_span(
            name, context=context, attributes=attributes, kind=self._span_kinds[kind]
        ) as current:
            yield current

    def inject(self) -> dict[str, str]:
        """Capture only JSON-safe W3C trace context for an asynchronous job."""
        carrier: dict[str, str] = {}
        if self.enabled and not self._closed:
            self._propagator.inject(carrier)
        return carrier

    def observe_queue_wait(self, seconds: float) -> None:
        if self.enabled and not self._closed:
            self._queue_wait.record(max(0.0, seconds))

    def observe_execution(self, seconds: float) -> None:
        if self.enabled and not self._closed:
            self._execution.record(max(0.0, seconds))

    def record_task(self, status: str) -> None:
        if self.enabled and not self._closed:
            bounded = status if status in {"completed", "failed", "cancelled"} else "other"
            self._tasks.add(1, {"status": bounded})

    def record_http(self, method: str, path: str, status: int, duration: float) -> None:
        """Record a request; ``path`` must be its route template, never a task ID."""
        if self.enabled and not self._closed:
            bounded_method = (
                method
                if method
                in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "CONNECT", "TRACE"}
                else "OTHER"
            )
            self._http.record(
                max(0.0, duration),
                {
                    "http.request.method": bounded_method,
                    "http.route": path,
                    "http.response.status_code": status,
                },
            )

    def shutdown(self) -> None:
        """Flush exporters once; disabled instances need no cleanup."""
        if not self.enabled or self._closed:
            return
        self._closed = True
        try:
            self.tracer_provider.shutdown()
        finally:
            self.meter_provider.shutdown()
