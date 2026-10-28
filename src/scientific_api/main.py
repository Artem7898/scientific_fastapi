"""Application factory with lifespan-scoped workers, adapters and telemetry."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import NoMatchFound
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .adapters.mlflow_adapter import build_tracker
from .api.v1.endpoints.analyses import router as analysis_router
from .api.v1.endpoints.simulations import router as sim_router
from .core.analysis_manager import AnalysisManager
from .core.config import Settings
from .core.queue_manager import QueueManager
from .core.task_store import MemoryTaskStore, RedisTaskStore, TaskStore
from .core.telemetry import Telemetry

logger = logging.getLogger(__name__)


class RequestTelemetryMiddleware:
    """Propagate W3C request context without buffering response bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        telemetry = (
            getattr(scope.get("app", None).state, "telemetry", None) if scope.get("app") else None
        )
        if scope["type"] != "http" or telemetry is None:
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        carrier = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope["headers"]
            if key in {b"traceparent", b"tracestate"}
        }
        status_code = 500
        started = time.perf_counter()

        async def traced_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        with telemetry.span(
            f"HTTP {method}",
            carrier=carrier,
            attributes={"http.request.method": method},
            kind="server",
        ) as span:
            try:
                await self.app(scope, receive, traced_send)
            finally:
                selected_route = scope.get("route")
                route = getattr(selected_route, "path_format", "unmatched")
                if selected_route is not None:
                    try:
                        placeholders = {
                            key: "{" + key + "}" for key in scope.get("path_params", {})
                        }
                        route = str(scope["app"].url_path_for(selected_route.name, **placeholders))
                    except (NoMatchFound, ValueError):
                        pass
                telemetry.record_http(method, route, status_code, time.perf_counter() - started)
                if span is not None:
                    span.set_attribute("http.route", route)
                    span.set_attribute("http.response.status_code", status_code)


def create_app(
    settings: Settings | None = None,
    *,
    backend: TaskStore | None = None,
    telemetry: Telemetry | None = None,
) -> FastAPI:
    """Construct an app; connections and workers exist only inside its lifespan."""
    configuration = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store: TaskStore | None = None
        manager: QueueManager | None = None
        monitoring: Telemetry | None = None
        analyses: AnalysisManager | None = None
        try:
            if backend is not None:
                store = backend
            else:
                options: dict[str, Any] = {
                    "max_queue_size": configuration.max_queue_size,
                    "ttl_seconds": configuration.task_ttl_seconds,
                    "lease_seconds": configuration.task_lease_seconds,
                    "max_attempts": configuration.max_task_attempts,
                }
                store = (
                    RedisTaskStore(
                        configuration.redis_uri, namespace=configuration.redis_namespace, **options
                    )
                    if configuration.redis_uri
                    else MemoryTaskStore(**options)
                )
            if not await asyncio.to_thread(store.health):
                raise RuntimeError("Task backend did not pass its readiness check")
            configuration.artifacts_dir.mkdir(parents=True, exist_ok=True)
            tracker = await asyncio.to_thread(
                build_tracker,
                configuration.mlflow_tracking_uri,
                configuration.mlflow_experiment_name,
            )
            monitoring = (
                telemetry
                if telemetry is not None
                else Telemetry(
                    enabled=configuration.otel_enabled,
                    service_name=configuration.otel_service_name,
                    endpoint=configuration.otel_exporter_otlp_endpoint,
                    console=configuration.otel_exporter_console,
                )
            )
            manager = QueueManager(store, configuration, tracker, monitoring)
            application.state.settings = configuration
            application.state.backend = store
            application.state.telemetry = monitoring
            application.state.queue_manager = manager
            analyses = AnalysisManager(
                configuration.artifacts_dir, configuration.analysis_worker_count
            )
            application.state.analysis_manager = analyses
            await manager.start()
            yield
        finally:
            try:
                if manager is not None:
                    await manager.stop()
            finally:
                try:
                    if analyses is not None:
                        await analyses.stop(configuration.shutdown_timeout_seconds)
                finally:
                    try:
                        if monitoring is not None:
                            await asyncio.to_thread(monitoring.shutdown)
                    finally:
                        if store is not None:
                            await asyncio.to_thread(store.close)

    application = FastAPI(
        title=configuration.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(GZipMiddleware, minimum_size=1000)
    application.add_middleware(RequestTelemetryMiddleware)
    application.include_router(sim_router, prefix="/api/v1")
    application.include_router(analysis_router, prefix="/api/v1")

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # A rejected numeric literal such as 1e309 is parsed as inf. Echoing
        # that input through Starlette's JSONResponse would turn a 422 into
        # a serialization failure. Return only JSON-safe validation details.
        details = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health/live", tags=["Health"])
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready", tags=["Health"])
    async def ready(request: Request) -> dict[str, str]:
        manager = getattr(request.app.state, "queue_manager", None)
        if manager is None or not manager.ready:
            raise HTTPException(503, "Simulation workers are unavailable")
        try:
            healthy = await asyncio.to_thread(manager.backend.health)
        except Exception:
            logger.exception("Task backend readiness check failed")
            raise HTTPException(503, "Task storage is unavailable") from None
        if not healthy:
            raise HTTPException(503, "Task storage is unavailable")
        return {"status": "ready"}

    return application


app = create_app()
