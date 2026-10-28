"""SSE replay, resource bounds, disconnects and cancellation HTTP contracts."""

import asyncio
import copy
import json
import os
import time
import uuid
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.requests import ClientDisconnect

from scientific_api.api.dependencies import get_queue_manager
from scientific_api.api.v1.endpoints import events
from scientific_api.core.config import Settings
from scientific_api.core.task_store import RedisTaskStore
from scientific_api.main import create_app


def snapshot(status="pending", completed_steps=0):
    return {
        "task_id": "task",
        "status": status,
        "phase": status,
        "attempt": 1 if status != "pending" else 0,
        "completed_steps": completed_steps,
        "total_steps": 20,
    }


class HistoryBackend:
    """A deterministic backend seam; no numerical worker runs for SSE reads."""

    def __init__(self, states):
        self.history = [
            {"event_id": f"{number}-0", "data": state}
            for number, state in enumerate(states, start=1)
        ]
        self.fail = False
        self.missing = False
        self.calls = []

    def events(self, task_id, after_id=None, limit=100):
        self.calls.append(after_id)
        if self.fail:
            raise ConnectionError("private connection details")
        if self.missing or task_id != "task":
            return None
        ids = [event["event_id"] for event in self.history]
        reset = after_id is not None and after_id not in ids
        first = ids.index(after_id) + 1 if after_id in ids else 0
        return copy.deepcopy(
            {
                "snapshot": self.history[-1]["data"],
                "events": [] if reset else self.history[first : first + limit],
                "reset": reset,
                "last_event_id": ids[-1],
            }
        )

    def get(self, task_id):
        if self.fail:
            raise ConnectionError("private connection details")
        return None if self.missing or task_id != "task" else self.history[-1]["data"]

    def request_cancel(self, task_id):
        if self.fail:
            raise ConnectionError("private connection details")
        state = self.get(task_id)
        if state is None or state["status"] in {"completed", "failed"}:
            return False
        if state["status"] == "pending":
            self.append(snapshot("cancelled"))
        return True

    def append(self, state):
        self.history.append({"event_id": f"{len(self.history) + 1}-0", "data": state})


def application(backend, *, capacity=32, interval=0.001):
    app = FastAPI()
    manager = SimpleNamespace(
        backend=backend,
        settings=SimpleNamespace(
            sse_max_connections=capacity, event_poll_interval_seconds=interval
        ),
    )
    app.include_router(events.router, prefix="/api/v1")
    app.dependency_overrides[get_queue_manager] = lambda: manager
    return app


def frames(content):
    parsed = []
    for frame in content.split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in frame.splitlines())
        fields["data"] = json.loads(fields["data"])
        parsed.append(fields)
    return parsed


async def test_terminal_history_replays_in_pages_and_keeps_polling_contract():
    states = [snapshot("running", step) for step in range(205)] + [snapshot("completed", 205)]
    backend = HistoryBackend(states)
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/simulations/task/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    progress = frames(response.text)
    assert len(progress) == 206
    assert progress[-1]["data"]["status"] == "completed"
    assert [event["id"] for event in progress] == [f"{number}-0" for number in range(1, 207)]
    assert backend.calls == [None, "100-0", "200-0"]


@pytest.mark.parametrize("last_id", ["1-0", "2-0", "3-0"])
async def test_reconnect_replays_after_id_and_always_delivers_terminal_snapshot(last_id):
    backend = HistoryBackend([snapshot(), snapshot("running", 10), snapshot("completed", 20)])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/simulations/task/events", headers={"Last-Event-ID": last_id}
        )
    received = frames(response.text)
    assert received[-1]["id"] == "3-0"
    assert received[-1]["data"]["status"] == "completed"
    assert len(received) == (2 if last_id == "1-0" else 1)


async def test_unknown_or_trimmed_cursor_resets_to_current_snapshot():
    backend = HistoryBackend([snapshot("completed", 20)])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/simulations/task/events", headers={"Last-Event-ID": "999-1"}
        )
    assert frames(response.text) == [
        {"id": "1-0", "event": "reset", "data": snapshot("completed", 20)}
    ]


@pytest.mark.parametrize("last_id", ["1", "-1-0", "1-0\n", "a-1", "1-2-3", "1" * 64 + "-0"])
async def test_cursor_is_bounded_and_validated_before_backend_access(last_id):
    backend = HistoryBackend([snapshot("completed", 20)])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/simulations/task/events", headers={"Last-Event-ID": last_id}
        )
    assert response.status_code == 422
    assert backend.calls == []


async def test_missing_or_unavailable_backend_is_reported_before_sse_headers():
    backend = HistoryBackend([snapshot()])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        assert (await client.get("/api/v1/simulations/missing/events")).status_code == 404
        backend.fail = True
        response = await client.get("/api/v1/simulations/task/events")
        assert response.status_code == 503
        assert "private connection details" not in response.text


def scope(*, spec="2.0"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec},
        "http_version": "1.1",
        "scheme": "http",
        "method": "GET",
        "path": "/api/v1/simulations/task/events",
        "raw_path": b"/api/v1/simulations/task/events",
        "query_string": b"",
        "headers": [],
        "server": ("test", 80),
        "client": ("test", 1234),
        "root_path": "",
    }


async def test_stream_capacity_is_released_on_disconnect_and_heartbeats_continue(monkeypatch):
    backend = HistoryBackend([snapshot()])
    app = application(backend, capacity=1)
    monkeypatch.setattr(events, "_HEARTBEAT_SECONDS", 0.01)
    incoming = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": b"", "more_body": False})
    heartbeat = asyncio.Event()

    async def send(message):
        if b": keep-alive" in message.get("body", b""):
            heartbeat.set()

    connection = asyncio.create_task(app(scope(), incoming.get, send))
    try:
        await asyncio.wait_for(heartbeat.wait(), timeout=2)
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            overflow = await asyncio.wait_for(
                client.get("/api/v1/simulations/task/events"), timeout=2
            )
            assert overflow.status_code == 429
            assert overflow.headers["retry-after"] == "1"
    finally:
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(connection, timeout=2)

    backend.append(snapshot("cancelled"))
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/api/v1/simulations/task/events")).status_code == 200


async def test_disconnect_before_generator_starts_does_not_leak_capacity():
    backend = HistoryBackend([snapshot()])
    app = application(backend, capacity=1)

    async def send(message):
        assert message["type"] == "http.response.start"
        raise OSError("connection already closed")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(ClientDisconnect):
        await app(scope(spec="2.4"), receive, send)
    backend.append(snapshot("cancelled"))
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/api/v1/simulations/task/events")).status_code == 200


async def test_backend_loss_after_stream_start_is_an_error_event():
    backend = HistoryBackend([snapshot()])
    original = backend.events

    def fail_after_first_read(*args, **kwargs):
        result = original(*args, **kwargs)
        backend.fail = True
        return result

    backend.events = fail_after_first_read
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/simulations/task/events")
    assert response.status_code == 200
    received = frames(response.text)
    assert received[-1]["event"] == "error"
    assert "private connection details" not in response.text


@pytest.mark.parametrize("status", ["pending", "running", "cancelled"])
async def test_cancellation_is_acknowledged_and_idempotent(status):
    backend = HistoryBackend([snapshot(status)])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        for _ in range(2):
            response = await client.post("/api/v1/simulations/task/cancel")
            assert response.status_code == 202
            assert response.json() == {"task_id": "task", "status": "cancellation_requested"}
            assert response.headers["location"] == "/api/v1/simulations/status/task"


@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_cancellation_cannot_change_completed_or_failed_tasks(status):
    backend = HistoryBackend([snapshot(status)])
    async with AsyncClient(
        transport=ASGITransport(application(backend)), base_url="http://test"
    ) as client:
        assert (await client.post("/api/v1/simulations/task/cancel")).status_code == 409
        assert (await client.post("/api/v1/simulations/missing/cancel")).status_code == 404
        backend.fail = True
        assert (await client.post("/api/v1/simulations/task/cancel")).status_code == 503


def _configuration(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        artifacts_dir=tmp_path,
        worker_count=1,
        redis_uri="",
        mlflow_tracking_uri="",
        otel_enabled=False,
        event_poll_interval_seconds=0.001,
        **overrides,
    )


def _terminal(client, url):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(url).json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("Task did not reach a terminal state")


def test_completed_task_can_be_replayed_without_repeating_numerical_work(tmp_path, monkeypatch):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    invocations = []

    def counted(*args, **kwargs):
        invocations.append(kwargs["task_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", counted)
    with TestClient(create_app(_configuration(tmp_path))) as client:
        accepted = client.post(
            "/api/v1/simulations/monte-carlo",
            json={"initial_temperature": 1.0, "num_particles": 24, "num_steps": 20, "seed": 42},
        ).json()
        result = _terminal(client, accepted["check_url"])
        assert result["status"] == "completed"
        url = f"/api/v1/simulations/{accepted['task_id']}/events"
        history = frames(client.get(url).text)
        assert history[0]["data"]["status"] == "pending"
        assert history[-1]["data"]["status"] == "completed"
        assert history[-1]["data"]["completed_steps"] == 20
        assert all(event["data"]["task_id"] == accepted["task_id"] for event in history)
        resumed = frames(client.get(url, headers={"Last-Event-ID": history[0]["id"]}).text)
        assert resumed == history[1:]
        terminal = frames(client.get(url, headers={"Last-Event-ID": history[-1]["id"]}).text)
        assert terminal == [history[-1]]
        assert client.get(accepted["check_url"]).json() == result
    assert invocations == [accepted["task_id"]]


def test_running_task_cancel_request_reaches_terminal_event_without_artifacts(
    tmp_path, monkeypatch
):
    from scientific_api.core import queue_manager

    original = queue_manager.run_simulation
    entered = Event()

    def wait_for_cancellation(*args, **kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(3), "Heartbeat must observe the durable cancel request"
        return original(*args, **kwargs)

    monkeypatch.setattr(queue_manager, "run_simulation", wait_for_cancellation)
    with TestClient(create_app(_configuration(tmp_path, task_lease_seconds=3))) as client:
        accepted = client.post(
            "/api/v1/simulations/monte-carlo",
            json={"initial_temperature": 1.0, "num_particles": 24, "num_steps": 20, "seed": 42},
        ).json()
        assert entered.wait(2)
        task_id = accepted["task_id"]
        assert client.post(f"/api/v1/simulations/{task_id}/cancel").status_code == 202
        assert _terminal(client, accepted["check_url"])["status"] == "cancelled"
        history = frames(client.get(f"/api/v1/simulations/{task_id}/events").text)
        assert history[-1]["data"]["status"] == "cancelled"
        assert any(event["data"]["phase"] == "cancel_requested" for event in history)
        assert client.get(f"/api/v1/simulations/{task_id}/artifacts/manifest").status_code == 409


@pytest.mark.skipif(
    not os.environ.get("SCIENTIFIC_TEST_REDIS_URI"),
    reason="SCIENTIFIC_TEST_REDIS_URI is not configured",
)
def test_redis_replay_resumes_on_another_api_instance():
    redis = pytest.importorskip("redis")
    uri = os.environ["SCIENTIFIC_TEST_REDIS_URI"]
    namespace = "scientific-sse-" + uuid.uuid4().hex
    first_store = RedisTaskStore(uri, namespace=namespace)
    second_store = RedisTaskStore(uri, namespace=namespace)
    cleanup = redis.Redis.from_url(uri, decode_responses=True)
    try:
        first_store.submit("task", {"num_steps": 20}, {})
        job = first_store.claim()
        assert job is not None
        assert first_store.report_progress(job, 10, 20, "simulating")
        assert first_store.finish(job, {"status": "completed"})
        with (
            TestClient(application(first_store)) as first,
            TestClient(application(second_store)) as second,
        ):
            url = "/api/v1/simulations/task/events"
            original = frames(first.get(url).text)
            assert original[-1]["data"]["status"] == "completed"
            assert frames(second.get(url).text) == original
            resumed = second.get(url, headers={"Last-Event-ID": original[1]["id"]})
            assert frames(resumed.text) == original[2:]
            terminal = second.get(url, headers={"Last-Event-ID": original[-1]["id"]})
            assert frames(terminal.text) == [original[-1]]
    finally:
        first_store.close()
        second_store.close()
        keys = list(cleanup.scan_iter(match=namespace + ":*"))
        if keys:
            cleanup.delete(*keys)
        cleanup.close()
