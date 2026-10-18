"""Real Redis integration; never flush databases or touch other namespaces."""

import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import pytest

from scientific_api.core.task_store import QueueFull, RedisTaskStore


@pytest.fixture
def redis_config() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("SCIENTIFIC_TEST_REDIS_URI")
    if not uri:
        pytest.skip("Set SCIENTIFIC_TEST_REDIS_URI to run real Redis integration tests")
    redis = pytest.importorskip("redis", reason="Redis optional extra is not installed")
    namespace = f"scientific-api-test-{uuid4()}"
    client = redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=2)
    client.ping()  # A configured but unavailable service is a failure, not a skip.
    try:
        yield uri, namespace
    finally:
        keys = list(client.scan_iter(match=f"{namespace}:*"))
        if keys:
            client.delete(*keys)
        client.close()


def make_store(config: tuple[str, str], **options: Any) -> RedisTaskStore:
    return RedisTaskStore(config[0], namespace=config[1], **options)


def test_queue_and_results_survive_client_restart(redis_config: tuple[str, str]) -> None:
    first = make_store(redis_config)
    first.submit("one", {"seed": 7}, {"traceparent": "private"})
    first.close()
    second = make_store(redis_config)
    try:
        assert second.health()
        assert second.get("one") == {"task_id": "one", "status": "pending"}
        job = second.claim()
        assert job is not None and job.payload == {"seed": 7}
        assert job.carrier == {"traceparent": "private"}
        assert second.get("one") == {"task_id": "one", "status": "running"}
        assert second.finish(job, {"task_id": "wrong", "status": "completed", "value": 42})
    finally:
        second.close()
    third = make_store(redis_config)
    try:
        assert third.get("one") == {"task_id": "one", "status": "completed", "value": 42}
        assert third.claim() is None
    finally:
        third.close()


def test_capacity_is_atomic_and_counts_running(redis_config: tuple[str, str]) -> None:
    store = make_store(redis_config, max_queue_size=4)
    try:

        def submit(index: int) -> bool:
            try:
                store.submit(str(index), {}, {})
                return True
            except QueueFull:
                return False

        with ThreadPoolExecutor(max_workers=12) as pool:
            assert sum(pool.map(submit, range(32))) == 4
        job = store.claim()
        assert job is not None
        with pytest.raises(QueueFull):
            store.submit("overflow", {}, {})
        assert store.finish(job, {"status": "completed"})
        store.submit("freed", {}, {})
        with pytest.raises(ValueError, match="already exists"):
            store.submit("freed", {}, {})
    finally:
        store.close()


def test_workers_never_claim_the_same_live_delivery(redis_config: tuple[str, str]) -> None:
    first, second = make_store(redis_config), make_store(redis_config)
    try:
        for index in range(20):
            first.submit(str(index), {}, {})
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda n: (first if n % 2 else second).claim(), range(20)))
        assert all(job is not None for job in jobs)
        assert len({job.task_id for job in jobs if job is not None}) == 20
        assert second.claim() is None
    finally:
        first.close()
        second.close()


def test_reclaim_rejects_all_obsolete_owner_operations(redis_config: tuple[str, str]) -> None:
    first = make_store(redis_config, lease_seconds=0.05)
    second = make_store(redis_config, lease_seconds=1)
    try:
        first.submit("one", {"seed": 9}, {"traceparent": "context"})
        old = first.claim()
        assert old is not None
        time.sleep(0.08)
        assert not first.heartbeat(old)
        assert not first.finish(old, {"status": "completed"})
        new = second.claim()
        assert new is not None and new.owner != old.owner
        assert new.task_id == old.task_id
        assert new.submitted_at == old.submitted_at
        assert new.payload == old.payload and new.carrier == old.carrier
        assert not first.release(old)
        assert not first.finish(old, {"status": "failed"})
        assert second.finish(new, {"status": "completed"})
    finally:
        first.close()
        second.close()


def test_heartbeat_renews_exclusive_lease(redis_config: tuple[str, str]) -> None:
    store = make_store(redis_config, lease_seconds=0.25)
    try:
        store.submit("one", {}, {})
        job = store.claim()
        assert job is not None
        time.sleep(0.15)
        assert store.heartbeat(job)
        time.sleep(0.15)
        assert store.claim() is None
        assert store.finish(job, {"status": "completed"})
    finally:
        store.close()


def test_attempt_limit_releases_admission_slot(redis_config: tuple[str, str]) -> None:
    store = make_store(redis_config, max_queue_size=1, lease_seconds=0.05, max_attempts=2)
    try:
        store.submit("one", {}, {})
        assert store.claim() is not None
        time.sleep(0.08)
        assert store.claim() is not None
        time.sleep(0.08)
        assert store.claim() is None
        record = store.get("one")
        assert record is not None and record["status"] == "failed"
        assert "maximum attempts" in record["error"]
        store.submit("two", {}, {})
    finally:
        store.close()


def test_only_terminal_records_expire(redis_config: tuple[str, str]) -> None:
    store = make_store(redis_config, ttl_seconds=0.05, lease_seconds=2)
    try:
        store.submit("one", {}, {})
        time.sleep(0.08)
        assert store.get("one") == {"task_id": "one", "status": "pending"}
        job = store.claim()
        assert job is not None
        time.sleep(0.08)
        assert store.get("one") == {"task_id": "one", "status": "running"}
        assert store.finish(job, {"status": "completed"})
        time.sleep(0.08)
        assert store.get("one") is None
    finally:
        store.close()


def test_cooperative_release_requeues_across_restart(redis_config: tuple[str, str]) -> None:
    first = make_store(redis_config, max_attempts=1)
    first.submit("one", {"seed": 1}, {})
    old = first.claim()
    assert old is not None
    assert first.release(old)
    assert not first.release(old)
    first.close()
    second = make_store(redis_config, max_attempts=1)
    try:
        assert second.get("one") == {"task_id": "one", "status": "pending"}
        new = second.claim()
        assert new is not None and new.owner != old.owner
        assert not second.finish(old, {"status": "completed"})
        assert second.finish(new, {"status": "completed"})
    finally:
        second.close()
