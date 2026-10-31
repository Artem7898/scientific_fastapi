"""SIGKILL recovery resumes registered state in a new standalone worker."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.services.simulation_service import run_simulation

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCIENTIFIC_TEST_REDIS_URI"),
        reason="SCIENTIFIC_TEST_REDIS_URI is not configured",
    ),
    pytest.mark.skipif(os.name != "posix", reason="This test exercises POSIX SIGKILL recovery"),
]

_BLOCK_AFTER_REGISTERED = """
import json
import os
import time
from pathlib import Path

from scientific_api.core.queue_manager import QueueManager
from scientific_api.worker import main

original_register = QueueManager._register_checkpoint

def register_and_block(self, job, cancelled, descriptor):
    original_register(self, job, cancelled, descriptor)
    if descriptor["completed_steps"] == 7:
        marker = Path(os.environ["SCIENTIFIC_CHECKPOINT_TEST_MARKER"])
        temporary = marker.with_suffix(".pending")
        temporary.write_text(json.dumps(descriptor))
        os.replace(temporary, marker)
        # The asyncio heartbeat remains active while this scientific thread
        # waits. SIGKILL must terminate the process without a graceful release.
        while True:
            time.sleep(0.1)

QueueManager._register_checkpoint = register_and_block
raise SystemExit(main())
"""


def _case(model: str):
    common = {"num_steps": 20, "seed": 2**64 - 1, "backend": "numpy"}
    if model == "ou":
        return "/api/v1/simulations/monte-carlo", SimulationParameters(
            initial_temperature=1.2, num_particles=32, time_step=0.1, **common
        )
    if model == "langevin":
        return "/api/v1/simulations/langevin", LangevinParameters(
            initial_temperature=2.0,
            bath_temperature=1.0,
            num_particles=32,
            mass=1.2,
            friction=0.7,
            time_step=0.2,
            sample_stride=3,
            acceleration=(0.2, -0.1),
            boundary="periodic",
            box_size=(0.3, 0.4),
            **common,
        )
    return "/api/v1/simulations/lennard-jones", LennardJonesParameters(
        num_particles=16,
        box_size=(6.0, 6.0),
        initial_temperature=0.1,
        ensemble="nvt",
        bath_temperature=0.15,
        friction=0.7,
        time_step=0.002,
        sample_stride=3,
        interaction_backend="verlet",
        **common,
    )


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=6)


def _assert_scientific_bytes(root: Path, baseline: dict, result: dict) -> None:
    for name, value in baseline.items():
        if name not in {"task_id", "attempt_id", "artifacts"}:
            assert result[name] == value, name
    for name in ("final_state", "trajectory"):
        if name not in baseline["artifacts"]:
            continue
        with (
            np.load(root / baseline["artifacts"][name], allow_pickle=False) as expected,
            np.load(root / result["artifacts"][name], allow_pickle=False) as actual,
        ):
            assert actual.files == expected.files
            for array_name in expected.files:
                assert actual[array_name].dtype == expected[array_name].dtype
                assert actual[array_name].shape == expected[array_name].shape
                assert actual[array_name].tobytes() == expected[array_name].tobytes(), (
                    name,
                    array_name,
                )
    for name in ("transport", "observables"):
        if name in baseline["artifacts"]:
            assert (root / result["artifacts"][name]).read_bytes() == (
                root / baseline["artifacts"][name]
            ).read_bytes()


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
def test_sigkill_worker_recovery_continues_registered_checkpoint(tmp_path: Path, model: str):
    redis = pytest.importorskip("redis")
    uri = os.environ["SCIENTIFIC_TEST_REDIS_URI"]
    namespace = "scientific-checkpoint-kill-" + uuid4().hex
    artifacts = tmp_path / "runs"
    marker = tmp_path / "checkpoint-registered.json"
    connection = redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=2)
    connection.ping()  # Configured but unavailable Redis must fail, not skip.
    endpoint, parameters = _case(model)
    baseline = run_simulation(
        parameters, task_id="baseline", attempt_id="uninterrupted", artifacts_dir=artifacts
    )
    settings_values: dict[str, Any] = {
        "_env_file": None,
        "execution_mode": "api",
        "redis_uri": uri,
        "redis_namespace": namespace,
        "artifacts_dir": artifacts,
        "worker_count": 1,
        "checkpoint_interval_steps": 7,
        "task_lease_seconds": 3,
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
    }
    settings = Settings(**settings_values)
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_MODE": "worker",
            "REDIS_URI": uri,
            "REDIS_NAMESPACE": namespace,
            "ARTIFACTS_DIR": str(artifacts),
            "WORKER_COUNT": "1",
            "CHECKPOINT_INTERVAL_STEPS": "7",
            "TASK_LEASE_SECONDS": "3",
            "TASK_TTL_SECONDS": "3600",
            "MAX_TASK_ATTEMPTS": "3",
            "MLFLOW_TRACKING_URI": "",
            "OTEL_ENABLED": "false",
            "SCIENTIFIC_CHECKPOINT_TEST_MARKER": str(marker),
        }
    )
    from scientific_api.main import create_app

    application = create_app(settings)
    first = second = None
    try:
        with (
            (tmp_path / "first-worker.log").open("w") as first_log,
            (tmp_path / "second-worker.log").open("w") as second_log,
            TestClient(application) as client,
        ):
            try:
                response = client.post(endpoint, json=asdict(parameters))
                assert response.status_code == 202, response.text
                accepted = response.json()
                first = subprocess.Popen(
                    [sys.executable, "-c", _BLOCK_AFTER_REGISTERED],
                    env=environment,
                    cwd=tmp_path,
                    stdout=first_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 20
                while not marker.exists():
                    assert first.poll() is None, "First worker exited; inspect first-worker.log"
                    if time.monotonic() >= deadline:
                        pytest.fail("First worker did not register its step-seven checkpoint")
                    time.sleep(0.02)
                checkpoint = json.loads(marker.read_text())
                assert checkpoint["task_id"] == accepted["task_id"]
                assert checkpoint["completed_steps"] == 7
                assert application.state.backend.get_latest_checkpoint(accepted["task_id"]) == (
                    checkpoint
                )
                first.kill()
                assert first.wait(timeout=6) == -signal.SIGKILL
                assert client.get(accepted["check_url"]).json()["status"] == "running"
                second = subprocess.Popen(
                    [sys.executable, "-m", "scientific_api.worker"],
                    env=environment,
                    cwd=tmp_path,
                    stdout=second_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    assert second.poll() is None, (
                        "Recovery worker exited; inspect second-worker.log"
                    )
                    status = client.get(accepted["check_url"])
                    assert status.status_code == 200, status.text
                    result = status.json()
                    if result["status"] == "completed":
                        break
                    assert result["status"] in {"pending", "running"}, result
                    time.sleep(0.03)
                else:
                    pytest.fail("Recovery worker did not complete after the killed lease expired")
                assert result["attempt_id"] != checkpoint["attempt_id"]
                _assert_scientific_bytes(artifacts, baseline, result)
                manifest = json.loads((artifacts / result["artifacts"]["manifest"]).read_text())
                continuation = manifest["continuation"]
                assert continuation["mode"] == "resumed_run"
                assert continuation["completed_steps_at_start"] == 7
                assert continuation["parent_attempt_id"] == checkpoint["attempt_id"]
                assert continuation["checkpoint_id"] == checkpoint["checkpoint_id"]
                latest = application.state.backend.get_latest_checkpoint(accepted["task_id"])
                assert latest["completed_steps"] == parameters.num_steps
                assert latest["attempt_id"] == result["attempt_id"]
                events = application.state.backend.timeline(accepted["task_id"])["events"]
                resumed = [event for event in events if event["event_type"] == "attempt_resumed"]
                assert len(resumed) == 1
                assert resumed[0]["details"]["checkpoint_id"] == checkpoint["checkpoint_id"]
                assert resumed[0]["details"]["completed_steps"] == 7
                archive = application.state.backend.get_task_archive(accepted["task_id"])
                assert len(archive["attempts"]) == 2
                assert archive["attempts"][0]["attempt_id"] == checkpoint["attempt_id"]
                assert archive["attempts"][1]["attempt_id"] == result["attempt_id"]
                _stop(second)
                assert second.returncode == 0
            finally:
                _stop(first)
                _stop(second)
    finally:
        try:
            keys = list(connection.scan_iter(match=namespace + ":*"))
            if keys:
                connection.delete(*keys)
        finally:
            connection.close()
