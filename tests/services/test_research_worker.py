"""A separate research CLI recovers killed leases without consuming other work."""

from __future__ import annotations

import hashlib
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

import pytest
from fastapi.testclient import TestClient

from scientific_api.core.config import Settings
from scientific_api.core.task_store import Job
from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCIENTIFIC_TEST_REDIS_URI"),
        reason="SCIENTIFIC_TEST_REDIS_URI is not configured",
    ),
    pytest.mark.skipif(os.name != "posix", reason="This test exercises POSIX SIGKILL recovery"),
]

_BLOCK_AFTER_CLAIM = """
import json
import os
import time
from pathlib import Path

from scientific_api.core.research_manager import ResearchManager
from scientific_api.research_worker import main

def block_after_claim(self, job, cancelled):
    marker = Path(os.environ["SCIENTIFIC_RESEARCH_TEST_MARKER"])
    temporary = marker.with_suffix(".pending")
    temporary.write_text(json.dumps({"analysis_id": job.task_id, "owner": job.owner}))
    os.replace(temporary, marker)
    # Real ownership and heartbeat are active. SIGKILL bypasses graceful release
    # so another CLI must wait for Redis expiry before publishing a report.
    while True:
        time.sleep(0.1)

ResearchManager._compute = block_after_claim
raise SystemExit(main())
"""


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=6)


def test_sigkill_research_cli_recovers_and_leaves_other_queues_untouched(tmp_path: Path):
    redis = pytest.importorskip("redis")
    uri = os.environ["SCIENTIFIC_TEST_REDIS_URI"]
    namespace = "scientific-research-kill-" + uuid4().hex
    artifacts = tmp_path / "runs"
    marker = tmp_path / "research-claimed.json"
    connection = redis.Redis.from_url(uri, socket_connect_timeout=2, socket_timeout=2)
    connection.ping()
    configuration: dict[str, Any] = {
        "_env_file": None,
        "execution_mode": "api",
        "redis_uri": uri,
        "redis_namespace": namespace,
        "artifacts_dir": artifacts,
        "worker_count": 1,
        "verification_worker_count": 1,
        "research_worker_count": 1,
        "checkpoint_interval_steps": 0,
        "task_lease_seconds": 3,
        "task_ttl_seconds": 1,
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_MODE": "research-worker",
            "REDIS_URI": uri,
            "REDIS_NAMESPACE": namespace,
            "ARTIFACTS_DIR": str(artifacts),
            "WORKER_COUNT": "1",
            "VERIFICATION_WORKER_COUNT": "1",
            "RESEARCH_WORKER_COUNT": "1",
            "CHECKPOINT_INTERVAL_STEPS": "0",
            "TASK_LEASE_SECONDS": "3",
            "TASK_TTL_SECONDS": "1",
            "MAX_TASK_ATTEMPTS": "3",
            "MLFLOW_TRACKING_URI": "",
            "OTEL_ENABLED": "false",
            "SCIENTIFIC_RESEARCH_TEST_MARKER": str(marker),
        }
    )
    from scientific_api.main import create_app

    application = create_app(Settings(**configuration))
    first = second = None
    try:
        with (
            (tmp_path / "first-research-worker.log").open("w") as first_log,
            (tmp_path / "second-research-worker.log").open("w") as second_log,
            TestClient(application) as client,
        ):
            try:
                source_store = application.state.backend
                parameters = LennardJonesParameters(
                    initial_temperature=0.1,
                    num_particles=16,
                    box_size=(8.0, 8.0),
                    num_steps=4,
                    sample_stride=1,
                    seed=2**64 - 1,
                    backend="numpy",
                )
                source_id = uuid4().hex
                payload = {"model": LJ_MODEL_VERSION, **asdict(parameters)}
                source_store.submit(source_id, payload, {})
                source_job = source_store.claim()
                assert source_job is not None
                result = run_simulation(
                    parameters,
                    task_id=source_id,
                    attempt_id=source_job.owner,
                    artifacts_dir=artifacts,
                )
                envelope = read_result_envelope(
                    result, artifacts_dir=artifacts, require_present=True
                )
                assert source_store.finish(
                    source_job, result, evidence=build_archive_evidence(envelope)
                )
                source_before = source_store.get_task_archive(source_id)
                files_before = {
                    path.relative_to(artifacts): path.read_bytes()
                    for path in artifacts.rglob("*")
                    if path.is_file()
                }
                simulation_sentinel = uuid4().hex
                source_store.submit(simulation_sentinel, payload, {})
                verification = client.post(
                    f"/api/v1/simulations/{source_id}/verifications", json={"rerun": False}
                )
                assert verification.status_code == 202, verification.text
                verification_accepted = verification.json()
                accepted = client.post(
                    "/api/v1/research/lj/analyses",
                    json={
                        "kind": "timeseries",
                        "task_ids": [source_id],
                        "observables": ["pressure"],
                        "bootstrap_samples": 200,
                        "bootstrap_seed": 2**64 - 1,
                    },
                )
                assert accepted.status_code == 202, accepted.text
                accepted_data = accepted.json()
                analysis_id = accepted_data["analysis_id"]
                research_store = application.state.research_manager.backend
                assert not application.state.queue_manager.ready
                assert not application.state.verification_manager.ready
                assert not application.state.research_manager.ready
                first = subprocess.Popen(
                    [sys.executable, "-c", _BLOCK_AFTER_CLAIM],
                    env=environment,
                    cwd=tmp_path,
                    stdout=first_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 20
                while not marker.exists():
                    assert first.poll() is None, (
                        "First research worker exited; inspect first-research-worker.log"
                    )
                    if time.monotonic() >= deadline:
                        pytest.fail("First research worker did not claim the analysis")
                    time.sleep(0.02)
                claimed = json.loads(marker.read_text())
                assert claimed["analysis_id"] == analysis_id
                first.kill()
                assert first.wait(timeout=6) == -signal.SIGKILL
                assert client.get(accepted_data["check_url"]).json()["status"] == "running"
                second = subprocess.Popen(
                    [sys.executable, "-m", "scientific_api.research_worker"],
                    env=environment,
                    cwd=tmp_path,
                    stdout=second_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    assert second.poll() is None, (
                        "Recovery worker exited; inspect second-research-worker.log"
                    )
                    response = client.get(accepted_data["check_url"])
                    assert response.status_code == 200, response.text
                    record = response.json()
                    if record["status"] == "completed":
                        break
                    assert record["status"] in {"pending", "running"}, record
                    time.sleep(0.03)
                else:
                    pytest.fail(
                        "Fresh research worker did not complete after the killed lease expired"
                    )
                assert record["attempts"] == 2
                # Insufficient archived samples is a successful report with
                # unavailable uncertainty, never an execution failure.
                assert record["availability"] == "unavailable", record
                report_url = accepted_data["check_url"] + "/report"
                report_response = client.get(report_url)
                assert report_response.status_code == 200, report_response.text
                raw = report_response.content
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == record["report_sha256"]
                assert digest == report_response.headers["X-Content-SHA256"]
                report = report_response.json()
                assert report["analysis_id"] == analysis_id
                assert report["analysis_attempt_id"] != claimed["owner"]
                assert report["availability"] == "unavailable"
                assert report["sources"][0]["identity"]["task_id"] == source_id
                assert report["sources"][0]["identity"]["attempt_id"] == source_job.owner
                assert report["sources"][0]["parameters"]["seed"] == 2**64 - 1
                assert report["options"]["bootstrap_seed"] == 2**64 - 1
                stale = Job(analysis_id, {}, {}, time.time(), claimed["owner"])
                assert not research_store.finish(
                    stale,
                    {"analysis_id": analysis_id, "status": "failed", "error": "stale"},
                )
                assert client.get(report_url).content == raw
                assert source_store.get_task_archive(source_id) == source_before
                assert (
                    source_store.get_task_archive(simulation_sentinel)["record"]["status"]
                    == "pending"
                )
                verification_status = client.get(verification_accepted["check_url"])
                assert verification_status.status_code == 200, verification_status.text
                assert verification_status.json()["status"] == "pending"
                assert verification_status.json()["attempts"] == 0
                assert client.get(f"/api/v1/simulations/status/{source_id}").status_code == 404
                assert {
                    path.relative_to(artifacts): path.read_bytes()
                    for path in artifacts.rglob("*")
                    if path.is_file()
                } == files_before
                listed = client.get(f"/api/v1/simulations/{source_id}/research-analyses")
                assert listed.status_code == 200, listed.text
                assert listed.json()["analyses"][0]["analysis_id"] == analysis_id
                _stop(second)
                assert second.returncode == 0
            finally:
                _stop(first)
                _stop(second)
    finally:
        # Remove only this test's UUID namespace, including its nested queues.
        keys = list(connection.scan_iter(namespace + ":*"))
        if keys:
            connection.delete(*keys)
        connection.close()
