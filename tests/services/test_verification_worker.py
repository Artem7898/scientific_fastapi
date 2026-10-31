"""A killed verifier is recovered by a fresh CLI without executing source work."""

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
from scientific_api.domain.simulation import SimulationParameters
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

from scientific_api.core.verification_manager import VerificationManager
from scientific_api.verification_worker import main

def block_after_claim(self, job, cancelled):
    marker = Path(os.environ["SCIENTIFIC_VERIFICATION_TEST_MARKER"])
    temporary = marker.with_suffix(".pending")
    temporary.write_text(json.dumps({"verification_id": job.task_id, "owner": job.owner}))
    os.replace(temporary, marker)
    # The real leased claim and heartbeat have started. SIGKILL prevents any
    # graceful release; only Redis lease expiry can authorize the next owner.
    while True:
        time.sleep(0.1)

VerificationManager._compute = block_after_claim
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


def test_sigkill_verification_cli_recovers_lease_and_preserves_source(tmp_path: Path):
    redis = pytest.importorskip("redis")
    uri = os.environ["SCIENTIFIC_TEST_REDIS_URI"]
    namespace = "scientific-verification-kill-" + uuid4().hex
    artifacts = tmp_path / "runs"
    marker = tmp_path / "verification-claimed.json"
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
        "checkpoint_interval_steps": 0,
        "task_lease_seconds": 3,
        "task_ttl_seconds": 1,
        "mlflow_tracking_uri": "",
        "otel_enabled": False,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_MODE": "verification-worker",
            "REDIS_URI": uri,
            "REDIS_NAMESPACE": namespace,
            "ARTIFACTS_DIR": str(artifacts),
            "WORKER_COUNT": "1",
            "VERIFICATION_WORKER_COUNT": "1",
            "CHECKPOINT_INTERVAL_STEPS": "0",
            "TASK_LEASE_SECONDS": "3",
            "TASK_TTL_SECONDS": "1",
            "MAX_TASK_ATTEMPTS": "3",
            "MLFLOW_TRACKING_URI": "",
            "OTEL_ENABLED": "false",
            "SCIENTIFIC_VERIFICATION_TEST_MARKER": str(marker),
        }
    )
    from scientific_api.main import create_app

    application = create_app(Settings(**configuration))
    first = second = None
    try:
        with (
            (tmp_path / "first-verifier.log").open("w") as first_log,
            (tmp_path / "second-verifier.log").open("w") as second_log,
            TestClient(application) as client,
        ):
            try:
                source_store = application.state.backend
                parameters = SimulationParameters(
                    initial_temperature=1.2,
                    num_particles=32,
                    num_steps=20,
                    time_step=0.1,
                    seed=2**64 - 1,
                    backend="numpy",
                )
                source_id = uuid4().hex
                source_store.submit(source_id, asdict(parameters), {})
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
                sentinel = uuid4().hex
                source_store.submit(sentinel, asdict(parameters), {})
                accepted = client.post(
                    f"/api/v1/simulations/{source_id}/verifications", json={"rerun": True}
                )
                assert accepted.status_code == 202, accepted.text
                accepted_data = accepted.json()
                verification_id = accepted_data["verification_id"]
                verification_store = application.state.verification_manager.backend
                first = subprocess.Popen(
                    [sys.executable, "-c", _BLOCK_AFTER_CLAIM],
                    env=environment,
                    cwd=tmp_path,
                    stdout=first_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 20
                while not marker.exists():
                    assert first.poll() is None, "First verifier exited; inspect first-verifier.log"
                    if time.monotonic() >= deadline:
                        pytest.fail("First verifier did not claim the verification")
                    time.sleep(0.02)
                claimed = json.loads(marker.read_text())
                assert claimed["verification_id"] == verification_id
                first.kill()
                assert first.wait(timeout=6) == -signal.SIGKILL
                assert client.get(accepted_data["check_url"]).json()["status"] == "running"
                second = subprocess.Popen(
                    [sys.executable, "-m", "scientific_api.verification_worker"],
                    env=environment,
                    cwd=tmp_path,
                    stdout=second_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    assert second.poll() is None, (
                        "Recovery verifier exited; inspect second-verifier.log"
                    )
                    response = client.get(accepted_data["check_url"])
                    assert response.status_code == 200, response.text
                    record = response.json()
                    if record["status"] == "completed":
                        break
                    assert record["status"] in {"pending", "running"}, record
                    time.sleep(0.03)
                else:
                    pytest.fail("Fresh verifier did not complete after the killed lease expired")
                assert record["attempts"] == 2
                assert record["verdict"] == "passed", record
                certificate_url = f"/api/v1/verifications/{verification_id}/certificate"
                certificate_response = client.get(certificate_url)
                assert certificate_response.status_code == 200, certificate_response.text
                raw = certificate_response.content
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == record["certificate_sha256"]
                assert digest == certificate_response.headers["X-Content-SHA256"]
                certificate = certificate_response.json()
                assert certificate["source"]["task_id"] == source_id
                assert certificate["source"]["attempt_id"] == source_job.owner
                assert certificate["verifier_attempt_id"] != claimed["owner"]
                assert certificate["options"] == {"rerun": True}
                assert all(check["status"] == "passed" for check in certificate["checks"])
                assert any(
                    check["name"] == "rerun_scientific_arrays" for check in certificate["checks"]
                )
                archived = verification_store.get(verification_id)
                assert archived["request"]["parameters"]["seed"] == 2**64 - 1
                stale = Job(verification_id, {}, {}, time.time(), claimed["owner"])
                assert not verification_store.finish(
                    stale,
                    {"verification_id": verification_id, "status": "failed", "error": "stale"},
                )
                assert client.get(certificate_url).content == raw
                assert source_store.get_task_archive(source_id) == source_before
                assert source_store.get_task_archive(sentinel)["record"]["status"] == "pending"
                assert client.get(f"/api/v1/simulations/status/{source_id}").status_code == 404
                assert {
                    path.relative_to(artifacts): path.read_bytes()
                    for path in artifacts.rglob("*")
                    if path.is_file()
                } == files_before
                listed = client.get(f"/api/v1/simulations/{source_id}/verifications")
                assert listed.status_code == 200, listed.text
                assert listed.json()["verifications"][0]["verification_id"] == verification_id
                _stop(second)
                assert second.returncode == 0
            finally:
                _stop(first)
                _stop(second)
    finally:
        # Delete this test's deliveries and its nested verification namespace.
        # The shared Redis database is concurrently used by other test suites.
        keys = list(connection.scan_iter(namespace + ":*"))
        if keys:
            connection.delete(*keys)
        connection.close()
