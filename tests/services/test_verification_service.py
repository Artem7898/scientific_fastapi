"""Verification pins, historical certificates and ambiguous commit outcomes."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scientific_api.core import task_store
from scientific_api.core.config import Settings
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.telemetry import Telemetry
from scientific_api.core.verification_manager import VerificationManager
from scientific_api.core.verification_store import MemoryVerificationStore
from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.schemas.verification import VerificationCheck
from scientific_api.services import verification_service
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import (
    publish_result_envelope,
    read_result_envelope,
)
from scientific_api.services.simulation_service import run_simulation
from scientific_api.services.verification_service import (
    VerificationError,
    certificate_bytes,
    prepare_verification,
    validate_verification_certificate,
    verification_verdict,
    verify_result,
)


def archived_source(root: Path, *, seed: int = 2**64 - 1):
    store = MemoryTaskStore()
    payload = {"initial_temperature": 2.0, "num_particles": 16, "num_steps": 4, "seed": seed}
    store.submit("source", payload, {})
    job = store.claim()
    assert job is not None
    result = run_simulation(
        SimulationParameters(**payload),
        task_id=job.task_id,
        attempt_id=job.owner,
        artifacts_dir=root,
    )
    envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
    assert store.finish(job, result, evidence=build_archive_evidence(envelope))
    archive = store.get_task_archive(job.task_id)
    assert archive is not None
    return store, result, archive


def certificate(root: Path, request: dict, **kwargs):
    return verify_result(
        request,
        verification_id="a" * 32,
        verifier_attempt_id="verifier-owner",
        artifacts_dir=root,
        temporary_root=root / "temporary-verification",
        check_cancelled=lambda: None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["passed"], "passed"),
        (["passed", "failed", "unsupported"], "failed"),
        (["passed", "unsupported"], "inconclusive"),
        (["inconclusive", "passed"], "inconclusive"),
        (["unsupported"], "unsupported"),
        ([], "inconclusive"),
    ],
)
def test_verdict_never_hides_required_check_outcome(statuses, expected):
    checks = [
        VerificationCheck(name=str(i), status=s, reason="reason") for i, s in enumerate(statuses)
    ]
    assert verification_verdict(checks) == expected


def test_admission_detaches_archive_preserves_seed_and_does_not_read_files(tmp_path):
    _, result, archive = archived_source(tmp_path)
    (tmp_path / result["artifacts"]["final_state"]).unlink()
    before = copy.deepcopy(archive)
    request = prepare_verification(archive, rerun=False)
    assert archive == before
    assert request["parameters"]["seed"] == 2**64 - 1
    request["record"]["artifacts"]["final_state"] = "changed"
    request["evidence"]["artifacts"][0]["bytes"] = -1
    assert archive == before


@pytest.mark.parametrize("bad", ["pending", "failed", "cancelled", "missing-evidence"])
def test_admission_requires_a_committed_source_not_just_files(tmp_path, bad):
    _, _, archive = archived_source(tmp_path)
    if bad == "missing-evidence":
        archive["evidence"] = None
    else:
        archive["record"]["status"] = bad
    with pytest.raises(VerificationError):
        prepare_verification(archive, rerun=False)


def test_source_replacement_cannot_replace_archival_anchor(tmp_path):
    source, result, archive = archived_source(tmp_path)
    pinned = prepare_verification(archive, rerun=False)
    state = tmp_path / result["artifacts"]["final_state"]
    with np.load(state, allow_pickle=False) as saved:
        velocities = -saved["velocities"]
    np.savez_compressed(state, velocities=velocities)
    manifest_path = tmp_path / result["artifacts"]["manifest"]
    manifest = json.loads(manifest_path.read_bytes())
    for item in manifest["artifacts"]:
        if item["path"] == result["artifacts"]["final_state"]:
            item["sha256"] = hashlib.sha256(state.read_bytes()).hexdigest()
            item["bytes"] = state.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    # All locally saved metadata agrees with the replacement, but the committed
    # archive must continue to identify the original bytes.
    publish_result_envelope(result, artifacts_dir=tmp_path)
    report = certificate(tmp_path, pinned)
    assert report.verdict == "failed"
    assert report.checks[0].reason == "VERIFICATION_ARCHIVE_MISMATCH"
    assert source.get_task_archive("source") == archive


def test_exact_snapshot_is_compared_after_envelope_read(tmp_path, monkeypatch):
    _, result, archive = archived_source(tmp_path)
    original = verification_service._snapshot
    state = tmp_path / result["artifacts"]["final_state"]

    def changed_snapshot(root, relative, maximum, *, retain=False):
        if relative == result["artifacts"]["final_state"]:
            state.write_bytes(b"changed after envelope verification")
        return original(root, relative, maximum, retain=retain)

    monkeypatch.setattr(verification_service, "_snapshot", changed_snapshot)
    report = certificate(tmp_path, prepare_verification(archive, rerun=False))
    assert report.verdict == "failed"
    assert report.checks[0].reason == "VERIFICATION_SNAPSHOT_MISMATCH"


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "0" * 64),
        ("model_contract_sha256", "0" * 64),
        ("model_version", "other-model-v1"),
        ("task_id", "other-source"),
        ("attempt_id", "other-attempt"),
    ],
)
def test_certificate_cannot_claim_another_pinned_source(tmp_path, field, value):
    _, _, archive = archived_source(tmp_path)
    request = prepare_verification(archive, rerun=False)
    request["source"][field] = value
    report = certificate(tmp_path, request)
    assert report.verdict == "failed"
    assert report.checks[0].status == "failed"


def test_offline_certificate_checks_hash_canonical_encoding_and_aggregate(tmp_path):
    _, _, archive = archived_source(tmp_path)
    report = certificate(tmp_path, prepare_verification(archive, rerun=False))
    raw = certificate_bytes(report)
    digest = hashlib.sha256(raw).hexdigest()
    assert validate_verification_certificate(raw, expected_sha256=digest) == report
    with pytest.raises(VerificationError, match="INTEGRITY_FAILED"):
        validate_verification_certificate(raw + b" ", expected_sha256=digest)
    with pytest.raises(VerificationError, match="CERTIFICATE_INVALID"):
        validate_verification_certificate(raw + b" ")
    changed = report.model_copy(deep=True)
    changed.checks[0] = VerificationCheck(name="artifact_integrity", status="failed", reason="bad")
    changed.verdict = "passed"
    with pytest.raises(VerificationError, match="CERTIFICATE_INVALID"):
        validate_verification_certificate(certificate_bytes(changed))


@pytest.mark.parametrize("invalid", [True, 1.0, "1"])
def test_certificate_version_requires_a_literal_integer(tmp_path, invalid):
    from scientific_api.schemas.verification import VerificationCertificate

    _, _, archive = archived_source(tmp_path)
    report = certificate(tmp_path, prepare_verification(archive, rerun=False)).model_dump(
        mode="json"
    )
    report["schema_version"] = invalid
    with pytest.raises(ValueError, match="must be an integer"):
        VerificationCertificate.model_validate(report)


def test_cancellation_is_not_converted_to_a_scientific_failure(tmp_path):
    _, _, archive = archived_source(tmp_path)

    def interrupted():
        raise SimulationCancelled("worker stopped")

    with pytest.raises(SimulationCancelled):
        verify_result(
            prepare_verification(archive, rerun=False),
            verification_id="b" * 32,
            verifier_attempt_id="owner",
            artifacts_dir=tmp_path,
            temporary_root=tmp_path,
            check_cancelled=interrupted,
        )


@pytest.mark.parametrize("missing", ["science", "rerun"])
def test_incomplete_policy_cannot_pass_from_checksums_or_environment_alone(
    tmp_path, monkeypatch, missing
):
    from scientific_api.services import verification_rerun, verification_science

    _, _, archive = archived_source(tmp_path)
    if missing == "science":
        monkeypatch.setattr(verification_science, "verify_scientific_result", lambda *a, **kw: [])
    else:
        monkeypatch.setattr(verification_rerun, "verify_same_seed_rerun", lambda *a, **kw: [])
    report = certificate(tmp_path, prepare_verification(archive, rerun=missing == "rerun"))
    assert report.verdict == "inconclusive"
    assert report.checks[-1].reason == "required_verification_checks_missing"
    # Removing the guard cannot turn the partial evidence into a valid passed certificate.
    report.checks.pop()
    report.verdict = "passed"
    with pytest.raises(VerificationError, match="CERTIFICATE_INVALID"):
        validate_verification_certificate(certificate_bytes(report))


@pytest.mark.asyncio
@pytest.mark.parametrize("committed_before_error", [False, True])
async def test_ambiguous_verification_commit_does_not_replace_possible_certificate(
    tmp_path, monkeypatch, committed_before_error
):
    source, _, original_archive = archived_source(tmp_path)
    backend = MemoryVerificationStore(lease_seconds=3)
    settings = Settings(
        _env_file=None, artifacts_dir=tmp_path, redis_uri="", mlflow_tracking_uri=""
    )
    manager = VerificationManager(backend, source, settings, Telemetry())
    accepted = await manager.submit("source")
    job = backend.claim()
    assert job is not None
    finish = backend.finish

    def lost_reply(delivery, record):
        if committed_before_error:
            assert finish(delivery, record)
        raise TimeoutError("commit acknowledgement lost")

    monkeypatch.setattr(backend, "finish", lost_reply)
    try:
        await manager._execute(job)
        stored = backend.get(accepted["verification_id"])
        assert stored is not None
        assert stored["status"] == ("completed" if committed_before_error else "running")
        if committed_before_error:
            raw, digest = manager.certificate(job.task_id)
            validate_verification_certificate(raw, expected_sha256=digest)
            assert backend.claim() is None
        else:
            monkeypatch.setattr(task_store, "_now", lambda: job.submitted_at + 100)
            retry = backend.claim()
            assert retry is not None and retry.owner != job.owner
        assert source.get_task_archive("source") == original_archive
    finally:
        await manager.stop()
        backend.close()
        source.close()
