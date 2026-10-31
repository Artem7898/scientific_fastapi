"""Archive acknowledgement failures must not erase committed scientific data."""

import json
from pathlib import Path

import pytest

from scientific_api.core.config import Settings
from scientific_api.core.experiment_tracker import NullTracker
from scientific_api.core.queue_manager import QueueManager
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.core.telemetry import Telemetry
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.services.analysis_service import AnalysisArchiveError, create_velocity_analysis
from scientific_api.services.provenance_evidence import (
    artifact_occurrence_id,
    build_archive_evidence,
)
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation


def completed_run(root: Path, store: MemoryTaskStore) -> dict:
    store.submit("task", {"initial_temperature": 1.0, "num_particles": 8, "num_steps": 2}, {})
    job = store.claim()
    assert job is not None
    result = run_simulation(
        SimulationParameters(1.0, 8, num_steps=2),
        task_id=job.task_id,
        attempt_id=job.owner,
        artifacts_dir=root,
    )
    envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
    assert store.finish(job, result, evidence=build_archive_evidence(envelope))
    return result


def test_artifact_occurrence_identity_separates_origins_and_roles():
    identifiers = {
        artifact_occurrence_id("task", "attempt", None, "final_state"),
        artifact_occurrence_id("other-task", "attempt", None, "final_state"),
        artifact_occurrence_id("task", "other-attempt", None, "final_state"),
        artifact_occurrence_id("task", "attempt", "analysis", "final_state"),
        artifact_occurrence_id("task", "attempt", None, "manifest"),
    }
    assert len(identifiers) == 5
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in identifiers)


def test_analysis_acknowledgement_loss_retains_durably_registered_files(tmp_path, monkeypatch):
    store = MemoryTaskStore()
    result = completed_run(tmp_path, store)
    original_register = store.register_analysis

    def lost_reply(*args, **kwargs):
        original_register(*args, **kwargs)
        raise TimeoutError("The server committed before the connection disappeared")

    monkeypatch.setattr(store, "register_analysis", lost_reply)
    with pytest.raises(AnalysisArchiveError, match="not confirmed"):
        create_velocity_analysis(
            result, artifacts_dir=tmp_path, bins=4, formats=(), provenance_store=store
        )
    archive = store.get_task_archive("task")
    assert archive is not None and len(archive["analyses"]) == 1
    analysis = archive["analyses"][0]
    verified = read_result_envelope(
        result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"], require_present=True
    )
    assert verified.integrity.status == "verified"
    assert build_archive_evidence(verified) == analysis["evidence"]
    assert store.get_analysis_archive(analysis["analysis_id"]) == analysis


def test_unconfirmed_analysis_write_retains_complete_unindexed_files(tmp_path, monkeypatch):
    store = MemoryTaskStore()
    result = completed_run(tmp_path, store)

    def no_reply(*args, **kwargs):
        raise ConnectionError("The request outcome cannot be established")

    monkeypatch.setattr(store, "register_analysis", no_reply)
    with pytest.raises(AnalysisArchiveError):
        create_velocity_analysis(
            result, artifacts_dir=tmp_path, bins=4, formats=(), provenance_store=store
        )
    assert store.get_task_archive("task")["analyses"] == []
    directory = tmp_path / result["task_id"] / result["attempt_id"] / "analyses"
    saved = list(directory.iterdir())
    assert len(saved) == 1
    metadata = json.loads((saved[0] / "manifest.json").read_text())
    analysis_id = metadata["analysis_id"]
    assert store.get_analysis_archive(analysis_id) is None
    assert (
        read_result_envelope(
            result, artifacts_dir=tmp_path, analysis_id=analysis_id, require_present=True
        ).integrity.status
        == "verified"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("committed_before_error", [False, True])
async def test_queue_archive_write_failure_preserves_actual_fenced_outcome(
    tmp_path, monkeypatch, committed_before_error
):
    store = MemoryTaskStore()
    store.submit("task", {"initial_temperature": 1.0, "num_particles": 8, "num_steps": 2}, {})
    job = store.claim()
    assert job is not None
    original_finish = store.finish

    def interrupted_finish(job, result, *, evidence=None):
        if evidence is not None:
            if committed_before_error:
                assert original_finish(job, result, evidence=evidence)
            raise TimeoutError("Archive acknowledgement unavailable")
        return original_finish(job, result)

    monkeypatch.setattr(store, "finish", interrupted_finish)
    manager = QueueManager(
        store,
        Settings(
            _env_file=None,
            artifacts_dir=tmp_path,
            redis_uri="",
            mlflow_tracking_uri="",
            otel_enabled=False,
        ),
        NullTracker(),
        Telemetry(),
    )
    try:
        await manager._execute(job)
    finally:
        await manager.stop()
    actual = store.get("task")
    archive = store.get_task_archive("task")
    expected = "completed" if committed_before_error else "failed"
    assert actual["status"] == archive["record"]["status"] == expected
    terminal = [
        event
        for event in store.timeline("task")["events"]
        if event["event_type"] in {"completed", "failed", "cancelled"}
    ]
    assert len(terminal) == 1 and terminal[0]["event_type"] == expected
    if committed_before_error:
        assert archive["evidence"] is not None
        assert all(
            store.get_artifact_archive(item["artifact_id"]) is not None
            for item in archive["evidence"]["artifacts"]
        )
    else:
        assert archive["evidence"] is None
