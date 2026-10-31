"""Archived occurrence binding, partial history and bounded PROV lineage."""

from dataclasses import asdict
from typing import cast

import pytest

from scientific_api.core import task_store
from scientific_api.core.task_store import MemoryTaskStore, TaskStore
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.services import provenance_service
from scientific_api.services.analysis_service import create_velocity_analysis
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.provenance_service import ProvenanceError, ProvenanceService
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation


def completed(tmp_path, parameters=None, store=None, *, evidence=True):
    parameters = parameters or SimulationParameters(2.0, 8, num_steps=4, seed=2**64 - 1)
    store = store or MemoryTaskStore()
    payload = asdict(parameters)
    if isinstance(parameters, LangevinParameters):
        payload["model"] = "langevin-free-v1"
    elif isinstance(parameters, LennardJonesParameters):
        payload["model"] = "lennard-jones-2d-v1"
    store.submit("experiment", payload, {})
    job = store.claim()
    assert job is not None
    result = run_simulation(
        parameters,
        task_id=job.task_id,
        attempt_id=job.owner,
        artifacts_dir=tmp_path,
    )
    envelope = read_result_envelope(result, artifacts_dir=tmp_path, require_present=True)
    assert store.finish(
        job, result, evidence=build_archive_evidence(envelope) if evidence else None
    )
    return store, result, envelope


@pytest.mark.parametrize(
    "parameters",
    [
        SimulationParameters(2.0, 8, num_steps=4),
        LangevinParameters(2.0, 1.0, 8, num_steps=4, sample_stride=1),
        LennardJonesParameters(num_particles=16, num_steps=4, sample_stride=1, rdf_bins=8),
    ],
)
def test_graph_covers_actual_parameters_model_code_rng_backend_and_committed_artifacts(
    tmp_path, parameters
):
    store, result, envelope = completed(tmp_path, parameters)
    service = ProvenanceService(store, tmp_path)
    snapshot = service.export_snapshot("experiment")
    assert snapshot.task.record == result
    assert snapshot.task.parameters == envelope.parameters
    assert snapshot.task.model_version == result["model_version"]
    assert snapshot.task.evidence_status == "verified"
    assert snapshot.task.coverage == "complete"
    assert {item.name for item in snapshot.task.artifacts} == set(result["artifacts"])
    assert all(item.attempt_id == result["attempt_id"] for item in snapshot.task.artifacts)
    graph = snapshot.graph.model_dump(mode="json", by_alias=True)
    assert graph["@context"]["prov"] == "http://www.w3.org/ns/prov#"
    nodes = {node["@id"]: node for node in graph["@graph"]}
    assert any(identifier.startswith("urn:scientific-api:model-contract:") for identifier in nodes)
    assert any(identifier.startswith("urn:scientific-api:backend:") for identifier in nodes)
    assert any(identifier.startswith("urn:scientific-api:rng:") for identifier in nodes)
    assert any(identifier.startswith("urn:scientific-api:environment:") for identifier in nodes)
    software = [node for node in nodes.values() if node["@type"] == "prov:SoftwareAgent"]
    assert len(software) == 1
    assert software[0]["sf:sourceSha256"] == envelope.environment["source_sha256"]
    activities = [node for node in nodes.values() if node["@type"] == "prov:Activity"]
    assert len(activities) == 1
    assert len(activities[0]["prov:used"]) == 5
    assert activities[0]["prov:wasAssociatedWith"]["@id"] == software[0]["@id"]
    for artifact in snapshot.task.artifacts:
        node = nodes[f"urn:scientific-api:artifact:{artifact.artifact_id}"]
        assert node["sf:sha256"] == artifact.sha256
        assert node["prov:wasGeneratedBy"]["@id"] == activities[0]["@id"]
        assert service.artifact(artifact.artifact_id) == artifact
    assert all(node["@type"] != "prov:Person" for node in nodes.values())
    assert [event.event_type for event in snapshot.timeline] == [
        "submitted",
        "attempt_claimed",
        "completed",
    ]


def test_analysis_graph_binds_saved_source_and_optional_reports_without_rerunning(tmp_path):
    store, result, _ = completed(tmp_path)
    analysis = create_velocity_analysis(
        result, artifacts_dir=tmp_path, bins=8, formats=("png", "pdf"), provenance_store=store
    )
    service = ProvenanceService(store, tmp_path)
    summary = service.analysis(analysis["analysis_id"])
    assert summary.record == analysis
    assert summary.evidence_status == "verified"
    assert {item.name for item in summary.artifacts} == {"manifest", "analysis", "png", "pdf"}
    assert summary.source["attempt_id"] == result["attempt_id"]
    nodes = service.graph("experiment").nodes
    activity = next(node for node in nodes if node.get("sf:analysisId") == analysis["analysis_id"])
    assert activity["@type"] == "prov:Activity"
    assert len(activity["prov:used"]) == 3  # source manifest, state, analysis environment
    assert "prov:wasAssociatedWith" in activity
    for artifact in summary.artifacts:
        node = next(item for item in nodes if item["@id"].endswith(artifact.artifact_id))
        assert node["prov:wasGeneratedBy"]["@id"] == activity["@id"]
        assert len(node["prov:wasDerivedFrom"]) == 2
        assert service.artifact(artifact.artifact_id) == artifact


def test_new_analysis_of_prejournal_parent_remains_verified_with_partial_parent_history(tmp_path):
    store, result, _ = completed(tmp_path, evidence=False)
    # Emulate a live completed task created before journal support: registration
    # observes its current record, rather than fabricating historical events.
    store._archives.clear()
    store._journals.clear()
    analysis = create_velocity_analysis(
        result, artifacts_dir=tmp_path, bins=8, formats=(), provenance_store=store
    )
    service = ProvenanceService(store, tmp_path)
    task = service.task("experiment")
    assert task.coverage == "partial"
    assert task.evidence_status == "unavailable"
    assert task.artifacts == []
    assert len(task.analyses) == 1
    assert task.analyses[0].evidence_status == "verified"
    assert service.analysis(analysis["analysis_id"]).record == analysis
    graph = service.graph("experiment")
    source_nodes = [node for node in graph.nodes if "sf:historicalParentAnchor" in node]
    assert len(source_nodes) == 2
    assert all(node["sf:historicalParentAnchor"] == "unavailable" for node in source_nodes)
    assert any(node["@id"].startswith("urn:scientific-api:model-contract:") for node in graph.nodes)
    assert not any(node["@id"].startswith("urn:scientific-api:software:") for node in graph.nodes)
    assert [item.event_type for item in service.timeline("experiment").events] == [
        "history_observed",
        "analysis_completed",
    ]


def test_operational_ttl_and_get_status_do_not_control_archive_access(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(task_store, "_now", lambda: now[0])
    store, result, _ = completed(tmp_path, store=MemoryTaskStore(ttl_seconds=1))
    now[0] += 2
    assert store.get("experiment") is None

    def no_operational_status(*args, **kwargs):
        pytest.fail("Provenance must not depend on expiring operational status")

    monkeypatch.setattr(store, "get", no_operational_status)
    service = ProvenanceService(store, tmp_path)
    task = service.task("experiment")
    assert task.record == result
    assert service.artifact(task.artifacts[0].artifact_id).task_id == "experiment"
    assert len(service.timeline("experiment").events) == 3


def test_missing_archive_never_imports_matching_files(tmp_path):
    _, result, _ = completed(tmp_path)
    service = ProvenanceService(MemoryTaskStore(), tmp_path)
    assert (tmp_path / result["artifacts"]["manifest"]).exists()
    with pytest.raises(ProvenanceError) as caught:
        service.task("experiment")
    assert caught.value.code == "PROVENANCE_NOT_FOUND"
    assert caught.value.status_code == 404


def test_completed_without_archival_evidence_does_not_guess_saved_files(tmp_path, monkeypatch):
    store, _, _ = completed(tmp_path, evidence=False)

    def no_file_read(*args, **kwargs):
        pytest.fail("Unanchored historical files must not be imported")

    monkeypatch.setattr(provenance_service, "read_result_envelope", no_file_read)
    task = ProvenanceService(store, tmp_path).task("experiment")
    assert task.status == "completed"
    assert task.evidence_status == "unavailable"
    assert task.artifacts == []
    assert task.manifest_sha256 is None


@pytest.mark.parametrize("status", ["pending", "running", "failed", "cancelled"])
def test_unfinished_or_failed_history_has_no_invented_evidence(tmp_path, status):
    store = MemoryTaskStore()
    store.submit("experiment", {"initial_temperature": 1.0, "num_particles": 8}, {})
    if status != "pending":
        job = store.claim()
        assert job is not None
        if status in {"failed", "cancelled"}:
            assert store.finish(job, {"task_id": "experiment", "status": status, "error": status})
    task = ProvenanceService(store, tmp_path).task("experiment")
    assert task.status == status
    assert task.evidence_status == "unavailable"
    assert task.environment == task.execution == task.rng == {}
    assert task.artifacts == []


def test_retried_attempts_do_not_inherit_final_attempt_runtime_attestation(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(task_store, "_now", lambda: now[0])
    store = MemoryTaskStore(lease_seconds=1)
    parameters = SimulationParameters(1.0, 8, num_steps=2)
    store.submit("experiment", asdict(parameters), {})
    original = store.claim()
    assert original is not None
    now[0] += 2  # An expired lease retries; orderly release is cancellation.
    final = store.claim()
    assert final is not None and original.owner != final.owner
    result = run_simulation(
        parameters, task_id="experiment", attempt_id=final.owner, artifacts_dir=tmp_path
    )
    envelope = read_result_envelope(result, artifacts_dir=tmp_path, require_present=True)
    assert store.finish(final, result, evidence=build_archive_evidence(envelope))
    nodes = ProvenanceService(store, tmp_path).graph("experiment").nodes
    activities = [node for node in nodes if node["@type"] == "prov:Activity"]
    old = next(node for node in activities if node["sf:attemptId"] == original.owner)
    new = next(node for node in activities if node["sf:attemptId"] == final.owner)
    assert len(old["prov:used"]) == 1
    assert "prov:wasAssociatedWith" not in old
    assert len(new["prov:used"]) == 5
    assert "prov:wasAssociatedWith" in new


def test_snapshot_event_cutoff_excludes_completion_after_archive_read(tmp_path, monkeypatch):
    store = MemoryTaskStore()
    store.submit("experiment", {"num_steps": 2}, {})
    original_read = store.get_task_archive

    def read_then_complete(task_id):
        archived = original_read(task_id)
        job = store.claim()
        assert job is not None
        assert store.finish(job, {"task_id": task_id, "status": "failed", "error": "failed"})
        return archived

    monkeypatch.setattr(store, "get_task_archive", read_then_complete)
    snapshot = ProvenanceService(store, tmp_path).export_snapshot("experiment")
    assert snapshot.task.status == "pending"
    assert snapshot.task.attempts == []
    assert [event.event_type for event in snapshot.timeline] == ["submitted"]
    assert [node["sf:eventType"] for node in snapshot.graph.nodes if "sf:eventType" in node] == [
        "submitted"
    ]
    assert len(store.timeline("experiment")["events"]) == 3


def test_snapshot_event_cutoff_excludes_analysis_registered_after_archive_read(
    tmp_path, monkeypatch
):
    store, result, _ = completed(tmp_path)
    original_read = store.get_task_archive

    def read_then_analyze(task_id):
        archived = original_read(task_id)
        create_velocity_analysis(
            result, artifacts_dir=tmp_path, bins=8, formats=(), provenance_store=store
        )
        return archived

    monkeypatch.setattr(store, "get_task_archive", read_then_analyze)
    snapshot = ProvenanceService(store, tmp_path).export_snapshot("experiment")
    assert snapshot.task.status == "completed"
    assert snapshot.task.analyses == []
    assert [event.event_type for event in snapshot.timeline] == [
        "submitted",
        "attempt_claimed",
        "completed",
    ]
    assert not any("sf:analysisId" in node for node in snapshot.graph.nodes)
    assert store.timeline("experiment")["events"][-1]["event_type"] == "analysis_completed"


def test_snapshot_requires_its_captured_journal_cutoff_to_exist(tmp_path, monkeypatch):
    store = MemoryTaskStore()
    store.submit("experiment", {}, {})
    archive = store.get_task_archive("experiment")
    archive["last_event_id"] = "999-0"
    monkeypatch.setattr(store, "get_task_archive", lambda task_id: archive)
    with pytest.raises(ProvenanceError, match="PROVENANCE_INTEGRITY_FAILED"):
        ProvenanceService(store, tmp_path).task("experiment")


def test_changed_bytes_and_sidecar_fail_relative_to_committed_archive(tmp_path):
    store, result, _ = completed(tmp_path)
    state = tmp_path / result["artifacts"]["final_state"]
    state.write_bytes(state.read_bytes() + b"modified")
    with pytest.raises(ProvenanceError) as caught:
        ProvenanceService(store, tmp_path).graph("experiment")
    assert caught.value.code == "PROVENANCE_INTEGRITY_FAILED"
    assert str(tmp_path) not in str(caught.value)


def test_matching_rewritten_files_cannot_replace_archived_binding(tmp_path, monkeypatch):
    store, _, envelope = completed(tmp_path)
    changed = envelope.model_copy(deep=True)
    changed.environment["processor"] = "changed-but-internally-consistent"
    changed.manifest.sha256 = "a" * 64
    next(item for item in changed.artifacts if item.name == "manifest").sha256 = "a" * 64
    monkeypatch.setattr(provenance_service, "read_result_envelope", lambda *args, **kwargs: changed)
    with pytest.raises(ProvenanceError, match="PROVENANCE_INTEGRITY_FAILED"):
        ProvenanceService(store, tmp_path).task("experiment")


def test_timeline_pages_are_ordered_and_do_not_require_artifacts(tmp_path):
    store = MemoryTaskStore()
    store.submit("experiment", {"num_steps": 2}, {})
    job = store.claim()
    assert job is not None
    assert store.finish(job, {"task_id": "experiment", "status": "failed", "error": "failed"})
    service = ProvenanceService(store, tmp_path)
    page = service.timeline("experiment", limit=2)
    assert [event.event_type for event in page.events] == ["submitted", "attempt_claimed"]
    assert page.has_more and page.next_cursor == page.events[-1].event_id
    last = service.timeline("experiment", after_id=page.next_cursor, limit=2)
    assert [event.event_type for event in last.events] == ["failed"]
    assert not last.has_more
    assert service.timeline("experiment", after_id=last.next_cursor).events == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": 257},
        {"limit": True},
        {"after_id": "../1"},
        {"after_id": "1-0-0"},
        {"after_id": f"{2**64}-0"},
    ],
)
def test_timeline_rejects_invalid_cursors_and_limits_before_backend(tmp_path, kwargs):
    service = ProvenanceService(cast(TaskStore, object()), tmp_path)
    with pytest.raises(ProvenanceError) as caught:
        service.timeline("experiment", **kwargs)
    assert caught.value.code == "INVALID_TIMELINE_QUERY"
    assert caught.value.status_code == 422


@pytest.mark.parametrize(
    "method,identifier", [("task", "../bad"), ("analysis", "bad"), ("artifact", "bad")]
)
def test_invalid_resource_identifiers_never_access_backend(tmp_path, method, identifier):
    service = ProvenanceService(cast(TaskStore, object()), tmp_path)
    with pytest.raises(ProvenanceError) as caught:
        getattr(service, method)(identifier)
    assert caught.value.status_code == 404


def test_analysis_and_graph_limits_fail_explicitly_without_silent_truncation(tmp_path, monkeypatch):
    store, _, _ = completed(tmp_path)
    service = ProvenanceService(store, tmp_path)
    monkeypatch.setattr(provenance_service, "MAX_GRAPH_NODES", 3)
    with pytest.raises(ProvenanceError) as caught:
        service.graph("experiment")
    assert caught.value.code == "PROVENANCE_LIMIT_EXCEEDED"
    assert caught.value.status_code == 413
    monkeypatch.setattr(provenance_service, "MAX_GRAPH_NODES", 4096)
    archive = store.get_task_archive("experiment")
    archive["analyses"] = [{}] * 257
    monkeypatch.setattr(store, "get_task_archive", lambda task_id: archive)
    with pytest.raises(ProvenanceError) as caught:
        service.task("experiment")
    assert caught.value.status_code == 413


def test_oversized_metadata_and_corrupt_locator_fail_safely(tmp_path, monkeypatch):
    store, _, _ = completed(tmp_path)
    service = ProvenanceService(store, tmp_path)
    artifact = service.task("experiment").artifacts[0]
    locator = store.get_artifact_archive(artifact.artifact_id)
    locator["descriptor"]["sha256"] = "a" * 64
    monkeypatch.setattr(store, "get_artifact_archive", lambda artifact_id: locator)
    with pytest.raises(ProvenanceError, match="PROVENANCE_INTEGRITY_FAILED"):
        service.artifact(artifact.artifact_id)
    monkeypatch.setattr(provenance_service, "MAX_METADATA_BYTES", 20)
    with pytest.raises(ProvenanceError) as caught:
        service.task("experiment")
    assert caught.value.status_code == 413


def test_archive_storage_failure_does_not_leak_internal_error(tmp_path):
    class BrokenStore:
        def get_task_archive(self, task_id):
            raise RuntimeError("private-credential-and-host")

    service = ProvenanceService(cast(TaskStore, BrokenStore()), tmp_path)
    with pytest.raises(ProvenanceError) as caught:
        service.task("experiment")
    assert caught.value.code == "PROVENANCE_STORAGE_UNAVAILABLE"
    assert caught.value.status_code == 503
    assert "private" not in str(caught.value)


def test_partial_unknown_archive_preserves_unknowns_and_does_not_invent_model(tmp_path):
    class HistoricalStore:
        def get_task_archive(self, task_id):
            return {
                "task_id": task_id,
                "payload": {},
                "record": {},
                "submitted_at": None,
                "ensemble_id": None,
                "evidence": None,
                "attempts": [],
                "analyses": [],
                "coverage": "partial",
                "storage": "redis",
            }

        def timeline(self, task_id, after_id=None, limit=100):
            return {"task_id": task_id, "events": [], "next_cursor": None, "has_more": False}

    service = ProvenanceService(cast(TaskStore, HistoricalStore()), tmp_path)
    task = service.task("historical")
    assert task.model_version is None
    assert task.status == "unknown"
    assert task.submitted_at is None
    assert task.attempts == task.artifacts == []
    assert len(service.graph("historical").nodes) == 2
