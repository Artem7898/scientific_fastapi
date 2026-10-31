"""Portable RO-Crate structure, byte-bound copies and offline verification."""

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.services import provenance_export as exporter
from scientific_api.services.analysis_service import create_velocity_analysis
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.provenance_service import ProvenanceError, ProvenanceService
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation


def _recorded(
    root: Path,
    model: str = "ou",
    *,
    formats: tuple[str, ...] | None = None,
    task_id: str = "crate-task",
):
    if model == "ou":
        parameters = SimulationParameters(1.0, 16, num_steps=4, seed=42)
    elif model == "langevin":
        parameters = LangevinParameters(1.0, 1.0, 16, num_steps=4, sample_stride=2, seed=42)
    else:
        parameters = LennardJonesParameters(num_particles=16, num_steps=4, sample_stride=2, seed=42)
    payload = asdict(parameters)
    version = {
        "ou": "ideal-gas-ou-v1",
        "langevin": "langevin-free-v1",
        "lj": "lennard-jones-2d-v1",
    }[model]
    if model != "ou":
        payload["model"] = version
    backend = MemoryTaskStore()
    backend.submit(task_id, payload, {})
    job = backend.claim()
    assert job is not None
    result = run_simulation(
        parameters, task_id=job.task_id, attempt_id=job.owner, artifacts_dir=root
    )
    envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
    assert backend.finish(job, result, evidence=build_archive_evidence(envelope))
    analysis = None
    if formats is not None:
        analysis = create_velocity_analysis(result, artifacts_dir=root, bins=8, formats=formats)
        analysis_envelope = read_result_envelope(
            result, artifacts_dir=root, analysis_id=analysis["analysis_id"], require_present=True
        )
        backend.register_analysis(
            job.task_id,
            job.owner,
            analysis["analysis_id"],
            analysis,
            evidence=build_archive_evidence(analysis_envelope),
        )
    return ProvenanceService(backend, root), result, analysis


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
def test_ro_crate_structure_content_hashes_and_offline_roundtrip(tmp_path, model):
    service, result, _ = _recorded(tmp_path, model)
    payload, filename = exporter.export_ro_crate(service, result["task_id"])
    assert filename == "scientific-api-crate-task-ro-crate.zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert all(
            not path.startswith("/") and ".." not in path.split("/") for path in archive.namelist()
        )
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        metadata = json.loads(archive.read("ro-crate-metadata.json"))
        nodes = {item["@id"]: item for item in metadata["@graph"]}
        assert metadata["@context"][0] == "https://w3id.org/ro/crate/1.1/context"
        assert nodes["ro-crate-metadata.json"]["conformsTo"] == {
            "@id": "https://w3id.org/ro/crate/1.1"
        }
        root = nodes["./"]
        assert root["@type"] == "Dataset"
        assert root["name"] and root["description"] and root["datePublished"]
        assert root["license"] == {"@id": "#data-license-unavailable"}
        assert "not supplied" in nodes["#data-license-unavailable"]["name"]
        assert "MIT" in nodes["#scientific-api"]["license"]
        assert not any(item.get("@type") == "Person" for item in nodes.values())
        members = set(archive.namelist()) - {"ro-crate-metadata.json"}
        assert {item["@id"] for item in root["hasPart"]} == members
        for path in members:
            data = archive.read(path)
            assert nodes[path]["sci:sha256"] == hashlib.sha256(data).hexdigest()
            assert int(nodes[path]["contentSize"]) == len(data)
            assert nodes[path]["encodingFormat"]
        manifest_path = result["artifacts"]["manifest"]
        assert archive.read(manifest_path) == (tmp_path / manifest_path).read_bytes()
        original_manifest = json.loads(archive.read(manifest_path))
        assert root["datePublished"] == original_manifest["created_at"]
        contract = service.export_snapshot(result["task_id"]).envelope.model_contract
        assert hashlib.sha256(archive.read("model-contract.json")).hexdigest() == contract.sha256
        graph = json.loads(archive.read("provenance.jsonld"))
        assert "@context" in graph and graph["@graph"]
        assert graph["evidence_status"] == "verified"
        history = json.loads(archive.read("timeline.json"))
        assert history["task_id"] == result["task_id"]
        assert history["events"][0]["event_type"] == "submitted"
        assert history["events"][-1]["event_type"] == "completed"
        restore = tmp_path / "restored"
        archive.extractall(restore)
    restored = read_result_envelope(result, artifacts_dir=restore, require_present=True)
    assert restored.integrity.status == "verified"
    assert restored.result.model_dump(mode="json") == result


def test_analysis_files_and_lineage_roundtrip(tmp_path):
    service, result, analysis = _recorded(tmp_path, formats=())
    payload, _ = exporter.export_ro_crate(service, result["task_id"])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = f"{result['task_id']}/{result['attempt_id']}/analyses/{analysis['analysis_id']}"
        assert f"{prefix}/analysis.json" in archive.namelist()
        assert f"{prefix}/manifest.json" in archive.namelist()
        assert f"{prefix}/envelope.json" in archive.namelist()
        metadata = json.loads(archive.read("ro-crate-metadata.json"))
        nodes = {item["@id"]: item for item in metadata["@graph"]}
        activity = nodes[f"#analysis-{analysis['analysis_id']}"]
        assert activity["prov:wasInformedBy"] == {"@id": "#simulation"}
        for path in (f"{prefix}/analysis.json", f"{prefix}/manifest.json"):
            assert nodes[path]["prov:wasGeneratedBy"] == {"@id": activity["@id"]}
            assert nodes[path]["sameAs"]["@id"].startswith("urn:scientific-api:artifact:")
        restore = tmp_path / "restored-analysis"
        archive.extractall(restore)
    restored = read_result_envelope(
        result, artifacts_dir=restore, analysis_id=analysis["analysis_id"], require_present=True
    )
    assert restored.result.analysis_id == analysis["analysis_id"]


def test_optional_png_and_pdf_are_included_without_license_inference(tmp_path):
    pytest.importorskip("matplotlib")
    service, result, analysis = _recorded(tmp_path, formats=("png", "pdf"))
    payload, _ = exporter.export_ro_crate(service, result["task_id"])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = f"{result['task_id']}/{result['attempt_id']}/analyses/{analysis['analysis_id']}"
        assert archive.read(f"{prefix}/velocity_distribution.png").startswith(b"\x89PNG")
        assert archive.read(f"{prefix}/report.pdf").startswith(b"%PDF")


def test_export_is_repeatable_for_one_unchanged_archived_snapshot(tmp_path):
    service, result, _ = _recorded(tmp_path, formats=())
    assert exporter.export_ro_crate(service, result["task_id"]) == exporter.export_ro_crate(
        service, result["task_id"]
    )


@pytest.mark.parametrize("task_id", ["CON", "NUL", "COM1"])
def test_recorded_paths_with_device_names_are_not_exported_as_portable_files(tmp_path, task_id):
    service, result, _ = _recorded(tmp_path, task_id=task_id)
    with pytest.raises(ProvenanceError) as caught:
        exporter.export_ro_crate(service, result["task_id"])
    assert caught.value.code == "PROVENANCE_INTEGRITY_FAILED"


@pytest.mark.parametrize("terminal", ["pending", "failed", "cancelled"])
def test_export_requires_completed_result_with_archived_evidence(tmp_path, terminal):
    backend = MemoryTaskStore()
    backend.submit("incomplete", {"initial_temperature": 1.0, "num_particles": 4}, {})
    if terminal != "pending":
        job = backend.claim()
        assert job is not None
        assert backend.finish(job, {"status": terminal, "error": "test"})
    with pytest.raises(ProvenanceError) as caught:
        exporter.export_ro_crate(ProvenanceService(backend, tmp_path), "incomplete")
    assert caught.value.code == "EXPORT_RESULT_UNAVAILABLE"
    assert caught.value.status_code == 409


@pytest.mark.parametrize("fault", ["changed", "symlink", "missing"])
def test_copy_rechecks_bytes_after_snapshot_without_reopening_unchecked(
    tmp_path, monkeypatch, fault
):
    service, result, _ = _recorded(tmp_path)
    original = service.export_snapshot
    state = tmp_path / result["artifacts"]["final_state"]

    def snapshot_then_change(task_id):
        snapshot = original(task_id)
        if fault == "changed":
            state.write_bytes(b"changed after verified snapshot")
        elif fault == "missing":
            state.unlink()
        else:
            target = tmp_path / "outside.npz"
            target.write_bytes(state.read_bytes())
            state.unlink()
            state.symlink_to(target)
        return snapshot

    monkeypatch.setattr(service, "export_snapshot", snapshot_then_change)
    with pytest.raises(ProvenanceError) as caught:
        exporter.export_ro_crate(service, result["task_id"])
    assert caught.value.code == "PROVENANCE_INTEGRITY_FAILED"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize(
    "setting,limit",
    [("MAX_EXPORT_FILES", 1), ("MAX_EXPORT_BYTES", 100), ("MAX_METADATA_BYTES", 100)],
)
def test_export_limits_are_explicit_and_never_truncate(tmp_path, monkeypatch, setting, limit):
    service, result, _ = _recorded(tmp_path)
    monkeypatch.setattr(exporter, setting, limit)
    with pytest.raises(ProvenanceError) as caught:
        exporter.export_ro_crate(service, result["task_id"])
    assert caught.value.code == "EXPORT_LIMIT_EXCEEDED"
    assert caught.value.status_code == 413


@pytest.mark.parametrize("task_id", ["unknown", "../outside", "x/y", "", "x" * 129])
def test_unknown_and_nonportable_task_ids_are_not_exported(tmp_path, task_id):
    with pytest.raises(ProvenanceError) as caught:
        exporter.export_ro_crate(ProvenanceService(MemoryTaskStore(), tmp_path), task_id)
    assert caught.value.status_code == 404
