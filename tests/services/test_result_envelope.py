"""Evidence sidecars verify bounded current bytes, without scientific overclaims."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.schemas.result_envelope import ResultEnvelope
from scientific_api.services import result_envelope_service as service
from scientific_api.services.analysis_service import create_velocity_analysis
from scientific_api.services.simulation_service import run_simulation


def _run(root: Path, *, model: str = "ou", attempt: str | None = "attempt") -> dict:
    if model == "ou":
        parameters = SimulationParameters(1.0, 8, num_steps=2, seed=42)
    elif model == "langevin":
        parameters = LangevinParameters(
            initial_temperature=1.0,
            bath_temperature=1.0,
            num_particles=8,
            time_step=0.1,
            num_steps=2,
            sample_stride=1,
            seed=42,
        )
    else:
        parameters = LennardJonesParameters(
            num_particles=16,
            num_steps=2,
            sample_stride=1,
            seed=42,
        )
    return run_simulation(parameters, task_id="task", attempt_id=attempt, artifacts_dir=root)


def _manifest(root: Path, result: dict) -> tuple[Path, dict]:
    path = root / result["artifacts"]["manifest"]
    return path, json.loads(path.read_text())


def _sidecar(root: Path, result: dict) -> Path:
    return (root / result["artifacts"]["manifest"]).with_name("envelope.json")


def _write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, allow_nan=False))


def _analysis(root: Path, result: dict) -> tuple[dict, Path]:
    analysis = create_velocity_analysis(result, artifacts_dir=root, bins=8, formats=())
    path = root / result["task_id"] / result["attempt_id"] / "analyses" / analysis["analysis_id"]
    return analysis, path


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
@pytest.mark.parametrize("attempt", ["attempt", None])
def test_all_model_and_standalone_records_hash_exact_manifest_without_changing_outputs(
    tmp_path, model, attempt
):
    result = _run(tmp_path, model=model, attempt=attempt)
    path, manifest = _manifest(tmp_path, result)
    envelope = service.read_result_envelope(result, artifacts_dir=tmp_path, require_present=True)
    assert envelope.integrity.status == "verified"
    assert envelope.attempt_id == attempt
    assert envelope.result.model_dump() == result == manifest["outputs"]
    assert envelope.manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert envelope.authenticity == "unsigned"
    assert envelope.scientific_verification == "not_performed"
    assert envelope.model_contract.snapshot.model_version == result["model_version"]
    assert ResultEnvelope.model_validate(envelope.model_dump(mode="json")) == envelope
    assert not list(tmp_path.rglob(".envelope-*"))
    for artifact in envelope.artifacts:
        data = (tmp_path / artifact.path).read_bytes()
        assert artifact.sha256 == hashlib.sha256(data).hexdigest()
        assert artifact.bytes == len(data)


@pytest.mark.parametrize(
    "model,availability", [("ou", "available"), ("langevin", "partial"), ("lj", "unavailable")]
)
def test_uncertainty_is_model_specific_without_fabricated_lj_errors(tmp_path, model, availability):
    result = _run(tmp_path, model=model)
    uncertainty = service.read_result_envelope(result, artifacts_dir=tmp_path).uncertainty
    assert uncertainty.availability == availability
    assert uncertainty.reason
    if model == "lj":
        assert "standard_error" not in uncertainty.details
        assert "correlated" in uncertainty.details["time_statistics"]
    if model == "langevin":
        assert uncertainty.details["diffusion"]["standard_error"] is None
        assert uncertainty.details["diffusion"]["reason"]


def test_legacy_pre_backend_record_returns_not_verified_without_modern_metadata(tmp_path):
    result = _run(tmp_path)
    path, manifest = _manifest(tmp_path, result)
    manifest.pop("result_envelope")
    manifest.pop("execution")
    manifest["parameters"].pop("backend")
    _write(path, manifest)
    _sidecar(tmp_path, result).unlink()
    envelope = service.read_result_envelope(result, artifacts_dir=tmp_path)
    assert envelope.result.model_dump() == result
    assert envelope.integrity.status == "not_verified"
    assert envelope.integrity.reason == "envelope_not_recorded"
    assert envelope.manifest is None and envelope.artifacts == []
    assert envelope.model_contract is None
    with pytest.raises(service.ResultEnvelopeError, match="ENVELOPE_MISSING"):
        service.read_result_envelope(result, artifacts_dir=tmp_path, require_present=True)


@pytest.mark.parametrize("target", ["manifest", "final_state", "envelope"])
def test_current_bytes_are_rechecked_on_every_call_without_repair(tmp_path, target):
    result = _run(tmp_path)
    original = service.read_result_envelope(result, artifacts_dir=tmp_path)
    path = (
        _sidecar(tmp_path, result)
        if target == "envelope"
        else tmp_path / result["artifacts"][target]
    )
    raw = path.read_bytes()
    path.write_bytes(b"corrupt /private/secret/value")
    with pytest.raises(service.ResultEnvelopeError) as caught:
        service.read_result_envelope(result, artifacts_dir=tmp_path)
    assert "/private" not in str(caught.value)
    assert caught.value.code == str(caught.value)
    assert path.read_bytes() == b"corrupt /private/secret/value"
    path.write_bytes(raw)
    assert service.read_result_envelope(result, artifacts_dir=tmp_path) == original


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_marker",
        "missing_sidecar",
        "wrong_marker",
        "extra_record",
        "duplicate_record",
        "wrong_size",
        "wrong_path",
    ],
)
def test_inconsistent_metadata_is_never_treated_as_historical(tmp_path, mutation):
    result = _run(tmp_path)
    path, manifest = _manifest(tmp_path, result)
    if mutation == "remove_marker":
        manifest.pop("result_envelope")
    elif mutation == "missing_sidecar":
        _sidecar(tmp_path, result).unlink()
    elif mutation == "wrong_marker":
        manifest["result_envelope"]["path"] = "other/envelope.json"
    elif mutation == "extra_record":
        manifest["artifacts"].append({"path": "surprise", "sha256": "0" * 64})
    elif mutation == "duplicate_record":
        manifest["artifacts"].append(manifest["artifacts"][0])
    elif mutation == "wrong_size":
        manifest["artifacts"][0]["bytes"] = True
    else:
        manifest["artifacts"][0]["path"] = "../outside.npz"
    _write(path, manifest)
    with pytest.raises(service.ResultEnvelopeError):
        service.read_result_envelope(result, artifacts_dir=tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1,"a":2}',
        '{"environment":{"x":NaN}}',
        '{"environment":{"x":Infinity}}',
        '{"environment":{"x":1e309}}',
    ],
)
def test_json_rejects_duplicate_keys_nonfinite_and_overflowing_literals(tmp_path, text):
    result = _run(tmp_path)
    path, _ = _manifest(tmp_path, result)
    path.write_text(text)
    with pytest.raises(service.ResultEnvelopeError):
        service.read_result_envelope(result, artifacts_dir=tmp_path)


@pytest.mark.parametrize("component", ["manifest", "state", "parent"])
def test_symbolic_links_are_rejected_for_every_artifact_path_component(tmp_path, component):
    result = _run(tmp_path)
    path = tmp_path / result["artifacts"]["manifest" if component == "manifest" else "final_state"]
    if component == "parent":
        parent = path.parent
        moved = parent.with_name("moved")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    else:
        moved = path.with_suffix(".copied")
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(service.ResultEnvelopeError):
        service.read_result_envelope(result, artifacts_dir=tmp_path)


def test_nonregular_fifo_and_oversize_json_are_rejected_without_loading_arrays(
    tmp_path, monkeypatch
):
    result = _run(tmp_path)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Integrity checks must not decode arrays")
    )
    assert (
        service.read_result_envelope(result, artifacts_dir=tmp_path).integrity.status == "verified"
    )
    state = tmp_path / result["artifacts"]["final_state"]
    state.unlink()
    os.mkfifo(state)
    with pytest.raises(service.ResultEnvelopeError, match="ARTIFACT_SIZE_INVALID"):
        service.read_result_envelope(result, artifacts_dir=tmp_path)
    monkeypatch.setattr(service, "MAX_JSON_BYTES", 10)
    with pytest.raises(service.ResultEnvelopeError, match="ARTIFACT_SIZE_INVALID"):
        service.read_result_envelope(result, artifacts_dir=tmp_path)


def test_historical_contract_snapshot_survives_documentation_metadata_changes(
    tmp_path, monkeypatch
):
    result = _run(tmp_path)
    original = service.read_result_envelope(result, artifacts_dir=tmp_path)
    get_contract = service.get_model_contract
    monkeypatch.setattr(
        service,
        "get_model_contract",
        lambda version: get_contract(version).model_copy(
            update={"name": "Improved documentation label"}
        ),
    )
    assert service.read_result_envelope(result, artifacts_dir=tmp_path) == original
    path = _sidecar(tmp_path, result)
    content = json.loads(path.read_text())
    content["model_contract"]["snapshot"]["name"] = "Changed without updating digest"
    _write(path, content)
    with pytest.raises(service.ResultEnvelopeError, match="MODEL_CONTRACT_MISMATCH"):
        service.read_result_envelope(result, artifacts_dir=tmp_path)


def test_analysis_envelope_is_separate_and_links_current_source_manifest_and_state(tmp_path):
    result = _run(tmp_path)
    original = service.read_result_envelope(result, artifacts_dir=tmp_path)
    analysis, directory = _analysis(tmp_path, result)
    envelope = service.read_result_envelope(
        result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"]
    )
    assert envelope.result.model_dump() == analysis
    assert envelope.source["manifest_sha256"] == original.manifest.sha256
    assert envelope.manifest.path.endswith(f"{analysis['analysis_id']}/manifest.json")
    assert service.read_result_envelope(result, artifacts_dir=tmp_path) == original
    assert service.read_result_envelope(analysis, artifacts_dir=tmp_path) == envelope
    state = tmp_path / result["artifacts"]["final_state"]
    state.write_bytes(b"changed source state")
    with pytest.raises(service.ResultEnvelopeError):
        service.read_result_envelope(
            result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"]
        )
    assert (directory / "envelope.json").is_file()


@pytest.mark.parametrize(
    "missing", [("execution",), ("execution", "rng", "environment", "uncertainty")]
)
def test_new_analysis_of_historical_run_preserves_unknown_source_metadata(tmp_path, missing):
    result = _run(tmp_path)
    path, manifest = _manifest(tmp_path, result)
    manifest.pop("result_envelope")
    manifest["parameters"].pop("backend")
    for name in missing:
        manifest.pop(name)
    _write(path, manifest)
    _sidecar(tmp_path, result).unlink()
    source_bytes = path.read_bytes()
    analysis, _ = _analysis(tmp_path, result)
    envelope = service.read_result_envelope(
        result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"]
    )
    assert envelope.integrity.status == "verified"
    assert envelope.execution == {}
    assert envelope.source["manifest_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert path.read_bytes() == source_bytes
    if "rng" in missing:
        assert envelope.rng == {}
        assert envelope.uncertainty.availability == "unavailable"
        assert envelope.uncertainty.reason == "source_manifest_does_not_record_uncertainty_metadata"


def test_analysis_detects_file_change_between_digest_and_json_decode(tmp_path, monkeypatch):
    result = _run(tmp_path)
    analysis, directory = _analysis(tmp_path, result)
    snapshot = service._snapshot
    changed = False

    def mutate(root, path, maximum, *, retain=False):
        nonlocal changed
        content = snapshot(root, path, maximum, retain=retain)
        if path.endswith("/analysis.json") and not retain and not changed:
            changed = True
            data = json.loads((directory / "analysis.json").read_text())
            data["concurrent_change"] = True
            _write(directory / "analysis.json", data)
        return content

    monkeypatch.setattr(service, "_snapshot", mutate)
    with pytest.raises(service.ResultEnvelopeError, match="ARTIFACT_INTEGRITY_FAILED"):
        service.read_result_envelope(
            result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"]
        )


def test_analysis_detects_source_manifest_change_between_reads(tmp_path, monkeypatch):
    result = _run(tmp_path)
    analysis, _ = _analysis(tmp_path, result)
    read_manifest = service._manifest
    changed = False

    def mutate(root, prefix):
        nonlocal changed
        record = read_manifest(root, prefix)
        if prefix == "task/attempt" and not changed:
            changed = True
            path, content = _manifest(root, result)
            content["environment"]["concurrent_change"] = True
            _write(path, content)
        return record

    monkeypatch.setattr(service, "_manifest", mutate)
    with pytest.raises(service.ResultEnvelopeError, match="SOURCE_INTEGRITY_FAILED"):
        service.read_result_envelope(
            result, artifacts_dir=tmp_path, analysis_id=analysis["analysis_id"]
        )


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_noninteger_envelope_schema_versions_are_not_silently_canonicalized(tmp_path, version):
    result = _run(tmp_path)
    path = _sidecar(tmp_path, result)
    content = json.loads(path.read_text())
    content["schema_version"] = version
    _write(path, content)
    with pytest.raises(service.ResultEnvelopeError):
        service.read_result_envelope(result, artifacts_dir=tmp_path)
