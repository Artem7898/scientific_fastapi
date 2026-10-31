"""LJ research pins survive TTL and reject cross-source or changed byte snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import numpy as np
import pytest

from scientific_api.core import task_store
from scientific_api.core.task_store import MemoryTaskStore
from scientific_api.domain.checkpoint import checkpoint_parameters_sha256
from scientific_api.domain.experiments import parameters_from_payload
from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled
from scientific_api.models.lennard_jones_engine import LennardJonesSimulator
from scientific_api.services import lj_research_source as service
from scientific_api.services.lennard_jones_service import run_lj_simulation
from scientific_api.services.provenance_evidence import build_archive_evidence
from scientific_api.services.result_envelope_service import (
    publish_result_envelope,
    read_result_envelope,
)
from scientific_api.services.verification_service import canonical_json


def replace_observables(root, result, raw):
    path = root / result["artifacts"]["observables"]
    path.write_bytes(raw)
    manifest_path = root / result["artifacts"]["manifest"]
    manifest = json.loads(manifest_path.read_bytes())
    descriptor = next(
        item for item in manifest["artifacts"] if item["path"] == result["artifacts"]["observables"]
    )
    descriptor.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest_path.write_text(json.dumps(manifest))
    publish_result_envelope(result, artifacts_dir=root)


def archived_source(root, *, store=None, task_id="source", mutate=None, **changes):
    store = store if store is not None else MemoryTaskStore(ttl_seconds=1)
    parameters = LennardJonesParameters(
        **{
            "num_particles": 16,
            "box_size": (8.0, 8.0),
            "initial_temperature": 0.1,
            "num_steps": 6,
            "sample_stride": 2,
            "equilibration_steps": 1,
            "seed": 2**64 - 1,
            **changes,
        }
    )
    store.submit(task_id, {"model": LJ_MODEL_VERSION, **asdict(parameters)}, {})
    job = store.claim()
    result = run_lj_simulation(
        parameters, task_id=task_id, attempt_id=job.owner, artifacts_dir=root
    )
    if mutate is not None:
        raw = (root / result["artifacts"]["observables"]).read_bytes()
        replace_observables(root, result, mutate(raw))
    envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
    assert store.finish(job, result, evidence=build_archive_evidence(envelope))
    return store, result, store.get_task_archive(task_id)


def load(root, pin, *, check_cancelled=lambda: None):
    return service.load_lj_research_source(pin, artifacts_dir=root, check_cancelled=check_cancelled)


def assert_error(code, status_code, function, *args, **kwargs):
    with pytest.raises(service.ResearchSourceError) as error:
        function(*args, **kwargs)
    assert error.value.code == code
    assert error.value.status_code == status_code


def test_sources_remain_available_after_status_ttl_without_source_or_archive_writes(
    tmp_path, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(task_store, "_now", lambda: clock[0])
    store, result, archive = archived_source(tmp_path)
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    original_history = store.timeline("source")
    clock[0] += 2
    assert store.get("source") is None
    pins = service.prepare_lj_research_sources(store, ["source"])
    assert pins[0]["parameters"]["seed"] == 2**64 - 1
    snapshot = load(tmp_path, pins[0])
    assert snapshot.parameters.seed == 2**64 - 1
    assert snapshot.observables["sample_steps"] == [0, 2, 4, 6]
    assert snapshot.envelope.manifest.sha256 == pins[0]["source"]["manifest_sha256"]
    assert store.get_task_archive("source") == archive
    assert store.timeline("source") == original_history
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == original_files
    with pytest.raises(FrozenInstanceError):
        snapshot.parameters = LennardJonesParameters()
    assert result["model_version"] == LJ_MODEL_VERSION


def test_admission_reads_only_archive_and_detaches_all_source_pins(tmp_path, monkeypatch):
    store, result, archive = archived_source(tmp_path)
    (tmp_path / result["artifacts"]["observables"]).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Admission must not inspect present-day artifact files")

    monkeypatch.setattr(service, "read_result_envelope", forbidden)
    monkeypatch.setattr(service, "_snapshot", forbidden)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    pin["parameters"]["seed"] = 0
    pin["evidence"]["artifacts"][0]["bytes"] = 0
    assert store.get_task_archive("source") == archive


def test_loading_never_decodes_npz_or_advances_a_model(tmp_path, monkeypatch):
    store, _, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]

    def forbidden(*args, **kwargs):
        pytest.fail("Research source loading must not decode NPZ or execute a simulation")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(LennardJonesSimulator, "advance", forbidden)
    snapshot = load(tmp_path, pin)
    assert set(snapshot.observables["series"]) == set(service.SERIES_NAMES)


@pytest.mark.parametrize("state", ["pending", "running", "failed", "cancelled"])
def test_admission_requires_committed_completed_lj_source(state):
    store = MemoryTaskStore()
    store.submit("source", {}, {})
    job = store.claim() if state != "pending" else None
    if state in {"failed", "cancelled"}:
        assert store.finish(job, {"status": state})
    assert_error(
        "RESEARCH_SOURCE_NOT_COMPLETED", 409, service.prepare_lj_research_sources, store, ["source"]
    )


def test_missing_non_lj_and_unavailable_evidence_have_explicit_admission_errors(tmp_path):
    assert_error(
        "RESEARCH_SOURCE_NOT_FOUND",
        404,
        service.prepare_lj_research_sources,
        MemoryTaskStore(),
        ["missing"],
    )
    store = MemoryTaskStore()
    store.submit("other", {}, {})
    assert store.finish(store.claim(), {"status": "completed", "model_version": "ideal-gas-ou-v1"})
    unsupported = store.get_task_archive("other")
    unsupported["evidence"] = {}
    assert_error(
        "RESEARCH_SOURCE_MODEL_UNSUPPORTED",
        409,
        service.prepare_lj_research_sources,
        SimpleNamespace(get_task_archive=lambda _: unsupported),
        ["other"],
    )
    _, _, archive = archived_source(tmp_path)
    archive["evidence"] = None
    backend = SimpleNamespace(get_task_archive=lambda _: archive)
    assert_error(
        "RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE",
        409,
        service.prepare_lj_research_sources,
        backend,
        ["source"],
    )


def test_completed_legacy_lj_without_result_model_version_has_explicit_unavailable_evidence():
    archive = {
        "task_id": "source",
        "record": {"task_id": "source", "status": "completed"},
        "payload": {"model": LJ_MODEL_VERSION},
        "evidence": None,
    }
    assert_error(
        "RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE",
        409,
        service.prepare_lj_research_sources,
        SimpleNamespace(get_task_archive=lambda _: archive),
        ["source"],
    )


def test_backend_failure_is_safe_503_without_private_exception_text():
    def broken(_task):
        raise RuntimeError("private-server-address-must-not-leak")

    backend = SimpleNamespace(get_task_archive=broken)
    with pytest.raises(service.ResearchSourceError) as error:
        service.prepare_lj_research_sources(backend, ["source"])
    assert error.value.status_code == 503
    assert str(error.value) == "RESEARCH_STORAGE_UNAVAILABLE"


@pytest.mark.parametrize(
    "tasks,code,status",
    [
        ([], "RESEARCH_INPUT_LIMIT_EXCEEDED", 413),
        ([str(i) for i in range(51)], "RESEARCH_INPUT_LIMIT_EXCEEDED", 413),
        (["source", "source"], "RESEARCH_SOURCE_DUPLICATE", 409),
        (["../escape"], "RESEARCH_SOURCE_NOT_FOUND", 404),
    ],
)
def test_invalid_source_selections_are_rejected_before_backend_read(tasks, code, status):
    def forbidden(_task):
        pytest.fail("Invalid source selection must fail before archive reads")

    assert_error(
        code,
        status,
        service.prepare_lj_research_sources,
        SimpleNamespace(get_task_archive=forbidden),
        tasks,
    )


def test_input_bytes_and_pin_json_caps_apply_collectively_at_the_exact_boundary(
    tmp_path, monkeypatch
):
    store, _, _ = archived_source(tmp_path, task_id="first")
    archived_source(tmp_path, store=store, task_id="second", seed=42)
    pins = service.prepare_lj_research_sources(store, ["first", "second"])
    byte_count = sum(item["bytes"] for pin in pins for item in pin["evidence"]["artifacts"])
    json_count = len(canonical_json(pins))
    monkeypatch.setattr(service, "MAX_RESEARCH_SOURCE_BYTES", byte_count)
    monkeypatch.setattr(service, "MAX_RESEARCH_PIN_BYTES", json_count)
    assert service.prepare_lj_research_sources(store, ["first", "second"]) == pins
    monkeypatch.setattr(service, "MAX_RESEARCH_SOURCE_BYTES", byte_count - 1)
    assert_error(
        "RESEARCH_INPUT_LIMIT_EXCEEDED",
        413,
        service.prepare_lj_research_sources,
        store,
        ["first", "second"],
    )
    monkeypatch.setattr(service, "MAX_RESEARCH_SOURCE_BYTES", byte_count)
    monkeypatch.setattr(service, "MAX_RESEARCH_PIN_BYTES", json_count - 1)
    assert_error(
        "RESEARCH_INPUT_LIMIT_EXCEEDED",
        413,
        service.prepare_lj_research_sources,
        store,
        ["first", "second"],
    )


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "attempt_id",
        "model_version",
        "manifest_sha256",
        "model_contract_sha256",
        "evidence_sha256",
        "parameters_sha256",
    ],
)
def test_pinned_source_cannot_claim_a_different_identity_or_anchor(tmp_path, field):
    store, _, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    pin["source"][field] = (
        "other" if field in {"task_id", "attempt_id", "model_version"} else "0" * 64
    )
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


def test_coordinated_pinned_parameter_change_still_fails_against_manifest(tmp_path):
    store, _, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    pin["parameters"]["seed"] = 42
    pin["source"]["parameters_sha256"] = checkpoint_parameters_sha256(
        parameters_from_payload(pin["parameters"])
    )
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


@pytest.mark.parametrize("options", [{"rerun": 0}, {"rerun": True}, {"rerun": False, "extra": 1}])
def test_research_source_pin_cannot_change_or_coerce_verification_options(tmp_path, options):
    store, _, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    pin["options"] = options
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


def test_consistently_replaced_current_files_cannot_replace_committed_archive_anchor(tmp_path):
    store, result, archive = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    path = tmp_path / result["artifacts"]["observables"]
    modified = json.loads(path.read_bytes())
    modified["production"]["method"] = "coherently replaced report"
    replace_observables(tmp_path, result, canonical_json(modified))
    # The replacement has a matching fresh envelope, but not the original commit anchor.
    read_result_envelope(result, artifacts_dir=tmp_path, require_present=True)
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)
    assert store.get_task_archive("source") == archive


def test_observables_changed_between_envelope_check_and_retained_read_are_rejected(
    tmp_path, monkeypatch
):
    store, result, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    snapshot = service._snapshot

    def mutate(root, path, maximum, *, retain=False):
        if path == result["artifacts"]["observables"]:
            (root / path).write_bytes(b"changed after envelope verification")
        return snapshot(root, path, maximum, retain=retain)

    monkeypatch.setattr(service, "_snapshot", mutate)
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


def test_scientific_values_are_decoded_from_the_same_retained_bytes_that_were_hashed(
    tmp_path, monkeypatch
):
    store, result, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    original = json.loads((tmp_path / result["artifacts"]["observables"]).read_bytes())
    snapshot = service._snapshot

    def mutate_after_capture(root, path, maximum, *, retain=False):
        captured = snapshot(root, path, maximum, retain=retain)
        if path == result["artifacts"]["observables"]:
            (root / path).write_bytes(b"subsequent file replacement must not be decoded")
        return captured

    monkeypatch.setattr(service, "_snapshot", mutate_after_capture)
    loaded = load(tmp_path, pin)
    assert loaded.observables == original


def mutate_json(raw, mutation):
    data = json.loads(raw)
    mutation(data)
    return json.dumps(data).encode()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(model_version="wrong-model"),
        lambda data: data.update(sample_size=999),
        lambda data: data.update(sample_steps=[False, 2, 4, 6]),
        lambda data: data.update(sample_steps=list(range(1002))),
        lambda data: data.update(times=[0, 1, 2, 3]),
        lambda data: data["series"].update(total_energy=[]),
        lambda data: data["series"].update(extra=[0, 1, 2, 3]),
        lambda data: data["series"]["pressure"].__setitem__(0, True),
        lambda data: data["production"].update(sample_count=99),
    ],
)
def test_byte_consistent_but_invalid_observable_schema_is_rejected(tmp_path, mutation):
    store, _, _ = archived_source(tmp_path, mutate=lambda raw: mutate_json(raw, mutation))
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


@pytest.mark.parametrize("invalid", ["duplicate", "nan", "overflow"])
def test_strict_saved_json_rejects_duplicate_keys_and_nonfinite_literals(tmp_path, invalid):
    def mutate(raw):
        if invalid == "duplicate":
            return b'{"duplicate":0,"duplicate":1,' + raw.lstrip()[1:]
        data = json.loads(raw)
        data["series"]["pressure"][0] = float("nan") if invalid == "nan" else "OVERFLOW"
        return json.dumps(data).encode().replace(b'"OVERFLOW"', b"1e309")

    store, _, _ = archived_source(tmp_path, mutate=mutate)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


def test_symlinked_observables_are_rejected_even_with_matching_target_bytes(tmp_path):
    store, result, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    path = tmp_path / result["artifacts"]["observables"]
    target = tmp_path / "outside-observables.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    assert_error("RESEARCH_SOURCE_INTEGRITY_FAILED", 409, load, tmp_path, pin)


@pytest.mark.parametrize("at_call", [1, 2, 3, 4])
def test_cancellation_propagates_without_becoming_a_source_integrity_failure(tmp_path, at_call):
    store, _, _ = archived_source(tmp_path)
    pin = service.prepare_lj_research_sources(store, ["source"])[0]
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if calls == at_call:
            raise SimulationCancelled("cancelled by verifier supervisor")

    with pytest.raises(SimulationCancelled):
        load(tmp_path, pin, check_cancelled=cancelled)
