"""Bounded reruns compare pinned scientific content, not packaging or identity."""

import io
import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled, SimulationParameters
from scientific_api.models.computation import BackendUnavailable
from scientific_api.schemas.verification import VerificationCheck
from scientific_api.services import verification_rerun as service
from scientific_api.services.result_envelope_service import read_result_envelope
from scientific_api.services.simulation_service import run_simulation


def parameters(model, backend="numpy"):
    common = {"num_steps": 12, "seed": 2**64 - 1, "backend": backend}
    if model == "ou":
        return SimulationParameters(1.2, 17, time_step=0.1, **common)
    if model == "langevin":
        return LangevinParameters(
            2.0,
            1.0,
            17,
            time_step=0.2,
            sample_stride=3,
            acceleration=(0.3, -0.2),
            boundary="periodic",
            box_size=(0.3, 0.4),
            **common,
        )
    _, ensemble, interaction = model.split("-")
    return LennardJonesParameters(
        num_particles=16,
        box_size=(6.0, 6.0),
        initial_temperature=0.1,
        ensemble=ensemble,
        bath_temperature=0.15 if ensemble == "nvt" else None,
        friction=0.7 if ensemble == "nvt" else None,
        interaction_backend=interaction,
        time_step=0.002,
        sample_stride=3,
        **common,
    )


def source(tmp_path, model="ou", backend="numpy"):
    configuration = parameters(model, backend)
    root = tmp_path / "source"
    result = run_simulation(
        configuration, task_id="source", attempt_id="original", artifacts_dir=root
    )
    envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
    artifacts = {item.name: (root / item.path).read_bytes() for item in envelope.artifacts}
    return configuration, envelope, artifacts, root


def execute(tmp_path, configuration, envelope, artifacts, **kwargs):
    return service.verify_same_seed_rerun(
        configuration, envelope, artifacts, temporary_root=tmp_path / "verifier", **kwargs
    )


def indexed(checks):
    assert len({item["name"] for item in checks}) == len(checks)
    for item in checks:
        VerificationCheck.model_validate(item)
        assert item["name"].startswith("rerun_")
    return {item["name"]: item for item in checks}


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    "model", ["ou", "langevin", "lj-nve-direct", "lj-nve-verlet", "lj-nvt-direct", "lj-nvt-verlet"]
)
def test_same_seed_observation_matches_science_in_all_supported_cpu_modes(tmp_path, model, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    configuration, envelope, artifacts, root = source(tmp_path, model, backend)
    original = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    envelope_before = envelope.model_dump(mode="json")
    artifacts_before = dict(artifacts)
    checks = execute(tmp_path, configuration, envelope, artifacts)
    assert all(item["status"] == "passed" for item in checks), checks
    values = indexed(checks)
    assert (
        values["rerun_scientific_arrays"]["expected"]
        == values["rerun_scientific_arrays"]["observed"]
    )
    assert (
        values["rerun_scientific_scalars"]["expected"]
        == values["rerun_scientific_scalars"]["observed"]
    )
    assert values["rerun_environment"]["details"]["repeatability_guaranteed"] is False
    assert envelope.model_dump(mode="json") == envelope_before
    assert artifacts == artifacts_before
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == original
    assert list((tmp_path / "verifier").iterdir()) == []


def test_rerun_uses_only_supplied_snapshots_even_when_original_files_disappear(tmp_path):
    configuration, envelope, artifacts, root = source(tmp_path, "langevin")
    shutil.rmtree(root)
    temporary_root = tmp_path / "verifier"
    temporary_root.mkdir()
    sentinel = temporary_root / "unrelated"
    sentinel.write_bytes(b"preserve")
    checks = execute(tmp_path, configuration, envelope, artifacts)
    assert all(item["status"] == "passed" for item in checks), checks
    assert not root.exists()
    assert list(temporary_root.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize(
    "model,limit", [("ou", "max_particle_steps"), ("lj-nvt-verlet", "max_lj_pair_checks")]
)
def test_budget_refusal_never_starts_physics(tmp_path, monkeypatch, model, limit):
    configuration, envelope, artifacts, _ = source(tmp_path, model)

    def forbidden(*args, **kwargs):
        pytest.fail("A refused verification must not execute dynamics")

    monkeypatch.setattr(service, "run_simulation", forbidden)
    checks = execute(tmp_path, configuration, envelope, artifacts, **{limit: 1})
    assert len(checks) == 1
    assert checks[0]["status"] == "unsupported"
    assert checks[0]["reason"] == "rerun_workload_exceeds_limit"


@pytest.mark.parametrize("changed", ["numpy", "backend", "numerical_fingerprint"])
def test_known_environment_mismatch_is_inconclusive_without_silent_fallback(
    tmp_path, monkeypatch, changed
):
    configuration, envelope, artifacts, _ = source(tmp_path)
    if changed == "numpy":
        envelope = envelope.model_copy(
            update={"environment": {**envelope.environment, "numpy": "0.0"}}
        )
    elif changed == "backend":
        envelope = envelope.model_copy(
            update={"execution": {**envelope.execution, "backend": "numba"}}
        )
    else:
        envelope = envelope.model_copy(update={"continuation": {"fingerprint_sha256": "0" * 64}})

    def forbidden(*args, **kwargs):
        pytest.fail("A known incompatible policy must not run a substitute backend")

    monkeypatch.setattr(service, "run_simulation", forbidden)
    values = indexed(execute(tmp_path, configuration, envelope, artifacts))
    assert values["rerun_environment"]["status"] == "inconclusive"
    assert values["rerun_environment"]["details"]["mismatched_fields"]


def test_unknown_cpu_and_additive_package_changes_do_not_fabricate_a_guarantee(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path)
    historical = {
        name: value
        for name, value in envelope.environment.items()
        if name not in {"processor", "machine"}
    }
    historical["source_sha256"] = "0" * 64
    historical["source_commit"] = "historical-checkout"
    envelope = envelope.model_copy(update={"environment": historical})
    checks = execute(tmp_path, configuration, envelope, artifacts)
    assert all(item["status"] == "passed" for item in checks), checks
    details = indexed(checks)["rerun_environment"]["details"]
    assert details["cpu_identity_recorded"] is False
    assert details["repeatability_guaranteed"] is False
    assert details["recorded_numerical_fingerprint"] is None


def test_different_seed_is_not_accepted_as_the_same_realization(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path)
    checks = execute(tmp_path, replace(configuration, seed=1), envelope, artifacts)
    assert checks[0]["name"] == "rerun_source_contract"
    assert checks[0]["status"] == "failed"


def _replace_array(raw, name, update):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    update(arrays[name])
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def test_array_comparison_detects_a_different_snapshot_without_mutating_it(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path)
    altered = dict(artifacts)
    altered["final_state"] = _replace_array(
        artifacts["final_state"],
        "velocities",
        lambda array: array.__setitem__((0, 0), array[0, 0] + 0.125),
    )
    values = indexed(execute(tmp_path, configuration, envelope, altered))
    assert values["rerun_scientific_arrays"]["status"] == "failed"
    assert values["rerun_scientific_scalars"]["status"] == "passed"
    assert artifacts["final_state"] != altered["final_state"]


def test_exact_array_scope_preserves_signed_zero(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path, "langevin")
    altered = dict(artifacts)
    altered["trajectory"] = _replace_array(
        artifacts["trajectory"], "times", lambda array: array.__setitem__(0, -0.0)
    )
    values = indexed(execute(tmp_path, configuration, envelope, altered))
    assert values["rerun_scientific_arrays"]["status"] == "failed"
    assert (
        values["rerun_scientific_arrays"]["expected"]["trajectory/times"]
        != values["rerun_scientific_arrays"]["observed"]["trajectory/times"]
    )


def test_scalar_comparison_can_fail_even_when_scientific_arrays_match(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path)
    candidate = envelope.result.model_copy(
        update={"final_temperature": envelope.result.final_temperature + 1}
    )
    envelope = envelope.model_copy(update={"result": candidate})
    values = indexed(execute(tmp_path, configuration, envelope, artifacts))
    assert values["rerun_scientific_arrays"]["status"] == "passed"
    assert values["rerun_scientific_scalars"]["status"] == "failed"


@pytest.mark.parametrize("raw", [b"{}", b'{"value":1e309}', b'{"duplicate":1,"duplicate":2}'])
def test_observable_snapshot_disagreement_is_reported(tmp_path, raw):
    configuration, envelope, artifacts, _ = source(tmp_path, "langevin")
    altered = {**artifacts, "transport": raw}
    values = indexed(execute(tmp_path, configuration, envelope, altered))
    assert values["rerun_transport"]["status"] == "failed"


def test_unavailable_optional_backend_is_not_silently_replaced(tmp_path, monkeypatch):
    configuration, envelope, artifacts, _ = source(tmp_path)

    def unavailable(backend):
        raise BackendUnavailable("/private/credentials must not escape")

    monkeypatch.setattr(service, "require_backend", unavailable)
    checks = execute(tmp_path, configuration, envelope, artifacts)
    assert checks[-1]["status"] == "unsupported"
    assert "private" not in json.dumps(checks)


def test_cancellation_propagates_before_execution(tmp_path):
    configuration, envelope, artifacts, _ = source(tmp_path)
    with pytest.raises(SimulationCancelled):
        execute(tmp_path, configuration, envelope, artifacts, cancelled=lambda: True)
    assert not (tmp_path / "verifier").exists()


def test_cancellation_is_checked_within_the_scientific_transition_loop(tmp_path, monkeypatch):
    configuration, envelope, artifacts, _ = source(tmp_path)
    active, calls = False, 0
    original = service.run_simulation

    def wrapped(*args, **kwargs):
        nonlocal active
        active = True
        return original(*args, **kwargs)

    def cancelled():
        nonlocal calls
        if active:
            calls += 1
        return calls >= 5

    monkeypatch.setattr(service, "run_simulation", wrapped)
    with pytest.raises(SimulationCancelled):
        execute(tmp_path, configuration, envelope, artifacts, cancelled=cancelled)
    assert calls >= 5
    assert list((tmp_path / "verifier").iterdir()) == []


def test_rerun_execution_error_does_not_expose_paths_or_overwrite_science(tmp_path, monkeypatch):
    configuration, envelope, artifacts, root = source(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def unavailable(*args, **kwargs):
        raise RuntimeError("/private/server/file: password")

    monkeypatch.setattr(service, "run_simulation", unavailable)
    checks = execute(tmp_path, configuration, envelope, artifacts)
    assert checks[-1]["status"] == "inconclusive"
    assert "password" not in json.dumps(checks) and "/private" not in json.dumps(checks)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert list((tmp_path / "verifier").iterdir()) == []
