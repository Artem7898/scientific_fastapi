"""Immutable checkpoint bytes, bounded decoding and numerical compatibility."""

import copy
import hashlib
import io
import json
import os
import struct
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scientific_api.domain.checkpoint import (
    MAX_CHECKPOINT_METADATA_BYTES,
    MAX_CHECKPOINT_TOTAL_BYTES,
    MAX_CHECKPOINTS_PER_TASK,
)
from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationParameters
from scientific_api.models.checkpoint_state import restore_simulator
from scientific_api.models.computation import create_simulator
from scientific_api.models.langevin_engine import create_langevin_simulator
from scientific_api.models.lennard_jones_engine import create_lj_simulator
from scientific_api.research.lj_analysis import SERIES_NAMES
from scientific_api.schemas.checkpoint import CheckpointDetails
from scientific_api.services import checkpoint_service as storage


def parameters(model="ou"):
    if model == "langevin":
        return LangevinParameters(0.7, 1.1, 8, num_steps=4, sample_stride=2, seed=2**64 - 1)
    if model == "lj":
        return LennardJonesParameters(num_particles=16, num_steps=4, sample_stride=2, seed=42)
    return SimulationParameters(1.0, 8, num_steps=4, seed=2**64 - 1)


def make_simulator(configuration):
    if isinstance(configuration, LennardJonesParameters):
        return create_lj_simulator(configuration)
    if isinstance(configuration, LangevinParameters):
        return create_langevin_simulator(configuration)
    return create_simulator(configuration)


def checkpoint(root, model="ou", step=0, *, register=None):
    configuration = parameters(model)
    simulator = make_simulator(configuration)
    controller = storage.CheckpointControl(
        configuration,
        task_id="task",
        attempt_id="attempt",
        artifacts_dir=root,
        interval_steps=1,
        register=register,
    )
    positions = []
    series = {name: [] for name in SERIES_NAMES}
    if model != "ou":
        positions.append(simulator.unwrapped_positions.copy())
        if model == "lj":
            observed = simulator.observables()
            for name in series:
                series[name].append(observed[name])
    for number in range(1, step + 1):
        simulator.advance()
        if model != "ou" and number in configuration.sampling_steps():
            positions.append(simulator.unwrapped_positions.copy())
            if model == "lj":
                observed = simulator.observables()
                for name in series:
                    series[name].append(observed[name])
    samples = {} if model == "ou" else {"trajectory": np.asarray(positions)}
    if model == "lj":
        samples.update(
            {name: np.asarray(values, dtype=np.float64) for name, values in series.items()}
        )
    descriptor = controller.save_if_due(
        step, simulator, samples=samples, next_sample=len(positions)
    )
    assert descriptor is not None
    return configuration, simulator, controller, descriptor, samples


def document(root, descriptor):
    return json.loads((root / descriptor["manifest"]["path"]).read_text())


def rewrite_document(root, descriptor, data, *, allow_nan=False):
    descriptor = copy.deepcopy(descriptor)
    encoded = (
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan).encode()
        + b"\n"
    )
    (root / descriptor["manifest"]["path"]).write_bytes(encoded)
    descriptor["manifest"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    descriptor["manifest"]["bytes"] = len(encoded)
    return descriptor


def rewrite_state(root, descriptor, state):
    descriptor = copy.deepcopy(descriptor)
    descriptor["state"]["sha256"] = hashlib.sha256(state).hexdigest()
    descriptor["state"]["bytes"] = len(state)
    (root / descriptor["state"]["path"]).write_bytes(state)
    data = document(root, descriptor)
    data.update(state_sha256=descriptor["state"]["sha256"], state_bytes=len(state))
    return rewrite_document(root, descriptor, data)


def npy(values):
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=True)
    return stream.getvalue()


@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
@pytest.mark.parametrize("step", [0, 1, 4])
def test_checkpoint_roundtrip_copies_only_completed_sample_prefix(tmp_path, model, step):
    configuration, simulator, controller, descriptor, samples = checkpoint(tmp_path, model, step)
    loaded = storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    assert loaded.completed_steps == step
    assert loaded.next_sample == (len(samples["trajectory"]) if samples else 0)
    assert loaded.policy == controller.policy
    restored = restore_simulator(configuration, loaded.model_metadata, loaded.model_arrays)
    np.testing.assert_array_equal(restored.velocities, simulator.velocities)
    assert restored.rng.bit_generator.state == simulator.rng.bit_generator.state
    for name, values in samples.items():
        np.testing.assert_array_equal(loaded.samples[name], values)
        assert not loaded.samples[name].flags.writeable
    assert all(not values.flags.writeable for values in loaded.model_arrays.values())
    before = loaded.model_arrays["velocities"].copy()
    simulator.velocities[:] = 0
    np.testing.assert_array_equal(loaded.model_arrays["velocities"], before)
    resumed = storage.CheckpointControl(
        configuration,
        task_id="task",
        attempt_id="new-attempt",
        artifacts_dir=tmp_path,
        interval_steps=1,
        loaded=loaded,
    )
    metadata = resumed.manifest_metadata()
    assert metadata["mode"] == "resumed_run" and metadata["parent_attempt_id"] == "attempt"
    assert metadata["completed_steps_at_start"] == step
    assert (
        resumed.save_if_due(step, restored, samples=samples, next_sample=loaded.next_sample) is None
    )
    assert not list(tmp_path.rglob(".checkpoint-*"))


@pytest.mark.parametrize("name", ["manifest", "state"])
def test_download_returns_exact_original_hashed_bytes(tmp_path, name):
    _, _, _, descriptor, _ = checkpoint(tmp_path)
    content, filename, _ = storage.read_checkpoint_artifact(
        descriptor, artifacts_dir=tmp_path, name=name
    )
    assert filename == ("checkpoint.json" if name == "manifest" else "state.npz")
    assert content == (tmp_path / descriptor[name]["path"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == descriptor[name]["sha256"]


@pytest.mark.parametrize("field", ["seed", "time_step", "num_steps", "backend"])
def test_parameter_changes_are_rejected_before_array_loading(tmp_path, monkeypatch, field):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    changed = {"seed": 1, "time_step": 0.002, "num_steps": 5, "backend": "numba"}[field]
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Incompatible request decoded arrays")
    )
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_PARAMETERS_MISMATCH"):
        storage.load_checkpoint(
            descriptor,
            parameters=replace(configuration, **{field: changed}),
            artifacts_dir=tmp_path,
        )


@pytest.mark.parametrize("change", ["cpu", "python", "arithmetic_policy", "model_source"])
def test_numerical_fingerprint_conflicts_but_saved_details_remain_inspectable(
    tmp_path, monkeypatch, change
):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    if change == "cpu":
        native = np._core._multiarray_umath
        monkeypatch.setattr(
            native, "__cpu_features__", {**native.__cpu_features__, "snapshot_test_feature": True}
        )
    elif change == "python":
        monkeypatch.setattr(storage.platform, "python_version", lambda: "99.0.0")
    elif change == "arithmetic_policy":
        monkeypatch.setattr(
            storage.np,
            "geterr",
            lambda: {"divide": "raise", "over": "raise", "under": "raise", "invalid": "raise"},
        )
    else:
        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda path: (
                original(path) + b"changed" if path.name == "physics_engine.py" else original(path)
            ),
        )
    with pytest.raises(storage.CheckpointError, match="REPRODUCIBILITY_CONFLICT"):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    saved = storage.inspect_checkpoint(descriptor, artifacts_dir=tmp_path)
    details = CheckpointDetails(descriptor=descriptor, metadata=saved)
    assert details.integrity_status == "verified" and details.fingerprint_status == "not_checked"


def test_unrelated_api_git_and_document_identity_do_not_change_numeric_fingerprint(
    tmp_path, monkeypatch
):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            b"a new API and README"
            if path.name in {"simulations.py", "README.md", "HEAD"}
            else original(path)
        ),
    )
    loaded = storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    assert loaded.completed_steps == 0


@pytest.mark.parametrize("target", ["manifest", "state"])
@pytest.mark.parametrize("fault", ["bytes", "sha256", "path", "missing", "symlink", "fifo"])
def test_registered_descriptor_and_file_mismatches_fail_closed(tmp_path, target, fault):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    descriptor = copy.deepcopy(descriptor)
    path = tmp_path / descriptor[target]["path"]
    if fault == "bytes":
        descriptor[target]["bytes"] += 1
    elif fault == "sha256":
        descriptor[target]["sha256"] = "0" * 64
    elif fault == "path":
        descriptor[target]["path"] = "../outside"
    elif fault == "missing":
        path.unlink()
    elif fault == "symlink":
        outside = tmp_path / "untrusted"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(storage.CheckpointError):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "object",
        "fortran",
        "huge_shape",
        "nested",
        "duplicate",
        "truncated",
        "unsupported_version",
        "compression",
        "encrypted",
    ],
)
def test_npz_headers_and_members_are_validated_before_numpy_allocation(
    tmp_path, monkeypatch, fault
):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    with zipfile.ZipFile(tmp_path / descriptor["state"]["path"]) as original:
        members = [(item.filename, original.read(item.filename)) for item in original.infolist()]
    name, content = members[0]
    if fault == "object":
        content = npy(np.zeros((8, 2), dtype=object))
    elif fault == "fortran":
        content = npy(np.asfortranarray(np.zeros((8, 2))))
    elif fault == "huge_shape":
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {"descr": "<f8", "fortran_order": False, "shape": (2**50, 2)}
        )
        content = header.getvalue()
    elif fault == "nested":
        name = "../" + name
    elif fault == "truncated":
        content = b"\x93NUM"
    elif fault == "unsupported_version":
        content = np.lib.format.magic(3, 0)
    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_BZIP2 if fault == "compression" else zipfile.ZIP_DEFLATED,
    ) as output:
        output.writestr(name, content)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                output.writestr(name, content)
    raw = bytearray(archive.getvalue())
    if fault == "encrypted":
        struct.pack_into("<H", raw, 6, struct.unpack_from("<H", raw, 6)[0] | 1)
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, central + 8, struct.unpack_from("<H", raw, central + 8)[0] | 1)
    descriptor = rewrite_state(tmp_path, descriptor, bytes(raw))
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Malformed header reached array allocator")
    )
    with pytest.raises(storage.CheckpointError):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


def test_nonfinite_json_rng_is_rejected_before_array_allocation(tmp_path, monkeypatch):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    data = document(tmp_path, descriptor)
    data["model_metadata"]["rng_state"]["state"]["state"] = float("nan")
    descriptor = rewrite_document(tmp_path, descriptor, data, allow_nan=True)
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Nonfinite RNG metadata allocated state")
    )
    with pytest.raises(storage.CheckpointError):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


@pytest.mark.parametrize("version", [1, 2])
def test_advertised_oversized_npy_header_is_never_read_or_allocated(tmp_path, monkeypatch, version):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    payload = np.lib.format.magic(version, 0)
    payload += (65535 if version == 1 else 2**31).to_bytes(2 if version == 1 else 4, "little")
    payload += b"\x00" * 128
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model_velocities.npy", payload)
    descriptor = rewrite_state(tmp_path, descriptor, packed.getvalue())
    original_read = zipfile.ZipExtFile.read
    reads = []

    def bounded_header_read(stream, size=-1):
        reads.append(size)
        assert 0 <= size <= 4096, "Decoder tried to read an unbounded advertised header"
        return original_read(stream, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded_header_read)
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Oversized header reached allocation")
    )
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_STATE_INVALID"):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    assert reads == [8, 2 if version == 1 else 4]


def test_invalid_deflate_is_wrapped_as_checkpoint_failure(tmp_path, monkeypatch):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    raw = bytearray((tmp_path / descriptor["state"]["path"]).read_bytes())
    filename_length, extra_length = struct.unpack_from("<HH", raw, 26)
    data_start = 30 + filename_length + extra_length
    # Deflate BTYPE=3 is reserved. The ZIP directory remains otherwise valid.
    raw[data_start] = (raw[data_start] & ~7) | 7
    descriptor = rewrite_state(tmp_path, descriptor, bytes(raw))
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Invalid deflate reached allocation")
    )
    with pytest.raises(storage.CheckpointError):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


@pytest.mark.parametrize("fault", ["rng", "model", "lj_steps", "lj_budget"])
def test_compatible_load_validates_numerical_state_before_accepting_continuation(tmp_path, fault):
    model = "lj" if fault.startswith("lj") else "ou"
    configuration, _, _, descriptor, _ = checkpoint(tmp_path, model)
    data = document(tmp_path, descriptor)
    if fault == "rng":
        data["model_metadata"]["rng_state"]["has_uint32"] = True
    elif fault == "model":
        data["model_metadata"]["model_version"] = "unknown-model-v1"
    elif fault == "lj_steps":
        data["model_metadata"]["engine_state"]["steps"] = 1
    else:
        data["model_metadata"]["engine_state"]["analysis_pair_evaluations"] = 100_000_000
    descriptor = rewrite_document(tmp_path, descriptor, data)
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_STATE_INVALID"):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_decoded_nonfinite_state_is_not_accepted(tmp_path, value):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    array = np.zeros((8, 2))
    array[0, 0] = value
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model_velocities.npy", npy(array))
    descriptor = rewrite_state(tmp_path, descriptor, packed.getvalue())
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_STATE_INVALID"):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


def test_oversized_metadata_is_rejected_before_file_decode(tmp_path, monkeypatch):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    path = tmp_path / descriptor["manifest"]["path"]
    path.write_bytes(b" " * (MAX_CHECKPOINT_METADATA_BYTES + 1))
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("Oversized metadata allocated arrays")
    )
    with pytest.raises(storage.CheckpointError):
        storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)


def test_all_directory_entries_are_synced_before_registration_acknowledgement(
    tmp_path, monkeypatch
):
    events = []
    original = storage._fsync_directory

    def synced(path):
        original(path)
        events.append(("directory", Path(path)))

    def register(descriptor):
        parent = (tmp_path / descriptor["manifest"]["path"]).parent.parent
        assert events[-1] == ("directory", parent)
        assert any(path.name.startswith(".checkpoint-") for kind, path in events)
        assert (
            storage.inspect_checkpoint(descriptor, artifacts_dir=tmp_path)["completed_steps"] == 0
        )
        events.append(("registered", descriptor["checkpoint_id"]))

    monkeypatch.setattr(storage, "_fsync_directory", synced)
    checkpoint(tmp_path, register=register)
    assert events[0] == ("directory", tmp_path.parent)
    assert events[-1][0] == "registered"


@pytest.mark.parametrize("phase", ["temporary", "after_rename"])
def test_directory_sync_failure_does_not_register_a_checkpoint(tmp_path, monkeypatch, phase):
    registrations = []
    original = storage._fsync_directory

    def failed_sync(path):
        if (phase == "temporary" and path.name.startswith(".checkpoint-")) or (
            phase == "after_rename" and path.name == "checkpoints"
        ):
            raise OSError("Directory durability could not be established")
        return original(path)

    monkeypatch.setattr(storage, "_fsync_directory", failed_sync)
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_PERSISTENCE_FAILED"):
        checkpoint(tmp_path, register=registrations.append)
    assert registrations == []


def test_new_nested_artifact_root_links_are_synced_before_registration(tmp_path, monkeypatch):
    root = tmp_path / "new-artifacts" / "research" / "runs"
    root.mkdir(parents=True)
    synchronized = []
    original = storage._fsync_directory

    def synced(path):
        original(path)
        synchronized.append(Path(path))

    def register(descriptor):
        assert root.parent in synchronized
        assert root.parent.parent in synchronized
        assert tmp_path in synchronized

    monkeypatch.setattr(storage, "_fsync_directory", synced)
    checkpoint(root, register=register)
    assert not list(tmp_path.rglob(".checkpoint-*"))


def test_hash_and_array_decode_use_same_snapshot_bytes(tmp_path, monkeypatch):
    configuration, simulator, _, descriptor, _ = checkpoint(tmp_path)
    original = storage.np.load

    def replace_after_hash(*args, **kwargs):
        (tmp_path / descriptor["state"]["path"]).write_bytes(b"changed after the trusted read")
        return original(*args, **kwargs)

    monkeypatch.setattr(storage.np, "load", replace_after_hash)
    loaded = storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    np.testing.assert_array_equal(loaded.model_arrays["velocities"], simulator.velocities)


@pytest.mark.parametrize("case", ["task", "attempt", "parameters", "root"])
def test_control_rejects_mismatched_run_context(tmp_path, case):
    configuration, _, control, _, _ = checkpoint(tmp_path)
    arguments = {"task_id": "task", "attempt_id": "attempt", "artifacts_dir": tmp_path}
    if case == "parameters":
        configuration = replace(configuration, seed=42)
    else:
        key = "artifacts_dir" if case == "root" else case + "_id"
        arguments[key] = tmp_path / "other" if case == "root" else "other"
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_CONTEXT_MISMATCH"):
        control.validate_run(configuration, **arguments)


def test_control_rejects_unsampled_full_tail_and_publishes_no_partial_checkpoint(tmp_path):
    configuration = parameters("langevin")
    control = storage.CheckpointControl(
        configuration,
        task_id="task",
        attempt_id="attempt",
        artifacts_dir=tmp_path,
        interval_steps=1,
    )
    simulator = make_simulator(configuration)
    entire_unfilled = np.zeros(
        (len(configuration.sampling_steps()), configuration.num_particles, 2)
    )
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_STATE_INVALID"):
        control.save_if_due(0, simulator, samples={"trajectory": entire_unfilled}, next_sample=1)
    assert not list(tmp_path.rglob("checkpoint.json"))


def test_acknowledgement_loss_preserves_immutable_published_checkpoint(tmp_path):
    admitted = []

    def lost_ack(descriptor):
        admitted.append(copy.deepcopy(descriptor))
        raise TimeoutError("The remote archive may already contain this descriptor")

    configuration = parameters()
    controller = storage.CheckpointControl(
        configuration,
        task_id="task",
        attempt_id="attempt",
        artifacts_dir=tmp_path,
        interval_steps=1,
        register=lost_ack,
    )
    with pytest.raises(TimeoutError):
        controller.save_if_due(0, make_simulator(configuration), samples={}, next_sample=0)
    assert len(admitted) == 1 and controller.last_descriptor == admitted[0]
    loaded = storage.load_checkpoint(admitted[0], parameters=configuration, artifacts_dir=tmp_path)
    assert loaded.completed_steps == 0
    assert not list(tmp_path.rglob(".checkpoint-*"))


def test_partial_directories_are_not_scanned_or_selected(tmp_path):
    configuration, _, _, descriptor, _ = checkpoint(tmp_path)
    partial = tmp_path / "task" / "attempt" / "checkpoints" / ".checkpoint-incomplete"
    partial.mkdir()
    (partial / "state.npz").write_bytes(b"partial")
    loaded = storage.load_checkpoint(descriptor, parameters=configuration, artifacts_dir=tmp_path)
    assert loaded.descriptor["checkpoint_id"] == descriptor["checkpoint_id"]


@pytest.mark.parametrize("interval", [-1, True, 1.5, 10_001])
def test_invalid_checkpoint_policy_is_rejected(tmp_path, interval):
    with pytest.raises(storage.CheckpointError, match="CHECKPOINT_POLICY_INVALID"):
        storage.CheckpointControl(
            parameters(),
            task_id="task",
            attempt_id="attempt",
            artifacts_dir=tmp_path,
            interval_steps=interval,
        )


def test_adaptive_schedule_bounds_count_and_conservative_total_storage(tmp_path):
    configuration = SimulationParameters(1.0, 1_000_000, num_steps=100)
    control = storage.CheckpointControl(
        configuration,
        task_id="task",
        attempt_id="attempt",
        artifacts_dir=tmp_path,
        interval_steps=1,
    )
    schedule = control.policy["schedule"]
    assert schedule[0] == 0 and schedule[-1] == configuration.num_steps
    assert len(schedule) <= MAX_CHECKPOINTS_PER_TASK
    assert control.policy["effective_interval_steps"] > 1
    worst_case_raw = configuration.num_particles * 2 * 8
    assert (
        len(schedule) * (worst_case_raw * 1.01 + MAX_CHECKPOINT_METADATA_BYTES)
        <= MAX_CHECKPOINT_TOTAL_BYTES
    )
    disabled = storage.CheckpointControl(
        configuration, task_id="task", attempt_id="attempt", artifacts_dir=tmp_path
    )
    assert disabled.policy["schedule"] == []
