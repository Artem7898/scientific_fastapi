"""Immutable bounded checkpoints, exact RNG state and strict continuation policy.

Only a descriptor registered by the queue is authoritative for an HTTP task.
Files alone never establish lease ownership, and hashes do not authenticate an
author. This module also supports explicit trusted descriptors in Python tools.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.checkpoint import (
    MAX_CHECKPOINT_ARCHIVE_BYTES,
    MAX_CHECKPOINT_ARRAY_BYTES,
    MAX_CHECKPOINT_METADATA_BYTES,
    MAX_CHECKPOINT_TOTAL_BYTES,
    MAX_CHECKPOINTS_PER_TASK,
    checkpoint_model_version,
    checkpoint_parameters,
    checkpoint_parameters_sha256,
)
from ..domain.experiments import ExperimentParameters
from ..domain.langevin import LangevinParameters
from ..domain.lennard_jones import LennardJonesParameters
from ..models.checkpoint_state import SimulatorStateError, capture_simulator, restore_simulator
from ..research.lj_analysis import SERIES_NAMES
from ..schemas.checkpoint import CheckpointDescriptor
from .result_envelope_service import _json, _root, _snapshot

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ARRAY_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
MAX_CHECKPOINT_ARRAYS = 16


class CheckpointError(ValueError):
    def __init__(self, code: str, status_code: int = 409):
        self.code, self.status_code = code, status_code
        super().__init__(code)


@dataclass(frozen=True)
class LoadedCheckpoint:
    descriptor: dict[str, Any]
    completed_steps: int
    next_sample: int
    model_metadata: dict[str, Any]
    model_arrays: dict[str, np.ndarray]
    samples: dict[str, np.ndarray]
    policy: dict[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_fingerprint(parameters: ExperimentParameters) -> dict[str, Any]:
    """Identify the numerical implementation, without Git/API/document identity."""
    package = Path(__file__).parents[1]
    names = ["models/checkpoint_state.py", "domain/simulation.py"]
    if isinstance(parameters, LennardJonesParameters):
        names += [
            "models/lennard_jones_engine.py",
            "models/lj_forces.py",
            "domain/lennard_jones.py",
            "research/lj_analysis.py",
        ]
        if parameters.backend == "numba":
            names.append("models/lj_numba.py")
    elif isinstance(parameters, LangevinParameters):
        names += [
            "models/langevin_engine.py",
            "domain/langevin.py",
            "research/transport_analysis.py",
        ]
        if parameters.backend == "numba":
            names.append("models/langevin_numba.py")
    else:
        names += ["models/physics_engine.py", "models/computation.py"]
        if parameters.backend == "numba":
            names.append("models/numba_engine.py")
    cpu = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    features = getattr(cpu, "__cpu_features__", None)
    if not isinstance(features, dict):
        raise CheckpointError("CHECKPOINT_FINGERPRINT_UNAVAILABLE", 503)
    backend: dict[str, Any] = {"name": parameters.backend, "numpy": np.__version__}
    if parameters.backend == "numba":
        try:
            numba = importlib.import_module("numba")
            llvmlite = importlib.import_module("llvmlite")
            llvm = importlib.import_module("llvmlite.binding")
        except ImportError as error:
            raise CheckpointError("REPRODUCIBILITY_CONFLICT") from error
        backend.update(
            numba=numba.__version__,
            llvmlite=llvmlite.__version__,
            llvm=list(llvm.llvm_version_info),
            cpu_name=numba.config.CPU_NAME,
            cpu_features=numba.config.CPU_FEATURES,
        )
    return {
        "schema_version": 1,
        "model_state_abi": 1,
        "service_sampling_abi": 1,
        "model_version": checkpoint_model_version(parameters),
        "implementation_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in sorted(names)
        },
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "backend": backend,
        "machine": platform.machine(),
        "system": platform.system(),
        "libc": list(platform.libc_ver()),
        "cpu_features": {name: bool(value) for name, value in sorted(features.items())},
        "byteorder": sys.byteorder,
        "dtype": np.dtype(np.float64).str,
        "float_rounds": sys.float_info.rounds,
        "numpy_error_policy": np.geterr(),
        "rng": "PCG64-full-state-v1",
        "fastmath": False,
        "parallel": False,
    }


def _descriptor(value: dict[str, Any]) -> dict[str, Any]:
    result = CheckpointDescriptor.model_validate(value).model_dump(mode="json")
    prefix = f"{result['task_id']}/{result['attempt_id']}/checkpoints/{result['checkpoint_id']}"
    if (
        result["completed_steps"] > result["total_steps"]
        or result["manifest"]["path"] != f"{prefix}/checkpoint.json"
        or result["state"]["path"] != f"{prefix}/state.npz"
        or result["manifest"]["bytes"] > MAX_CHECKPOINT_METADATA_BYTES
        or result["state"]["bytes"] > MAX_CHECKPOINT_ARCHIVE_BYTES
    ):
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    return result


def _read_bundle(
    descriptor: dict[str, Any], artifacts_dir: Path, *, retain_state: bool
) -> tuple[dict[str, Any], dict[str, Any], bytes | None]:
    selected = _descriptor(descriptor)
    root = _root(artifacts_dir)
    mh, ms, raw = _snapshot(
        root, selected["manifest"]["path"], MAX_CHECKPOINT_METADATA_BYTES, retain=True
    )
    assert raw is not None
    sh, ss, state = _snapshot(
        root, selected["state"]["path"], MAX_CHECKPOINT_ARCHIVE_BYTES, retain=retain_state
    )
    if (mh, ms) != (selected["manifest"]["sha256"], selected["manifest"]["bytes"]) or (sh, ss) != (
        selected["state"]["sha256"],
        selected["state"]["bytes"],
    ):
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    metadata = _json(raw)
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    for name in (
        "checkpoint_id",
        "task_id",
        "attempt_id",
        "model_version",
        "completed_steps",
        "total_steps",
        "parameters_sha256",
        "fingerprint_sha256",
        "created_at",
    ):
        if metadata.get(name) != selected[name]:
            raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    if any(type(metadata.get(name)) is not int for name in ("completed_steps", "total_steps")):
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    if (
        metadata.get("state_sha256") != sh
        or metadata.get("state_bytes") != ss
        or not isinstance(metadata.get("fingerprint"), dict)
        or _digest(metadata["fingerprint"]) != selected["fingerprint_sha256"]
        or not isinstance(metadata.get("model_metadata"), dict)
    ):
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
    return selected, metadata, state


def inspect_checkpoint(descriptor: dict[str, Any], *, artifacts_dir: Path) -> dict[str, Any]:
    """Check saved bytes without claiming compatibility with the current CPU."""
    try:
        return _read_bundle(descriptor, artifacts_dir, retain_state=False)[1]
    except CheckpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED") from error


def _sampling_shapes(
    parameters: ExperimentParameters, step: int, cursor: int
) -> dict[str, tuple[int, ...]]:
    if isinstance(parameters, (LangevinParameters, LennardJonesParameters)):
        expected = sum(saved <= step for saved in parameters.sampling_steps())
        if cursor != expected:
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        shapes: dict[str, tuple[int, ...]] = {"trajectory": (cursor, parameters.num_particles, 2)}
        if isinstance(parameters, LennardJonesParameters):
            shapes.update({name: (cursor,) for name in SERIES_NAMES})
        return shapes
    if cursor != 0:
        raise CheckpointError("CHECKPOINT_STATE_INVALID")
    return {}


def _array_contract(
    parameters: ExperimentParameters, step: int, cursor: int, records: Any
) -> dict[str, tuple[str, str, tuple[int, ...], np.dtype]]:
    if not isinstance(records, dict) or not 1 <= len(records) <= MAX_CHECKPOINT_ARRAYS:
        raise CheckpointError("CHECKPOINT_STATE_INVALID")
    n = parameters.num_particles
    model_shapes: dict[str, tuple[int, ...] | None] = {"velocities": (n, 2)}
    if isinstance(parameters, (LangevinParameters, LennardJonesParameters)):
        model_shapes["unwrapped_positions"] = (n, 2)
    if isinstance(parameters, LennardJonesParameters):
        model_shapes["forces"] = (n, 2)
        if parameters.interaction_backend == "verlet":
            model_shapes.update(neighbor_reference=(n, 2), neighbor_pairs=None)
    expected = {f"model_{name}": ("model", name, shape) for name, shape in model_shapes.items()}
    expected.update(
        {
            f"sample_{name}": ("sample", name, shape)
            for name, shape in _sampling_shapes(parameters, step, cursor).items()
        }
    )
    if set(records) != set(expected):
        raise CheckpointError("CHECKPOINT_STATE_INVALID")
    total = 0
    result = {}
    for key, (role, name, expected_shape) in expected.items():
        entry = records[key]
        if (
            not isinstance(entry, dict)
            or entry.get("role") != role
            or entry.get("name") != name
            or not isinstance(entry.get("shape"), list)
            or any(type(size) is not int or size < 0 for size in entry["shape"])
        ):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        shape = tuple(entry["shape"])
        if name == "neighbor_pairs":
            if len(shape) != 2 or shape[1] != 2 or shape[0] > min(1_000_000, n * (n - 1) // 2):
                raise CheckpointError("CHECKPOINT_STATE_INVALID")
        elif shape != expected_shape:
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        dtype = np.dtype(np.int64 if name == "neighbor_pairs" else np.float64)
        count = math.prod(shape) * dtype.itemsize
        if (
            entry.get("dtype") != dtype.str
            or type(entry.get("nbytes")) is not int
            or entry["nbytes"] != count
        ):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        total += count
        if total > MAX_CHECKPOINT_ARRAY_BYTES:
            raise CheckpointError("CHECKPOINT_LIMIT_EXCEEDED")
        result[key] = role, name, shape, dtype
    return result


def load_checkpoint(
    descriptor: dict[str, Any],
    *,
    parameters: ExperimentParameters,
    artifacts_dir: Path,
    validate_fingerprint: bool = True,
) -> LoadedCheckpoint:
    try:
        selected, metadata, raw = _read_bundle(descriptor, artifacts_dir, retain_state=True)
        if (
            selected["parameters_sha256"] != checkpoint_parameters_sha256(parameters)
            or _canonical(metadata.get("parameters"))
            != _canonical(checkpoint_parameters(parameters))
            or selected["total_steps"] != parameters.num_steps
            or selected["model_version"] != checkpoint_model_version(parameters)
        ):
            raise CheckpointError("CHECKPOINT_PARAMETERS_MISMATCH")
        if validate_fingerprint and _canonical(metadata["fingerprint"]) != _canonical(
            checkpoint_fingerprint(parameters)
        ):
            raise CheckpointError("REPRODUCIBILITY_CONFLICT")
        cursor = metadata.get("next_sample")
        if type(cursor) is not int or cursor < 0:
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        contract = _array_contract(
            parameters, selected["completed_steps"], cursor, metadata.get("arrays")
        )
        assert raw is not None
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if (
                len(members) != len(contract)
                or {member.filename for member in members} != {key + ".npy" for key in contract}
                or any(
                    member.flag_bits & 1
                    or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    for member in members
                )
            ):
                raise CheckpointError("CHECKPOINT_STATE_INVALID")
            for member in members:
                role, name, shape, dtype = contract[member.filename[:-4]]
                expected_bytes = math.prod(shape) * dtype.itemsize
                if member.file_size > expected_bytes + 4096 + 12:
                    raise CheckpointError("CHECKPOINT_STATE_INVALID")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in {(1, 0), (2, 0)}:
                        raise CheckpointError("CHECKPOINT_STATE_INVALID")
                    length_bytes = stream.read(2 if version == (1, 0) else 4)
                    if len(length_bytes) != (2 if version == (1, 0) else 4):
                        raise CheckpointError("CHECKPOINT_STATE_INVALID")
                    header_size = int.from_bytes(length_bytes, "little")
                    if (
                        not 0 < header_size <= 4096
                        or member.file_size != 8 + len(length_bytes) + header_size + expected_bytes
                    ):
                        raise CheckpointError("CHECKPOINT_STATE_INVALID")
                    header = stream.read(header_size)
                    if len(header) != header_size:
                        raise CheckpointError("CHECKPOINT_STATE_INVALID")
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    # NumPy checks max_header_size only after reading the
                    # advertised header; pass it an already bounded buffer.
                    actual_shape, fortran, actual_dtype = reader(
                        io.BytesIO(length_bytes + header), max_header_size=4096
                    )
                    if actual_shape != shape or fortran or actual_dtype != dtype:
                        raise CheckpointError("CHECKPOINT_STATE_INVALID")
        models, samples = {}, {}
        with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096) as archive:
            for key, (role, name, shape, dtype) in contract.items():
                values = archive[key]
                if values.shape != shape or values.dtype != dtype or not np.isfinite(values).all():
                    raise CheckpointError("CHECKPOINT_STATE_INVALID")
                values.setflags(write=False)
                (models if role == "model" else samples)[name] = values
        policy = metadata.get("checkpoint_policy")
        if not isinstance(policy, dict):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        if validate_fingerprint:
            # Validate RNG, cached forces and work counters on a compatible
            # numerical backend before admitting continuation. Restoration
            # copies state only: no initialization draws or force work occur.
            try:
                restored = restore_simulator(parameters, metadata["model_metadata"], models)
                if (
                    isinstance(parameters, LennardJonesParameters)
                    and getattr(restored, "_steps", None) != selected["completed_steps"]
                ):
                    raise CheckpointError("CHECKPOINT_STATE_INVALID")
            except SimulatorStateError as error:
                raise CheckpointError("CHECKPOINT_STATE_INVALID") from error
        return LoadedCheckpoint(
            selected,
            selected["completed_steps"],
            cursor,
            metadata["model_metadata"],
            models,
            samples,
            policy,
        )
    except CheckpointError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        zipfile.BadZipFile,
        zlib.error,
        RuntimeError,
        EOFError,
    ) as error:
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED") from error


def read_checkpoint_artifact(
    descriptor: dict[str, Any], *, artifacts_dir: Path, name: str
) -> tuple[bytes, str, str]:
    if name not in {"manifest", "state"}:
        raise CheckpointError("CHECKPOINT_NOT_FOUND", 404)
    try:
        selected, metadata, state = _read_bundle(descriptor, artifacts_dir, retain_state=True)
        if name == "state":
            assert state is not None
            return state, "state.npz", "application/octet-stream"
        # Read and hash the same returned JSON snapshot; do not serialize it anew.
        digest, size, raw = _snapshot(
            _root(artifacts_dir),
            selected["manifest"]["path"],
            MAX_CHECKPOINT_METADATA_BYTES,
            retain=True,
        )
        if (digest, size) != (selected["manifest"]["sha256"], selected["manifest"]["bytes"]):
            raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
        assert raw is not None
        return raw, "checkpoint.json", "application/json"
    except CheckpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED") from error


def _policy(parameters: ExperimentParameters, interval_steps: int) -> dict[str, Any]:
    if type(interval_steps) is not int or not 0 <= interval_steps <= 10_000:
        raise CheckpointError("CHECKPOINT_POLICY_INVALID")
    n = parameters.num_particles
    estimate = n * 2 * 8
    if isinstance(parameters, (LangevinParameters, LennardJonesParameters)):
        estimate += n * 2 * 8 + len(parameters.sampling_steps()) * n * 2 * 8
    if isinstance(parameters, LennardJonesParameters):
        estimate += n * 2 * 8 + len(parameters.sampling_steps()) * len(SERIES_NAMES) * 8
        if parameters.interaction_backend == "verlet":
            estimate += n * 2 * 8 + min(1_000_000, n * (n - 1) // 2) * 2 * 8
    # Compression can expand slightly. Reserve metadata/header margin as well.
    estimate = math.ceil(estimate * 1.01) + MAX_CHECKPOINT_METADATA_BYTES
    count = max(2, min(MAX_CHECKPOINTS_PER_TASK, MAX_CHECKPOINT_TOTAL_BYTES // estimate))
    effective = (
        max(interval_steps, math.ceil(parameters.num_steps / (count - 1))) if interval_steps else 0
    )
    schedule = (
        [0, *range(effective, parameters.num_steps, effective), parameters.num_steps]
        if effective
        else []
    )
    return {
        "requested_interval_steps": interval_steps,
        "effective_interval_steps": effective,
        "max_checkpoints": MAX_CHECKPOINTS_PER_TASK,
        "max_total_bytes": MAX_CHECKPOINT_TOTAL_BYTES,
        "schedule": schedule,
    }


class CheckpointControl:
    def __init__(
        self,
        parameters: ExperimentParameters,
        *,
        task_id: str,
        attempt_id: str,
        artifacts_dir: Path,
        interval_steps: int = 0,
        loaded: LoadedCheckpoint | None = None,
        register: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not all(
            isinstance(value, str) and _IDENTIFIER.fullmatch(value)
            for value in (task_id, attempt_id)
        ):
            raise CheckpointError("CHECKPOINT_CONTEXT_MISMATCH")
        self.parameters, self.task_id, self.attempt_id = parameters, task_id, attempt_id
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.loaded, self.register = loaded, register
        self.fingerprint = checkpoint_fingerprint(parameters)
        self.parameters_digest = checkpoint_parameters_sha256(parameters)
        self.policy = _policy(parameters, interval_steps)
        self._source = dict(loaded.descriptor) if loaded else None
        self._start = loaded.completed_steps if loaded else 0
        if loaded is not None:
            if (
                loaded.descriptor["task_id"] != task_id
                or loaded.descriptor["parameters_sha256"] != self.parameters_digest
                or loaded.descriptor["fingerprint_sha256"] != _digest(self.fingerprint)
            ):
                raise CheckpointError("REPRODUCIBILITY_CONFLICT")
            if interval_steps:
                # Continue the saved execution policy, avoiding a denser new
                # cadence that could exceed the logical task's storage budget.
                saved_interval = loaded.policy.get("requested_interval_steps")
                if type(saved_interval) is not int:
                    raise CheckpointError("CHECKPOINT_POLICY_INVALID")
                expected = _policy(parameters, saved_interval)
                if loaded.policy != expected:
                    raise CheckpointError("CHECKPOINT_POLICY_INVALID")
                self.policy = expected
        self.last_descriptor = self._source

    def validate_run(
        self,
        parameters: ExperimentParameters,
        *,
        task_id: str,
        attempt_id: str | None,
        artifacts_dir: Path,
    ) -> None:
        if (
            self.task_id != task_id
            or self.attempt_id != attempt_id
            or self.artifacts_dir != Path(artifacts_dir).resolve()
            or self.parameters_digest != checkpoint_parameters_sha256(parameters)
        ):
            raise CheckpointError("CHECKPOINT_CONTEXT_MISMATCH")

    def release_loaded(self) -> None:
        self.loaded = None

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "mode": "resumed_run" if self._source else "fresh_run",
            "logical_task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "parent_attempt_id": self._source["attempt_id"] if self._source else None,
            "checkpoint_id": self._source["checkpoint_id"] if self._source else None,
            "completed_steps_at_start": self._start,
            "fingerprint_sha256": _digest(self.fingerprint),
            "checkpoint_policy": self.policy,
            "repeatability_scope": "scientific arrays, sampled observables and model diagnostics under a matching numerical fingerprint",
        }

    def save_if_due(
        self,
        completed_steps: int,
        simulator: Any,
        *,
        samples: dict[str, np.ndarray],
        next_sample: int,
    ) -> dict[str, Any] | None:
        if completed_steps not in self.policy["schedule"] or (
            self._source is not None and completed_steps <= self._start
        ):
            return None
        metadata, arrays = capture_simulator(simulator)
        if (
            type(completed_steps) is not int
            or not 0 <= completed_steps <= self.parameters.num_steps
            or type(next_sample) is not int
            or next_sample < 0
        ):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        if (
            isinstance(self.parameters, LennardJonesParameters)
            and metadata["engine_state"]["steps"] != completed_steps
        ):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        expected_samples = _sampling_shapes(self.parameters, completed_steps, next_sample)
        if set(samples) != set(expected_samples):
            raise CheckpointError("CHECKPOINT_STATE_INVALID")
        packed, records = {}, {}
        total = 0
        for role, values in (("model", arrays), ("sample", samples)):
            for name, value in values.items():
                if (
                    not isinstance(name, str)
                    or not _ARRAY_NAME.fullmatch(name)
                    or not isinstance(value, np.ndarray)
                ):
                    raise CheckpointError("CHECKPOINT_STATE_INVALID")
                key = f"{role}_{name}"
                if (
                    value.dtype not in {np.dtype(np.float64), np.dtype(np.int64)}
                    or not np.isfinite(value).all()
                ):
                    raise CheckpointError("CHECKPOINT_STATE_INVALID")
                total += value.nbytes
                if total > MAX_CHECKPOINT_ARRAY_BYTES:
                    raise CheckpointError("CHECKPOINT_LIMIT_EXCEEDED")
                packed[key] = np.array(value, copy=True, order="C")
                records[key] = {
                    "role": role,
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "nbytes": value.nbytes,
                }
        _array_contract(self.parameters, completed_steps, next_sample, records)
        try:
            descriptor = self._publish(completed_steps, next_sample, metadata, packed, records)
        except OSError as error:
            raise CheckpointError("CHECKPOINT_PERSISTENCE_FAILED", 503) from error
        # Keep files after publication even if acknowledgement is lost: their
        # descriptor may already have been committed by a remote task store.
        self.last_descriptor = descriptor
        if self.register is not None:
            self.register(descriptor)
        return descriptor

    def _publish(self, step: int, cursor: int, model: dict, arrays: dict, records: dict) -> dict:
        checkpoint_id = uuid.uuid4().hex
        relative = f"{self.task_id}/{self.attempt_id}/checkpoints"
        root = _root(self.artifacts_dir)
        # The artifact root may have been created with missing ancestors by
        # startup or a standalone caller. Persist every directory link before
        # a durable Redis record is allowed to refer to this new checkpoint.
        for ancestor in root.parents:
            _fsync_directory(ancestor)
        parent = root
        for part in relative.split("/"):
            parent = parent / part
            if parent.is_symlink():
                raise CheckpointError("CHECKPOINT_INTEGRITY_FAILED")
            parent.mkdir(exist_ok=True)
            _fsync_directory(parent.parent)
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=parent))
        try:
            with (temporary / "state.npz").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            state_hash, state_size, _ = _snapshot(
                root,
                (temporary / "state.npz").relative_to(root).as_posix(),
                MAX_CHECKPOINT_ARCHIVE_BYTES,
            )
            created = datetime.now(UTC).isoformat()
            document = {
                "schema_version": 1,
                "checkpoint_id": checkpoint_id,
                "task_id": self.task_id,
                "attempt_id": self.attempt_id,
                "model_version": checkpoint_model_version(self.parameters),
                "completed_steps": step,
                "total_steps": self.parameters.num_steps,
                "parameters": checkpoint_parameters(self.parameters),
                "parameters_sha256": self.parameters_digest,
                "fingerprint": self.fingerprint,
                "fingerprint_sha256": _digest(self.fingerprint),
                "model_metadata": model,
                "next_sample": cursor,
                "arrays": records,
                "state_sha256": state_hash,
                "state_bytes": state_size,
                "checkpoint_policy": self.policy,
                "created_at": created,
            }
            encoded = _canonical(document) + b"\n"
            if len(encoded) > MAX_CHECKPOINT_METADATA_BYTES:
                raise CheckpointError("CHECKPOINT_LIMIT_EXCEEDED")
            with (temporary / "checkpoint.json").open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            target = parent / checkpoint_id
            if target.exists() or target.is_symlink():
                raise CheckpointError("CHECKPOINT_CONFLICT")
            _fsync_directory(temporary)
            os.rename(temporary, target)
            _fsync_directory(parent)
            prefix = f"{relative}/{checkpoint_id}"
            descriptor = {
                "schema_version": 1,
                "checkpoint_id": checkpoint_id,
                "task_id": self.task_id,
                "attempt_id": self.attempt_id,
                "completed_steps": step,
                "total_steps": self.parameters.num_steps,
                "model_version": document["model_version"],
                "parameters_sha256": self.parameters_digest,
                "fingerprint_sha256": document["fingerprint_sha256"],
                "created_at": created,
                "manifest": {
                    "path": f"{prefix}/checkpoint.json",
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "bytes": len(encoded),
                },
                "state": {"path": f"{prefix}/state.npz", "sha256": state_hash, "bytes": state_size},
            }
            return _descriptor(descriptor)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
