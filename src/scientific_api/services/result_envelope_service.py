"""Publish and recheck unsigned, bounded evidence sidecars without running physics.

The manifest does not hash the sidecar: the sidecar hashes the final manifest.
Every read reconstructs the evidence from current bytes and compares the stored
record. This detects disagreement with saved metadata, not a malicious author
who can replace the metadata and all of its matching hashes.
"""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ..schemas.analysis import VelocityAnalysisResponse
from ..schemas.model_contract import ModelContract
from ..schemas.result_envelope import (
    EnvelopeArtifact,
    EnvelopeIntegrity,
    EnvelopeModelContract,
    EnvelopeUncertainty,
    RecordedSimulation,
    ResultEnvelope,
)
from .model_registry import get_model_contract, model_contract_sha256

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
ANALYSIS_ID = re.compile(r"[0-9a-f]{32}")
DIGEST = re.compile(r"[0-9a-f]{64}")
RESULT_ADAPTER = TypeAdapter(RecordedSimulation | VelocityAnalysisResponse)
ANALYSIS_ARTIFACTS = {
    "manifest": ("manifest.json", "application/json"),
    "analysis": ("analysis.json", "application/json"),
    "png": ("velocity_distribution.png", "image/png"),
    "pdf": ("report.pdf", "application/pdf"),
}


class ResultEnvelopeError(ValueError):
    """A stable public reason; exception text never contains storage paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _parts(relative: str) -> list[str]:
    parts = relative.split("/")
    if (
        not relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in parts)
        or any("\x00" in part for part in parts)
    ):
        raise ResultEnvelopeError("ARTIFACT_PATH_INVALID")
    return parts


def _root(artifacts_dir: Path) -> Path:
    root = Path(artifacts_dir).absolute()
    for part in (root, *root.parents):
        if part.is_symlink():
            raise ResultEnvelopeError("ARTIFACT_PATH_INVALID")
    if not root.is_dir():
        raise ResultEnvelopeError("ARTIFACT_STORAGE_UNAVAILABLE")
    return root


def _open(root: Path, relative: str) -> int:
    """Walk directory descriptors so no component may be a symbolic link."""
    parts = _parts(relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)


def _snapshot(root: Path, relative: str, maximum: int, *, retain: bool = False):
    descriptor = _open(root, relative)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ResultEnvelopeError("ARTIFACT_SIZE_INVALID")
        digest = hashlib.sha256()
        chunks = []
        size = 0
        while chunk := stream.read(min(1024 * 1024, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ResultEnvelopeError("ARTIFACT_SIZE_INVALID")
            digest.update(chunk)
            if retain:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    if size != before.st_size or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ResultEnvelopeError("ARTIFACT_CHANGED_DURING_READ")
    return digest.hexdigest(), size, b"".join(chunks) if retain else None


def _json(data: bytes) -> dict[str, Any]:
    def constant(_value: str):
        raise ResultEnvelopeError("MANIFEST_INVALID")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for name, value in items:
            if name in result:
                raise ResultEnvelopeError("MANIFEST_INVALID")
            result[name] = value
        return result

    def floating(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ResultEnvelopeError("MANIFEST_INVALID")
        return parsed

    result = json.loads(
        data, parse_constant=constant, parse_float=floating, object_pairs_hook=pairs
    )
    if not isinstance(result, dict):
        raise ResultEnvelopeError("MANIFEST_INVALID")
    return result


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultEnvelopeError("MANIFEST_INVALID")
    return value


def _identity(result: dict[str, Any]) -> tuple[str, str | None, str]:
    task, attempt = result.get("task_id"), result.get("attempt_id")
    if (
        result.get("status") != "completed"
        or not isinstance(task, str)
        or not IDENTIFIER.fullmatch(task)
        or (
            attempt is not None
            and (not isinstance(attempt, str) or not IDENTIFIER.fullmatch(attempt))
        )
    ):
        raise ResultEnvelopeError("RESULT_IDENTITY_INVALID")
    return task, attempt, f"{task}/{attempt}" if attempt is not None else task


def _manifest(root: Path, prefix: str) -> tuple[dict[str, Any], EnvelopeArtifact]:
    path = f"{prefix}/manifest.json"
    digest, size, data = _snapshot(root, path, MAX_JSON_BYTES, retain=True)
    assert data is not None
    return _json(data), EnvelopeArtifact(
        name="manifest", path=path, media_type="application/json", sha256=digest, bytes=size
    )


def _marker(manifest: dict[str, Any], prefix: str) -> bool:
    if "result_envelope" not in manifest:
        return False
    marker = manifest["result_envelope"]
    if (
        not isinstance(marker, dict)
        or type(marker.get("schema_version")) is not int
        or marker != {"schema_version": 1, "path": f"{prefix}/envelope.json"}
    ):
        raise ResultEnvelopeError("ENVELOPE_MARKER_INVALID")
    return True


def _record(
    root: Path,
    *,
    name: str,
    path: str,
    media_type: str,
    expected: dict[str, Any],
) -> EnvelopeArtifact:
    maximum = MAX_ARRAY_BYTES if path.endswith(".npz") else MAX_REPORT_BYTES
    digest, size, _ = _snapshot(root, path, maximum)
    if (
        not isinstance(expected.get("sha256"), str)
        or not DIGEST.fullmatch(expected["sha256"])
        or expected["sha256"] != digest
        or (
            "bytes" in expected
            and (type(expected["bytes"]) is not int or expected["bytes"] != size)
        )
    ):
        raise ResultEnvelopeError("ARTIFACT_INTEGRITY_FAILED")
    return EnvelopeArtifact(name=name, path=path, media_type=media_type, sha256=digest, bytes=size)


def _uncertainty(result: dict[str, Any], manifest: dict[str, Any]) -> EnvelopeUncertainty:
    details = _object(manifest.get("uncertainty"))
    model = result["model_version"]
    if model == "lennard-jones-2d-v1":
        return EnvelopeUncertainty(
            availability="unavailable",
            reason="correlated_interacting_samples_have_no_declared_uncertainty_estimator",
            details=details,
        )
    if model == "langevin-free-v1":
        diffusion = _object(result.get("diffusion"))
        available = diffusion.get("status") == "available"
        return EnvelopeUncertainty(
            availability="available" if available else "partial",
            reason=(
                "declared_snapshot_and_diffusion_sampling_errors_only"
                if available
                else "diffusion_uncertainty_unavailable_for_declared_fit_window"
            ),
            details={
                **details,
                "thermal_temperature_standard_error": result["thermal_temperature_standard_error"],
                "diffusion": diffusion,
            },
        )
    return EnvelopeUncertainty(
        availability="available",
        reason="declared_independent_equilibrium_snapshot_standard_error_only",
        details={**details, "temperature_standard_error": result["temperature_standard_error"]},
    )


def _simulation(
    root: Path,
    result: dict[str, Any],
    contract_snapshot: ModelContract | None = None,
    *,
    allow_historical_metadata: bool = False,
) -> tuple[ResultEnvelope, str, dict[str, Any]]:
    task, attempt, prefix = _identity(result)
    parsed = RESULT_ADAPTER.validate_python(result)
    if isinstance(parsed, VelocityAnalysisResponse):
        raise ResultEnvelopeError("RESULT_IDENTITY_INVALID")
    model_version = parsed.model_version
    current_contract = get_model_contract(model_version)
    contract = contract_snapshot or current_contract
    if (
        contract.model_version != model_version
        or contract.required_artifacts != current_contract.required_artifacts
    ):
        raise ResultEnvelopeError("MODEL_CONTRACT_MISMATCH")
    manifest, manifest_record = _manifest(root, prefix)
    if (
        manifest.get("task_id") != task
        or manifest.get("attempt_id") != attempt
        or manifest.get("outputs") != result
        or _object(manifest.get("model")).get("version") != model_version
        or manifest.get("schema_version") != 1
    ):
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    parameters = _object(manifest.get("parameters"))
    historical = allow_historical_metadata and "result_envelope" not in manifest
    rng = _object(manifest.get("rng", {} if historical else None))
    execution = _object(manifest.get("execution", {} if historical else None))
    backend_parameters = parameters.get("backend")
    backend_execution = execution.get("backend")
    backend_mismatch = (
        (
            (backend_parameters is not None and backend_parameters not in {"numpy", "numba"})
            or (backend_execution is not None and backend_execution not in {"numpy", "numba"})
            or (
                backend_parameters is not None
                and backend_execution is not None
                and backend_parameters != backend_execution
            )
        )
        if historical
        else (
            backend_parameters != backend_execution or backend_execution not in {"numpy", "numba"}
        )
    )
    if (
        parameters.get("seed") != parsed.seed
        or (rng.get("seed", parsed.seed) if historical else rng.get("seed")) != parsed.seed
        or parameters.get("num_steps") != parsed.num_steps
        or backend_mismatch
        or (rng.get("bit_generator", "PCG64") if historical else rng.get("bit_generator"))
        != "PCG64"
    ):
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    expected_paths = {
        item.name: f"{prefix}/{item.filename}" for item in contract.required_artifacts
    }
    if result.get("artifacts") != expected_paths:
        raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list) or not all(isinstance(item, dict) for item in raw_records):
        raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
    records_by_path = {}
    for item in raw_records:
        path = item.get("path")
        if not isinstance(path, str) or path in records_by_path:
            raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
        records_by_path[path] = item
    if set(records_by_path) != set(expected_paths.values()) - {manifest_record.path}:
        raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
    artifacts = [manifest_record]
    for item in contract.required_artifacts:
        if item.name == "manifest":
            continue
        path = expected_paths[item.name]
        artifacts.append(
            _record(
                root,
                name=item.name,
                path=path,
                media_type=item.media_type,
                expected=records_by_path[path],
            )
        )
    diagnostics = manifest.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ResultEnvelopeError("MANIFEST_INVALID")
    envelope = ResultEnvelope(
        kind="simulation",
        task_id=task,
        attempt_id=attempt,
        result=parsed,
        model_contract=EnvelopeModelContract(
            model_version=model_version,
            schema_version=contract.schema_version,
            sha256=model_contract_sha256(contract),
            snapshot=contract,
        ),
        integrity=EnvelopeIntegrity(status="verified"),
        manifest=manifest_record,
        artifacts=artifacts,
        parameters=parameters,
        execution=execution,
        rng=rng,
        environment=_object(manifest.get("environment", {} if historical else None)),
        diagnostics=diagnostics,
        units=parsed.units,
        uncertainty=(
            EnvelopeUncertainty(
                availability="unavailable",
                reason="source_manifest_does_not_record_uncertainty_metadata",
            )
            if historical and "uncertainty" not in manifest
            else _uncertainty(result, manifest)
        ),
    )
    return envelope, prefix, manifest


def _analysis(
    root: Path,
    result: dict[str, Any],
    analysis_id: str,
    contract_snapshot: ModelContract | None = None,
) -> tuple[ResultEnvelope, str, dict[str, Any]]:
    task, attempt, run_prefix = _identity(result)
    if not ANALYSIS_ID.fullmatch(analysis_id) or attempt is None:
        raise ResultEnvelopeError("RESULT_IDENTITY_INVALID")
    prefix = f"{run_prefix}/analyses/{analysis_id}"
    try:
        manifest, manifest_record = _manifest(root, prefix)
    except FileNotFoundError as error:
        raise ResultEnvelopeError("ANALYSIS_NOT_FOUND") from error
    source = _object(manifest.get("source"))
    if (
        manifest.get("analysis_id") != analysis_id
        or manifest.get("schema_version") != 1
        or source.get("task_id") != task
        or source.get("attempt_id") != attempt
        or source.get("model_version") != "ideal-gas-ou-v1"
    ):
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    source_manifest, source_record = _manifest(root, run_prefix)
    source_result = _object(source_manifest.get("outputs"))
    if "analysis_id" not in result and result != source_result:
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    source_envelope, _, _ = _simulation(
        root, source_result, contract_snapshot, allow_historical_metadata=True
    )
    if source_envelope.manifest is None or source_envelope.manifest.sha256 != source_record.sha256:
        raise ResultEnvelopeError("SOURCE_INTEGRITY_FAILED")
    state_record = next(item for item in source_envelope.artifacts if item.name == "final_state")
    if (
        source.get("manifest_sha256") != source_record.sha256
        or source.get("state_sha256") != state_record.sha256
    ):
        raise ResultEnvelopeError("SOURCE_INTEGRITY_FAILED")
    parameters = _object(manifest.get("parameters"))
    formats = parameters.get("formats")
    if (
        not isinstance(formats, list)
        or any(item not in {"png", "pdf"} for item in formats)
        or len(set(formats)) != len(formats)
        or type(parameters.get("bins")) is not int
        or not 2 <= parameters["bins"] <= 256
    ):
        raise ResultEnvelopeError("MANIFEST_INVALID")
    expected_names = {"analysis", *formats}
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list) or not all(isinstance(item, dict) for item in raw_records):
        raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
    by_name = {}
    for item in raw_records:
        name = item.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
        by_name[name] = item
    if set(by_name) != expected_names:
        raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
    artifacts = [manifest_record]
    for name in sorted(expected_names):
        filename, media_type = ANALYSIS_ARTIFACTS[name]
        record = by_name[name]
        if record.get("filename") != filename or record.get("media_type") != media_type:
            raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
        artifacts.append(
            _record(
                root, name=name, path=f"{prefix}/{filename}", media_type=media_type, expected=record
            )
        )
    analysis_digest, _, analysis_bytes = _snapshot(
        root, f"{prefix}/analysis.json", MAX_JSON_BYTES, retain=True
    )
    if analysis_digest != next(item.sha256 for item in artifacts if item.name == "analysis"):
        raise ResultEnvelopeError("ARTIFACT_INTEGRITY_FAILED")
    assert analysis_bytes is not None
    analysis = _json(analysis_bytes)
    if (
        analysis.get("analysis_id") != analysis_id
        or analysis.get("analysis_version") != manifest.get("analysis_version")
        or analysis.get("source") != source
    ):
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    snapshot = VelocityAnalysisResponse(
        analysis_id=analysis_id,
        task_id=task,
        attempt_id=attempt,
        analysis_version=manifest["analysis_version"],
        artifacts={name: ANALYSIS_ARTIFACTS[name][0] for name in {"manifest", *expected_names}},
    )
    if "analysis_id" in result and snapshot.model_dump() != result:
        raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
    return (
        ResultEnvelope(
            kind="velocity_analysis",
            task_id=task,
            attempt_id=attempt,
            analysis_id=analysis_id,
            result=snapshot,
            model_contract=source_envelope.model_contract,
            integrity=EnvelopeIntegrity(status="verified"),
            manifest=manifest_record,
            artifacts=artifacts,
            parameters=parameters,
            execution=source_envelope.execution,
            rng=source_envelope.rng,
            environment=_object(manifest.get("environment")),
            diagnostics={"analysis": analysis},
            units=source_envelope.units,
            uncertainty=source_envelope.uncertainty,
            source=source,
        ),
        prefix,
        manifest,
    )


def _prepare(
    root: Path,
    result: dict[str, Any],
    analysis_id: str | None = None,
    contract_snapshot: ModelContract | None = None,
):
    analysis_id = analysis_id or result.get("analysis_id")
    if analysis_id is not None:
        if not isinstance(analysis_id, str):
            raise ResultEnvelopeError("RESULT_IDENTITY_INVALID")
        return _analysis(root, result, analysis_id, contract_snapshot)
    return _simulation(root, result, contract_snapshot)


def _unavailable(
    result: dict[str, Any], manifest: dict[str, Any], analysis_id: str | None
) -> ResultEnvelope:
    """Do not impose today's execution metadata on unmarked older records."""
    task, attempt, _ = _identity(result)
    if analysis_id is None:
        if (
            manifest.get("outputs") != result
            or manifest.get("task_id") != task
            or manifest.get("attempt_id") != attempt
        ):
            raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
        parsed = RESULT_ADAPTER.validate_python(result)
        kind = "simulation"
    else:
        source = _object(manifest.get("source"))
        if (
            manifest.get("analysis_id") != analysis_id
            or source.get("task_id") != task
            or source.get("attempt_id") != attempt
            or attempt is None
        ):
            raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
        records = manifest.get("artifacts")
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
        names = [item.get("name") for item in records]
        if (
            "analysis" not in names
            or len(set(names)) != len(names)
            or any(name not in {"analysis", "png", "pdf"} for name in names)
        ):
            raise ResultEnvelopeError("ARTIFACT_RECORDS_INVALID")
        parsed = VelocityAnalysisResponse(
            analysis_id=analysis_id,
            task_id=task,
            attempt_id=attempt,
            analysis_version=manifest["analysis_version"],
            artifacts={name: ANALYSIS_ARTIFACTS[name][0] for name in ["manifest", *names]},
        )
        if "analysis_id" in result and parsed.model_dump() != result:
            raise ResultEnvelopeError("MANIFEST_RESULT_MISMATCH")
        kind = "velocity_analysis"
    return ResultEnvelope(
        kind=kind,
        task_id=task,
        attempt_id=attempt,
        analysis_id=analysis_id,
        result=parsed,
        integrity=EnvelopeIntegrity(status="not_verified", reason="envelope_not_recorded"),
        uncertainty=EnvelopeUncertainty(availability="unavailable", reason="envelope_not_recorded"),
    )


def _stored(root: Path, prefix: str, expected: ResultEnvelope) -> ResultEnvelope:
    try:
        _, _, data = _snapshot(root, f"{prefix}/envelope.json", MAX_ENVELOPE_BYTES, retain=True)
    except FileNotFoundError as error:
        raise ResultEnvelopeError("ENVELOPE_MISSING") from error
    assert data is not None
    stored = ResultEnvelope.model_validate(_json(data))
    if stored.model_dump() != expected.model_dump():
        raise ResultEnvelopeError("ENVELOPE_MISMATCH")
    return expected


def _historical_contract(root: Path, prefix: str) -> ModelContract:
    try:
        _, _, data = _snapshot(root, f"{prefix}/envelope.json", MAX_ENVELOPE_BYTES, retain=True)
    except FileNotFoundError as error:
        raise ResultEnvelopeError("ENVELOPE_MISSING") from error
    assert data is not None
    record = ResultEnvelope.model_validate(_json(data)).model_contract
    if (
        record is None
        or record.model_version != record.snapshot.model_version
        or record.schema_version != record.snapshot.schema_version
        or record.sha256 != model_contract_sha256(record.snapshot)
    ):
        raise ResultEnvelopeError("MODEL_CONTRACT_MISMATCH")
    return record.snapshot


def read_result_envelope(
    result: dict[str, Any],
    *,
    artifacts_dir: Path,
    analysis_id: str | None = None,
    require_present: bool = False,
) -> ResultEnvelope:
    """Hash current bytes; never trust a persisted ``verified`` status alone."""
    try:
        root = _root(artifacts_dir)
        _, _, prefix = _identity(result)
        selected_analysis = analysis_id or result.get("analysis_id")
        if selected_analysis is not None:
            if not isinstance(selected_analysis, str) or not ANALYSIS_ID.fullmatch(
                selected_analysis
            ):
                raise ResultEnvelopeError("ANALYSIS_NOT_FOUND")
            prefix = f"{prefix}/analyses/{selected_analysis}"
        try:
            manifest, _ = _manifest(root, prefix)
        except FileNotFoundError as error:
            if selected_analysis is not None:
                raise ResultEnvelopeError("ANALYSIS_NOT_FOUND") from error
            raise
        if not _marker(manifest, prefix):
            # A sidecar without its manifest marker is inconsistent, including
            # when somebody removed only the marker after publication.
            try:
                descriptor = _open(root, f"{prefix}/envelope.json")
            except FileNotFoundError:
                if require_present:
                    raise ResultEnvelopeError("ENVELOPE_MISSING") from None
                return _unavailable(result, manifest, selected_analysis)
            else:
                os.close(descriptor)
                raise ResultEnvelopeError("ENVELOPE_MARKER_MISSING")
        contract = _historical_contract(root, prefix)
        envelope, prefix, _ = _prepare(root, result, analysis_id, contract)
        return _stored(root, prefix, envelope)
    except ResultEnvelopeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
        raise ResultEnvelopeError("ENVELOPE_INVALID") from error


def publish_result_envelope(result: dict[str, Any], *, artifacts_dir: Path) -> ResultEnvelope:
    """Atomically write a sidecar only after the final manifest exists."""
    temporary = None
    try:
        root = _root(artifacts_dir)
        envelope, prefix, manifest = _prepare(root, result)
        if not _marker(manifest, prefix):
            raise ResultEnvelopeError("ENVELOPE_MARKER_MISSING")
        payload = (
            json.dumps(envelope.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode()
        if len(payload) > MAX_ENVELOPE_BYTES:
            raise ResultEnvelopeError("ARTIFACT_SIZE_INVALID")
        # All components have just been walked without following links. The
        # parent is a unique, server-managed attempt or analysis directory.
        parent = root.joinpath(*_parts(prefix))
        descriptor, temporary = tempfile.mkstemp(prefix=".envelope-", dir=parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, parent / "envelope.json")
        temporary = None
        return _stored(root, prefix, envelope)
    except ResultEnvelopeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
        raise ResultEnvelopeError("ENVELOPE_INVALID") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
