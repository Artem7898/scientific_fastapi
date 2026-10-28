"""Analyze a committed velocity snapshot and atomically publish new artifacts."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..domain.simulation import MODEL_VERSION, SimulationParameters
from ..research.reporting import REPORT_VERSION, render_velocity_report
from ..research.velocity_analysis import analyze_velocities, load_velocity_state

ARTIFACTS = {
    "analysis": ("analysis.json", "application/json"),
    "manifest": ("manifest.json", "application/json"),
    "png": ("velocity_distribution.png", "image/png"),
    "pdf": ("report.pdf", "application/pdf"),
}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024


class AnalysisSourceError(ValueError):
    """The committed run cannot be safely interpreted or verified."""


def _read_bytes(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
            raise ValueError("Artifact is not a bounded regular file")
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Artifact exceeds the size limit")
    return data


def _json(data: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    result = json.loads(data, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected an artifact object")
    return result


def _inside(root: Path, relative: str, *, exists: bool = True) -> Path:
    """Reject links and escapes; artifact directories are server-managed."""
    path = root
    for part in Path(relative).parts:
        if part in {"..", "."} or not part or Path(part).is_absolute():
            raise ValueError("Invalid artifact path")
        path = path / part
        if path.is_symlink():
            raise ValueError("Artifact paths must not contain symbolic links")
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ValueError("Artifact path escapes its root")
    return resolved


def _identities(result: dict[str, Any]) -> tuple[str, str]:
    task_id, attempt_id = result.get("task_id"), result.get("attempt_id")
    if (
        result.get("status") != "completed"
        or not isinstance(task_id, str)
        or not IDENTIFIER.fullmatch(task_id)
        or not isinstance(attempt_id, str)
        or not IDENTIFIER.fullmatch(attempt_id)
    ):
        raise ValueError("Expected a completed task with a committed attempt")
    return task_id, attempt_id


def _source(root: Path, result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], float]:
    try:
        task_id, attempt_id = _identities(result)
        prefix = f"{task_id}/{attempt_id}"
        expected = {
            "manifest": f"{prefix}/manifest.json",
            "final_state": f"{prefix}/final_state.npz",
        }
        if result.get("artifacts") != expected:
            raise ValueError("Unexpected committed artifact paths")
        manifest_bytes = _read_bytes(_inside(root, expected["manifest"]), MAX_JSON_BYTES)
        manifest = _json(manifest_bytes)
        if (
            manifest.get("task_id") != task_id
            or manifest.get("attempt_id") != attempt_id
            or manifest.get("outputs") != result
        ):
            raise ValueError("Source manifest does not describe the committed result")
        parameters = SimulationParameters(**manifest["parameters"])
        if (
            parameters.seed != result.get("seed")
            or parameters.num_steps != result.get("num_steps")
            or parameters.initial_temperature / math.sqrt(parameters.num_particles)
            != result.get("temperature_standard_error")
        ):
            raise ValueError("Source parameters disagree with the committed run")
        model = manifest["model"]
        if (
            not isinstance(model, dict)
            or result.get("model_version") != MODEL_VERSION
            or model.get("version") != MODEL_VERSION
            or model.get("dimensions") != 2
            or model.get("mass") != 1.0
            or model.get("boltzmann_constant") != 1.0
            or model.get("relaxation_rate") != 1.0
            or model.get("target_temperature") != parameters.initial_temperature
        ):
            raise ValueError("Only the reference 2D equilibrium model is supported")
        entries = [
            entry for entry in manifest["artifacts"] if entry["path"] == expected["final_state"]
        ]
        if len(entries) != 1 or not re.fullmatch(r"[0-9a-f]{64}", entries[0]["sha256"]):
            raise ValueError("Source state must have one SHA256 record")
        digest = entries[0]["sha256"]
        velocities = load_velocity_state(
            _inside(root, expected["final_state"]),
            expected_sha256=digest,
            expected_particles=parameters.num_particles,
        )
        with np.errstate(over="raise", invalid="raise"):
            energy = float(0.5 * np.einsum("ij,ij->", velocities, velocities))
        if not math.isfinite(energy) or not all(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
            for actual, expected in (
                (energy, result["total_energy"]),
                (energy / parameters.num_particles, result["final_temperature"]),
            )
        ):
            raise ValueError("Saved velocities disagree with the committed observables")
        source = {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "model_version": MODEL_VERSION,
            "state_sha256": digest,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        return velocities, source, parameters.initial_temperature
    except (OSError, ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as error:
        raise AnalysisSourceError(
            "Saved simulation artifacts failed integrity or model checks"
        ) from error


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def create_velocity_analysis(
    result: dict[str, Any], *, artifacts_dir: Path, bins: int, formats: tuple[str, ...]
) -> dict[str, Any]:
    """Read one verified source snapshot; never call the simulation engine."""
    root = Path(artifacts_dir).resolve()
    velocities, source, temperature = _source(root, result)
    try:
        analysis = analyze_velocities(velocities, target_temperature=temperature, bins=bins)
    except ValueError as error:
        raise AnalysisSourceError(
            "Saved state cannot produce finite velocity diagnostics"
        ) from error
    analysis_id = uuid.uuid4().hex
    created = datetime.now(UTC).isoformat()
    analysis.update(analysis_id=analysis_id, created_at=created, source=source)
    prefix = f"{source['task_id']}/{source['attempt_id']}/analyses"
    parent = _inside(root, prefix, exists=False)
    parent.mkdir(exist_ok=True)
    parent = _inside(root, prefix)
    temporary = Path(tempfile.mkdtemp(prefix=".analysis-", dir=parent))
    try:
        _write_json(temporary / "analysis.json", analysis)
        rendered = render_velocity_report(analysis, output_dir=temporary, formats=formats)
        names = ["analysis", *sorted(rendered)]
        records = []
        for name in names:
            filename, media_type = ARTIFACTS[name]
            path = temporary / filename
            data = _read_bytes(path, MAX_REPORT_BYTES)
            records.append(
                {
                    "name": name,
                    "filename": filename,
                    "media_type": media_type,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        source_hash = hashlib.sha256()
        for path in (
            Path(__file__),
            Path(__file__).parents[1] / "research" / "velocity_analysis.py",
            Path(__file__).parents[1] / "research" / "reporting.py",
        ):
            source_hash.update(path.name.encode())
            source_hash.update(path.read_bytes())
        environment = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "package_version": __version__,
            "analysis_source_sha256": source_hash.hexdigest(),
            "matplotlib": importlib.metadata.version("matplotlib") if formats else None,
        }
        manifest = {
            "schema_version": 1,
            "analysis_id": analysis_id,
            "analysis_version": analysis["analysis_version"],
            "report_version": REPORT_VERSION if formats else None,
            "created_at": created,
            "source": source,
            "parameters": {"bins": bins, "formats": list(formats)},
            "environment": environment,
            "artifacts": records,
        }
        _write_json(temporary / "manifest.json", manifest)
        destination = parent / analysis_id
        if destination.exists():
            raise FileExistsError("Analysis already exists")
        os.rename(temporary, destination)
        return {
            "analysis_id": analysis_id,
            "task_id": source["task_id"],
            "attempt_id": source["attempt_id"],
            "status": "completed",
            "analysis_version": analysis["analysis_version"],
            "artifacts": {name: ARTIFACTS[name][0] for name in ["manifest", *names]},
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def read_analysis_artifact(
    result: dict[str, Any], *, artifacts_dir: Path, analysis_id: str, name: str
) -> tuple[bytes, str, str]:
    """Return a bounded verified download snapshot from an allowed filename."""
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id) or name not in ARTIFACTS:
        raise ValueError("Unknown analysis artifact")
    task_id, attempt_id = _identities(result)
    root = Path(artifacts_dir).resolve()
    prefix = f"{task_id}/{attempt_id}/analyses/{analysis_id}"
    raw_manifest = _read_bytes(_inside(root, f"{prefix}/manifest.json"), MAX_JSON_BYTES)
    manifest = _json(raw_manifest)
    source = manifest.get("source")
    if (
        manifest.get("analysis_id") != analysis_id
        or not isinstance(source, dict)
        or source.get("task_id") != task_id
        or source.get("attempt_id") != attempt_id
    ):
        raise ValueError("Analysis does not belong to the committed attempt")
    filename, media_type = ARTIFACTS[name]
    if name == "manifest":
        return raw_manifest, filename, media_type
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not all(isinstance(item, dict) for item in artifacts):
        raise ValueError("Invalid analysis artifact records")
    entries = [item for item in artifacts if item.get("name") == name]
    if len(entries) != 1 or entries[0].get("filename") != filename:
        raise ValueError("Artifact was not generated for this analysis")
    data = _read_bytes(_inside(root, f"{prefix}/{filename}"), MAX_REPORT_BYTES)
    if hashlib.sha256(data).hexdigest() != entries[0].get("sha256"):
        raise ValueError("Analysis artifact checksum mismatch")
    return data, filename, media_type
