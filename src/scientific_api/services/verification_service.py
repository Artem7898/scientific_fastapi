"""Pinned, bounded verification snapshots and unsigned immutable certificates.

The archived commit is the authority for the expected bytes. Files are read
through no-follow descriptors and the exact hashed bytes reach the numerical
verifier. Neither admission nor execution updates the scientific task archive.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain.checkpoint import checkpoint_parameters_sha256
from ..domain.experiments import parameters_from_payload
from ..domain.simulation import SimulationCancelled
from ..schemas.verification import (
    VerificationCertificate,
    VerificationCheck,
    VerificationRequest,
    VerificationSource,
    VerificationVerdict,
)
from .provenance_evidence import build_archive_evidence
from .result_envelope_service import (
    MAX_ARRAY_BYTES,
    MAX_JSON_BYTES,
    ResultEnvelopeError,
    _root,
    _snapshot,
    read_result_envelope,
)
from .simulation_service import _environment

POLICY_VERSION = "result-verification-v1"
MAX_CERTIFICATE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
VERIFICATION_ID = re.compile(r"[0-9a-f]{32}")
_COMMON_CHECKS = frozenset(
    {"source_scientific_contract", "terminal_array_contract", "terminal_kinetic_energy"}
)
_MODEL_CHECKS = {
    "ideal-gas-ou-v1": {"terminal_temperature", "declared_snapshot_standard_error"},
    "langevin-free-v1": {
        "terminal_thermal_temperature",
        "trajectory_sampling_contract",
        "terminal_trajectory_binding",
        "presentation_coordinates",
        "raw_msd_series",
        "centered_msd_series",
        "diffusion_applicability",
    },
    "lennard-jones-2d-v1": {
        "terminal_thermal_temperature",
        "trajectory_sampling_contract",
        "terminal_trajectory_binding",
        "presentation_coordinates",
        "sampled_energy_closure",
        "sampled_pressure_closure",
        "terminal_potential_reference",
        "terminal_pressure_reference",
        "rdf_pair_counts",
    },
}
_RERUN_CHECKS = frozenset({"rerun_scientific_arrays", "rerun_scientific_scalars"})


class VerificationError(ValueError):
    def __init__(self, code: str, status_code: int = 409):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def verification_verdict(checks: list[VerificationCheck]) -> VerificationVerdict:
    """A partial pass never suppresses an unsuccessful required check."""
    statuses = {check.status for check in checks}
    if "failed" in statuses:
        return "failed"
    if statuses == {"passed"}:
        return "passed"
    if statuses == {"unsupported"}:
        return "unsupported"
    return "inconclusive"


def _missing_checks(model: str, rerun: bool, checks: list[VerificationCheck]) -> list[str]:
    required = _COMMON_CHECKS | _MODEL_CHECKS.get(model, {"unsupported_model_policy"})
    if rerun:
        required = required | _RERUN_CHECKS
    return sorted(required - {check.name for check in checks})


def prepare_verification(archive: dict[str, Any], *, rerun: bool) -> dict[str, Any]:
    """Pin committed evidence without requiring intact present-day files."""
    record = archive.get("record", {})
    evidence = archive.get("evidence")
    if record.get("status") != "completed":
        raise VerificationError("VERIFICATION_SOURCE_NOT_COMPLETED")
    if not isinstance(evidence, dict):
        raise VerificationError("VERIFICATION_SOURCE_EVIDENCE_UNAVAILABLE")
    try:
        task_id, attempt_id = record["task_id"], record["attempt_id"]
        if (
            task_id != archive["task_id"]
            or not IDENTIFIER.fullmatch(task_id)
            or not isinstance(attempt_id, str)
            or not IDENTIFIER.fullmatch(attempt_id)
            or not isinstance(evidence["artifacts"], list)
            or not 1 <= len(evidence["artifacts"]) <= 8
            or evidence.get("schema_version") != 1
            or evidence.get("model_version") != record.get("model_version")
            or not isinstance(evidence.get("manifest"), dict)
            or evidence["manifest"].get("sha256") != evidence.get("manifest_sha256")
        ):
            raise ValueError("Invalid archived identity")
        parameters = parameters_from_payload(archive["payload"])
        source = VerificationSource(
            task_id=task_id,
            attempt_id=attempt_id,
            model_version=record["model_version"],
            manifest_sha256=evidence["manifest_sha256"],
            model_contract_sha256=evidence["model_contract_sha256"],
            evidence_sha256=content_sha256(evidence),
            parameters_sha256=checkpoint_parameters_sha256(parameters),
        )
        pinned = {
            "source_id": task_id,
            "source": source.model_dump(mode="json"),
            "record": record,
            "parameters": archive["payload"],
            "evidence": evidence,
            "options": VerificationRequest(rerun=rerun).model_dump(mode="json"),
            "policy_version": POLICY_VERSION,
        }
        encoded = canonical_json(pinned)
        if len(encoded) > MAX_REQUEST_BYTES:
            raise VerificationError("VERIFICATION_SOURCE_LIMIT_EXCEEDED", 413)
        return json.loads(encoded)  # Detach all nested mutable archive values.
    except VerificationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise VerificationError("VERIFICATION_SOURCE_EVIDENCE_INVALID") from error


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _implementation() -> dict[str, Any]:
    digest = hashlib.sha256()
    for name in ("verification_service.py", "verification_science.py", "verification_rerun.py"):
        digest.update(name.encode("ascii"))
        digest.update(b"\x00")
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {"implementation_sha256": digest.hexdigest(), "environment": _environment()}


def verify_result(
    request: dict[str, Any],
    *,
    verification_id: str,
    verifier_attempt_id: str,
    artifacts_dir: Path,
    temporary_root: Path,
    check_cancelled: Callable[[], None],
    max_particle_steps: int = 1_000_000,
    max_lj_pair_checks: int = 5_000_000,
) -> VerificationCertificate:
    """Run one verification; every outcome is about this pinned occurrence."""
    from .verification_science import verify_scientific_result

    started = _now()
    source = VerificationSource.model_validate(request["source"])
    options = VerificationRequest.model_validate(request["options"])
    checks: list[VerificationCheck] = []
    envelope = None
    artifacts: dict[str, bytes] = {}
    check_cancelled()
    try:
        if (
            request["policy_version"] != POLICY_VERSION
            or request["source_id"] != source.task_id
            or source.evidence_sha256 != content_sha256(request["evidence"])
            or source.model_version != request["record"].get("model_version")
            or source.model_version != request["evidence"].get("model_version")
            or source.manifest_sha256 != request["evidence"].get("manifest_sha256")
            or source.model_contract_sha256 != request["evidence"].get("model_contract_sha256")
        ):
            raise ResultEnvelopeError("VERIFICATION_PIN_MISMATCH")
        envelope = read_result_envelope(
            request["record"], artifacts_dir=artifacts_dir, require_present=True
        )
        if (
            envelope.task_id != source.task_id
            or envelope.attempt_id != source.attempt_id
            or build_archive_evidence(envelope) != request["evidence"]
        ):
            raise ResultEnvelopeError("VERIFICATION_ARCHIVE_MISMATCH")
        parameters = parameters_from_payload(request["parameters"])
        manifest_parameters = parameters_from_payload(
            {**envelope.parameters, "model": source.model_version}
        )
        if (
            checkpoint_parameters_sha256(parameters) != source.parameters_sha256
            or checkpoint_parameters_sha256(manifest_parameters) != source.parameters_sha256
        ):
            raise ResultEnvelopeError("VERIFICATION_PARAMETERS_MISMATCH")
        root = _root(artifacts_dir)
        total = 0
        for descriptor in envelope.artifacts:
            check_cancelled()
            maximum = MAX_ARRAY_BYTES if descriptor.path.endswith(".npz") else MAX_JSON_BYTES
            if descriptor.bytes > maximum or total + descriptor.bytes > MAX_SNAPSHOT_BYTES:
                raise VerificationError("VERIFICATION_SNAPSHOT_LIMIT_EXCEEDED", 413)
            digest, size, data = _snapshot(root, descriptor.path, maximum, retain=True)
            if digest != descriptor.sha256 or size != descriptor.bytes:
                raise ResultEnvelopeError("VERIFICATION_SNAPSHOT_MISMATCH")
            assert data is not None
            artifacts[descriptor.name] = data
            total += size
        checks.append(
            VerificationCheck(
                name="artifact_integrity",
                status="passed",
                reason="exact_snapshot_matches_committed_archive_and_envelope",
                details={"artifact_count": len(artifacts), "snapshot_bytes": total},
            )
        )
    except VerificationError as error:
        checks.append(
            VerificationCheck(name="artifact_integrity", status="unsupported", reason=error.code)
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        checks.append(
            VerificationCheck(
                name="artifact_integrity",
                status="failed",
                reason=getattr(error, "code", "VERIFICATION_SOURCE_INVALID"),
            )
        )
    if checks[0].status == "passed":
        assert envelope is not None
        check_cancelled()
        checks.extend(
            VerificationCheck.model_validate(check)
            for check in verify_scientific_result(
                parameters, request["record"], artifacts, check_cancelled=check_cancelled
            )
        )
        if options.rerun:
            from .verification_rerun import verify_same_seed_rerun

            check_cancelled()
            checks.extend(
                VerificationCheck.model_validate(check)
                for check in verify_same_seed_rerun(
                    parameters,
                    envelope,
                    artifacts,
                    temporary_root=temporary_root,
                    cancelled=lambda: _cancelled(check_cancelled),
                    max_particle_steps=max_particle_steps,
                    max_lj_pair_checks=max_lj_pair_checks,
                )
            )
    elif options.rerun:
        checks.append(
            VerificationCheck(
                name="same_seed_rerun",
                status="inconclusive",
                reason="source_snapshot_unavailable",
            )
        )
    if verification_verdict(checks) == "passed":
        missing = _missing_checks(source.model_version, options.rerun, checks)
        if missing:
            checks.append(
                VerificationCheck(
                    name="verification_policy_coverage",
                    status="inconclusive",
                    reason="required_verification_checks_missing",
                    details={"missing_checks": missing},
                )
            )
    return VerificationCertificate(
        verification_id=verification_id,
        verifier_attempt_id=verifier_attempt_id,
        source=source,
        options=options,
        started_at=started,
        completed_at=_now(),
        verdict=verification_verdict(checks),
        checks=checks,
        verifier=_implementation(),
        limitations=[
            "Unsigned evidence; hashes establish agreement with the saved archive, not authorship.",
            "Checks apply to the pinned recorded artifacts at verification time, not future bytes.",
            "Independent reference calculations check stated numerical relations, not experimental validity.",
            "No equilibrium, convergence, confidence interval or cross-backend equivalence is certified.",
            "A same-seed rerun, when requested, observes agreement in the stated current environment.",
        ],
    )


def _cancelled(check: Callable[[], None]) -> bool:
    try:
        check()
    except SimulationCancelled:
        return True
    return False


def certificate_bytes(certificate: VerificationCertificate) -> bytes:
    data = canonical_json(certificate.model_dump(mode="json"))
    if len(data) > MAX_CERTIFICATE_BYTES:
        raise VerificationError("VERIFICATION_CERTIFICATE_LIMIT_EXCEEDED", 413)
    return data


def validate_verification_certificate(
    data: bytes, *, expected_sha256: str | None = None
) -> VerificationCertificate:
    """Offline structural/hash check. This does not repeat scientific checks."""
    if len(data) > MAX_CERTIFICATE_BYTES:
        raise VerificationError("VERIFICATION_CERTIFICATE_LIMIT_EXCEEDED", 413)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise VerificationError("VERIFICATION_CERTIFICATE_INTEGRITY_FAILED")
    try:
        certificate = VerificationCertificate.model_validate_json(data)
        names = [check.name for check in certificate.checks]
        if (
            not names
            or names[0] != "artifact_integrity"
            or len(names) != len(set(names))
            or certificate.verdict != verification_verdict(certificate.checks)
            or (
                certificate.verdict == "passed"
                and _missing_checks(
                    certificate.source.model_version, certificate.options.rerun, certificate.checks
                )
            )
            or certificate_bytes(certificate) != data
        ):
            raise ValueError("Invalid certificate policy or canonical encoding")
        return certificate
    except (ValueError, TypeError) as error:
        raise VerificationError("VERIFICATION_CERTIFICATE_INVALID") from error
