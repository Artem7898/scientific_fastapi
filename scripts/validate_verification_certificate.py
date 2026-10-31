"""Check downloaded certificate bytes and structure offline; do not rerun physics."""

import argparse
import json
from pathlib import Path

from scientific_api.services.verification_service import (
    MAX_CERTIFICATE_BYTES,
    VerificationError,
    validate_verification_certificate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--sha256", required=True, help="Digest recorded independently at download")
    args = parser.parse_args()
    try:
        with args.certificate.open("rb") as stream:
            raw = stream.read(MAX_CERTIFICATE_BYTES + 1)
        certificate = validate_verification_certificate(raw, expected_sha256=args.sha256)
    except (OSError, VerificationError):
        parser.exit(1, "Certificate hash or structure validation failed\n")
    print(
        json.dumps(
            {
                "verification_id": certificate.verification_id,
                "source_task_id": certificate.source.task_id,
                "recorded_verdict": certificate.verdict,
                "certificate_bytes_and_structure": "verified",
                "authenticity": "unsigned",
                "scientific_checks_repeated_offline": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
