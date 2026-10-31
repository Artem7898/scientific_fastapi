"""Validate a downloaded unsigned LJ report and its digest without reanalysis."""

import argparse
import json
from pathlib import Path

from scientific_api.services.lj_research_service import (
    MAX_REPORT_BYTES,
    ResearchError,
    validate_lj_research_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--sha256", required=True, help="Digest recorded independently at download")
    arguments = parser.parse_args()
    try:
        with arguments.report.open("rb") as stream:
            data = stream.read(MAX_REPORT_BYTES + 1)
        report = validate_lj_research_report(data, expected_sha256=arguments.sha256)
    except (OSError, ResearchError):
        parser.exit(1, "Research report hash or structure validation failed\n")
    print(
        json.dumps(
            {
                "analysis_id": report.analysis_id,
                "kind": report.kind,
                "availability": report.availability,
                "report_bytes_and_structure": "verified",
                "authenticity": "unsigned",
                "statistical_analysis_repeated_offline": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
