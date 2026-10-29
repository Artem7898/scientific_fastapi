"""Interacting BAOAB configurations agree with independent canonical quadrature."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "validate_lj_reference.py"
REFERENCE = REPOSITORY / "tests" / "reference" / "lj2d_canonical.json"


def test_reference_identifies_independent_matched_two_dimensional_model():
    reference = json.loads(REFERENCE.read_text())
    assert reference["physics"]["model_version"] == "lennard-jones-2d-v1"
    assert reference["physics"]["dimensions"] == 2
    assert reference["physics"]["num_particles"] == 3
    assert reference["physics"]["cutoff_policy"] == "shifted-force"
    assert reference["source"]["production_imports"] is False
    assert reference["source"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert [item["grid_per_coordinate"] for item in reference["grids"]] == [64, 96, 128]
    assert -0.43 < reference["estimates"]["potential_energy"]["value"] < -0.41
    for estimate in reference["estimates"].values():
        assert (
            abs(estimate["successive_grid_differences"][-1])
            < estimate["absolute_resolution_allowance"]
        )


def test_interacting_nvt_samples_match_independent_canonical_position_reference(tmp_path):
    pytest.importorskip("numba")
    output = tmp_path / "canonical-comparison.json"
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--reference", str(REFERENCE), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, (
        process.stdout + process.stderr + (output.read_text() if output.exists() else "")
    )
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["parameters"]["replicas"] == 12
    assert len(report["replica_means"]) == 12
    assert len({item["seed"] for item in report["replica_means"]}) == 12
    assert all(item["passed"] for item in report["checks"].values())
    assert report["checks"]["potential_energy"]["mean"] < -0.25
    assert "replica" in report["uncertainty"]
