"""Exercise the actual publication CLI and verify its scientific artifacts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="The research extra is needed for artifact tests")
REPOSITORY = Path(__file__).resolve().parents[2]


def run_pipeline(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/run_paper_pipeline.py"),
            "--output-dir",
            str(output),
            "--seed",
            "42",
            "--num-points",
            "3",
            "--num-particles",
            "48",
            "--num-steps",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_paper_pipeline_reproduces_results_and_provenance(tmp_path: Path) -> None:
    datasets = []
    provenance = []
    for name in ("first", "second"):
        directory = tmp_path / name
        process = run_pipeline(directory)
        assert process.returncode == 0, process.stdout + process.stderr
        dataset = json.loads((directory / "results.json").read_text())
        assert len(dataset["rows"]) == 3
        assert [row["parameters"]["seed"] for row in dataset["rows"]] == [42, 43, 44]
        assert [row["parameters"]["initial_temperature"] for row in dataset["rows"]] == [
            100.0,
            550.0,
            1000.0,
        ]
        datasets.append(dataset)

        manifest = json.loads((directory / "manifest.json").read_text())
        assert (directory / manifest["figure"]).stat().st_size > 1000
        assert len(manifest["artifacts"]) >= 8
        for artifact in manifest["artifacts"]:
            artifact_path = directory / artifact["path"]
            assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]

        run_provenance = []
        for row in dataset["rows"]:
            result = row["result"]
            assert result["status"] == "completed"
            assert result["temperature_standard_error"] > 0
            assert result["units"]["temperature"] == "reduced"
            run_manifest = json.loads(
                (directory / "runs" / result["artifacts"]["manifest"]).read_text()
            )
            assert run_manifest["parameters"] == row["parameters"]
            # Timestamps vary; model, inputs, RNG, outputs and environment must match.
            run_provenance.append(
                {
                    key: run_manifest[key]
                    for key in (
                        "parameters",
                        "model",
                        "rng",
                        "outputs",
                        "uncertainty",
                        "environment",
                    )
                }
            )
        provenance.append(run_provenance)
    assert datasets[0] == datasets[1]
    assert provenance[0] == provenance[1]


def test_paper_pipeline_preserves_existing_artifacts(tmp_path: Path) -> None:
    sentinel = tmp_path / "results.json"
    sentinel.write_text("previous result\n")
    process = run_pipeline(tmp_path)
    assert process.returncode != 0
    assert "must be empty" in process.stderr
    assert sentinel.read_text() == "previous result\n"
