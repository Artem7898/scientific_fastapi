"""The benchmark must compare equal workloads and reject excessive allocations."""

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def test_benchmark_validates_serial_thread_and_process_outputs(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/benchmark.py"),
            "--particles",
            "64",
            "--steps",
            "4",
            "--repeats",
            "2",
            "--workers",
            "2",
            "--seed",
            "42",
            "--backends",
            "numpy,threads,processes",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["workload"]["particle_steps_per_sample"] == 512
    assert report["workload"]["independent_batches"] == 2
    assert report["workload"]["rng"] == "PCG64"
    assert isinstance(report["environment"]["free_threaded_build"], bool)
    assert isinstance(report["environment"]["gil_enabled_after_backends"], bool)
    assert set(report["backends"]) == {"numpy", "threads", "processes"}
    for result in report["backends"].values():
        assert result["status"] == "passed"
        assert isinstance(result["gil_enabled"], bool)
        assert len(result["samples_seconds"]) == 2
        assert 0 < result["median_seconds"] <= result["p95_seconds"]
        assert result["warmup_seconds"] > 0
        assert result["max_absolute_error"] == 0.0


def test_benchmark_rejects_excessive_workload_before_allocation(tmp_path: Path) -> None:
    output = tmp_path / "excessive.json"
    process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/benchmark.py"),
            "--particles",
            "1000000",
            "--steps",
            "10000",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode != 0
    assert "256 MiB" in process.stderr
    assert not output.exists()
