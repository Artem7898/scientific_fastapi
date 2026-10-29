"""The runtime benchmark must exercise persisted service results, not only a kernel."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_runtime.py"


def invoke(output: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *options],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_runtime_benchmark_validates_real_numpy_and_optional_numba_runs(tmp_path: Path) -> None:
    backends = ["numpy"]
    if importlib.util.find_spec("numba") is not None:
        backends.append("numba")
    output = tmp_path / "runtime.json"
    process = invoke(
        output,
        "--particles",
        "64",
        "--steps",
        "4",
        "--repeats",
        "2",
        "--seed",
        str(2**64 - 1),
        "--backends",
        ",".join(backends),
        "--require-backends",
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["benchmark"] == "production-runtime"
    assert report["workload"]["seed"] == 2**64 - 1
    assert report["workload"]["particle_steps_per_run"] == 256
    assert report["workload"]["velocity_array_bytes"] == 64 * 2 * 8
    assert set(report["backends"]) == set(backends)
    for name, measured in report["backends"].items():
        assert measured["status"] == "passed"
        assert measured["execution"]["backend"] == name
        assert measured["execution"]["dtype"] == "float64"
        assert measured["execution"]["device"] == "cpu"
        assert measured["cold_run"]["wall_seconds"] > 0
        for phase in (
            "state_persistence_seconds",
            "manifest_persistence_seconds",
            "environment_capture_seconds",
        ):
            assert 0 < measured["cold_run"][phase] < measured["cold_run"]["wall_seconds"]
        for phases in (measured["warm_full_run"], measured["warm_profile"]):
            for timing in phases.values():
                assert len(timing["samples_seconds"]) == 2
                assert 0 < timing["median_seconds"] <= timing["p95_seconds"]
        assert measured["validation"]["reference"] == "numpy"
        assert measured["validation"]["same_backend_repeatability"] == "exact final arrays"
        assert len(measured["validation"]["final_array_sha256"]) == 64
        assert measured["validation"]["max_absolute_velocity_error"] < 1e-11
        assert measured["measured_speedup_over_numpy"]["warm_full_run"] > 0
        assert measured["environment"]["source_sha256"]
    assert report["backends"]["numpy"]["validation"]["max_absolute_velocity_error"] == 0.0


@pytest.mark.parametrize(
    "options,reason",
    [
        (("--particles", "1000000", "--steps", "10000"), "num_particles * num_steps"),
        (("--backends", "jax"), "Choose unique runtime backends"),
        (("--temperature", "nan"), "initial_temperature"),
        (("--repeats", "0"), "repeats must"),
    ],
)
def test_runtime_benchmark_rejects_invalid_work_before_child_or_artifact_creation(
    tmp_path: Path,
    options: tuple[str, ...],
    reason: str,
) -> None:
    output = tmp_path / "invalid.json"
    process = invoke(output, *options)
    assert process.returncode != 0
    assert reason in process.stderr
    assert not output.exists()


def test_runtime_benchmark_does_not_overwrite_an_existing_report(tmp_path: Path) -> None:
    output = tmp_path / "reviewed.json"
    output.write_text("reviewed result\n")
    process = invoke(output, "--particles", "2", "--steps", "1", "--repeats", "1")
    assert process.returncode != 0
    assert "already exists" in process.stderr
    assert output.read_text() == "reviewed result\n"
