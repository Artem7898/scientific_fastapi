"""Engine comparisons validate forces/states and identify their timing scope."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_lj.py"


def invoke(output: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *options],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_lj_benchmark_compares_actual_direct_and_neighbor_engines(tmp_path):
    variants = ["numpy-direct", "numpy-verlet"]
    if importlib.util.find_spec("numba") is not None:
        variants.extend(("numba-direct", "numba-verlet"))
    output = tmp_path / "lj.json"
    process = invoke(
        output,
        "--particles",
        "36",
        "--steps",
        "4",
        "--repeats",
        "2",
        "--seed",
        str(2**64 - 1),
        "--variants",
        ",".join(variants),
        "--require-backends",
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["benchmark"] == "lj-engine"
    assert report["model_version"] == "lennard-jones-2d-v1"
    assert report["workload"]["num_particles"] == 36
    assert report["workload"]["num_steps"] == 4
    assert report["workload"]["seed"] == 2**64 - 1
    assert set(report["variants"]) == set(variants)
    for variant, sample in report["variants"].items():
        backend, interaction = variant.split("-", 1)
        assert sample["status"] == "passed"
        assert sample["execution"]["backend"] == backend
        assert sample["diagnostics"]["interaction_backend"] == interaction
        assert sample["diagnostics"]["analysis_pair_evaluations"] == 0
        assert sample["diagnostics"]["total_pair_checks"] > 0
        assert sample["cold_call"]["wall_seconds"] > 0
        for timing in sample["warm_call"].values():
            assert len(timing["samples_seconds"]) == 2
            assert 0 < timing["median_seconds"] <= timing["p95_seconds"]
        assert sample["validation"]["reference"] == "numpy-direct"
        assert (
            sample["validation"]["same_variant_repeatability"]
            == "exact final arrays and observables"
        )
        assert max(sample["validation"]["max_absolute_errors"].values()) < 1e-9
        assert all(len(value) == 64 for value in sample["validation"]["state_sha256"].values())
        assert len(sample["environment"]["source_sha256"]) == 64
        assert sample["warm_speed_ratio_over_numpy_direct"] > 0
    direct = report["variants"]["numpy-direct"]["diagnostics"]
    assert direct["pair_evaluations"] == 36 * 35 // 2 * 5
    assert direct["neighbor_rebuilds"] == 0


@pytest.mark.parametrize(
    "options,reason",
    [
        (("--particles", "10"), "particles must"),
        (("--steps", "1000"), "steps must"),
        (("--density", "nan"), "density must"),
        (("--repeats", "0"), "repeats must"),
        (("--variants", "jax-verlet"), "Choose unique variants"),
    ],
)
def test_lj_benchmark_rejects_invalid_work_before_measurement(tmp_path, options, reason):
    output = tmp_path / "invalid.json"
    process = invoke(output, *options)
    assert process.returncode != 0
    assert reason in process.stderr
    assert not output.exists()


def test_lj_benchmark_preserves_existing_reports(tmp_path):
    output = tmp_path / "preserved.json"
    output.write_text("reviewed report\n")
    process = invoke(output)
    assert process.returncode != 0
    assert "already exists" in process.stderr
    assert output.read_text() == "reviewed report\n"
