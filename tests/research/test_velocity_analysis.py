"""Scientific diagnostics and bounded, verified terminal-state loading."""

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scientific_api.research.velocity_analysis import (
    MAX_STATE_BYTES,
    analyze_velocities,
    load_velocity_state,
)


def save_state(path: Path, velocities: np.ndarray) -> str:
    np.savez_compressed(path, velocities=velocities)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_observed_moments_come_from_saved_velocities_without_mutation() -> None:
    velocities = np.array([[3.0, 4.0], [-3.0, -4.0], [0.0, 0.0]])
    original = velocities.copy()
    result = analyze_velocities(velocities, target_temperature=4.0, bins=8)
    assert result["analysis_version"] == "velocity-distribution-v1"
    assert result["sample_size"] == 3
    assert result["dimensions"] == 2
    assert result["moments"] == pytest.approx(
        {
            "vx_mean": 0.0,
            "vy_mean": 0.0,
            "vx_variance": 6.0,
            "vy_variance": 32.0 / 3.0,
            "speed_mean": 10.0 / 3.0,
            "speed_second_moment": 50.0 / 3.0,
            "temperature_estimate": 25.0 / 3.0,
        }
    )
    assert result["theory"]["components"]["variance"] == 4.0
    assert result["theory"]["speed"]["second_moment"] == 8.0
    assert result["theory"]["temperature_standard_error"] == 4.0 / math.sqrt(3)
    assert result["theory"]["variance_ddof"] == 0
    for histogram in result["histograms"].values():
        edges = np.array(histogram["bin_edges"])
        widths = np.diff(edges)
        assert len(edges) == 9
        assert np.all(widths > 0)
        assert sum(histogram["counts"]) == 3
        assert np.dot(histogram["density"], widths) == pytest.approx(1.0)
        assert np.dot(histogram["theoretical_density"], widths) == pytest.approx(
            histogram["theoretical_coverage"]
        )
        assert histogram["theoretical_coverage"] + histogram["theoretical_missing_mass"] == 1
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(velocities, original)
    assert result == analyze_velocities(velocities, target_temperature=4.0, bins=8)


def test_single_zero_particle_has_finite_bins_and_correct_2d_theory() -> None:
    result = analyze_velocities(np.zeros((1, 2)), target_temperature=4.0, bins=4)
    component = result["histograms"]["vx"]
    assert component["bin_edges"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    cdf = lambda value: 0.5 * (1 + math.erf(value / math.sqrt(8)))  # noqa: E731
    expected = [cdf(right) - cdf(left) for left, right in zip(range(-2, 2), range(-1, 3))]
    assert component["theoretical_bin_probability"] == pytest.approx(expected)
    assert component["theoretical_coverage"] == pytest.approx(math.erf(1 / math.sqrt(2)))
    assert result["moments"]["vx_variance"] == 0

    speed = result["histograms"]["speed"]
    edges = speed["bin_edges"]
    assert edges[0] == 0
    assert edges[-1] == 2
    rayleigh_mass = [
        math.exp(-(left**2) / 8) - math.exp(-(right**2) / 8)
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert speed["theoretical_bin_probability"] == pytest.approx(rayleigh_mass)
    assert speed["theoretical_coverage"] == pytest.approx(1 - math.exp(-0.5))
    # Density evaluated at a bin centre is not the probability averaged over it.
    midpoint_pdf = 0.25 / 4 * math.exp(-(0.25**2) / 8)
    assert not math.isclose(speed["theoretical_density"][0], midpoint_pdf, rel_tol=1e-4)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("constant,temperature", [(3.0, 4.0), (0.0, 1e-300), (1e100, 1.0)])
def test_constant_snapshots_have_positive_width_finite_histograms(
    constant: float, temperature: float
) -> None:
    result = analyze_velocities(np.full((4, 2), constant), target_temperature=temperature, bins=256)
    json.dumps(result, allow_nan=False)
    for histogram in result["histograms"].values():
        assert all(np.diff(histogram["bin_edges"]) > 0)
        assert sum(histogram["counts"]) == 4


def test_equilibrium_gaussian_and_rayleigh_moments_have_model_predicted_scale() -> None:
    size, temperature = 60_000, 3.0
    velocities = np.random.Generator(np.random.PCG64(2026)).normal(
        0, math.sqrt(temperature), size=(size, 2)
    )
    result = analyze_velocities(velocities, target_temperature=temperature)
    moments = result["moments"]
    for component in ("vx", "vy"):
        assert abs(moments[f"{component}_mean"]) < 6 * math.sqrt(temperature / size)
        assert abs(moments[f"{component}_variance"] - temperature) < (
            6 * temperature * math.sqrt(2 / size)
        )
    expected_speed = math.sqrt(math.pi * temperature / 2)
    speed_variance = (4 - math.pi) * temperature / 2
    assert abs(moments["speed_mean"] - expected_speed) < 6 * math.sqrt(speed_variance / size)
    assert abs(moments["temperature_estimate"] - temperature) < 6 * temperature / math.sqrt(size)
    assert result["theory"]["speed"]["distribution"] == "Rayleigh (2D Maxwell-Boltzmann)"


@pytest.mark.parametrize(
    "velocities",
    [
        np.zeros((0, 2)),
        np.zeros((3, 3)),
        np.zeros(2),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=object),
        np.array([[float("nan"), 0.0]]),
        np.array([[float("inf"), 0.0]]),
    ],
)
def test_analysis_rejects_unsupported_or_nonfinite_snapshots(velocities: np.ndarray) -> None:
    with pytest.raises(ValueError):
        analyze_velocities(velocities, target_temperature=1.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), True, "1", 1e7])
def test_analysis_rejects_invalid_target_temperature(temperature: object) -> None:
    with pytest.raises(ValueError):
        analyze_velocities(np.zeros((1, 2)), target_temperature=temperature)  # type: ignore[arg-type]


@pytest.mark.parametrize("bins", [1, 257, 2.5, True])
def test_analysis_bounds_histogram_work(bins: object) -> None:
    with pytest.raises(ValueError):
        analyze_velocities(np.zeros((1, 2)), target_temperature=1, bins=bins)  # type: ignore[arg-type]


def test_analysis_rejects_finite_inputs_with_nonfinite_observables() -> None:
    with pytest.raises(ValueError, match="finite float64"):
        analyze_velocities(np.full((1, 2), 1e308), target_temperature=1)


def test_loader_checks_hash_shape_dtype_and_finite_values(tmp_path: Path) -> None:
    velocities = np.array([[3.0, 4.0], [1.0, 2.0]])
    path = tmp_path / "state.npz"
    digest = save_state(path, velocities)
    loaded = load_velocity_state(path, expected_sha256=digest, expected_particles=2)
    np.testing.assert_array_equal(loaded, velocities)
    assert loaded.dtype == np.float64
    with pytest.raises(ValueError, match="SHA256"):
        load_velocity_state(path, expected_sha256="0" * 64, expected_particles=2)
    with pytest.raises(ValueError, match="header"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=3)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)


@pytest.mark.parametrize(
    "velocities",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=object),
        np.zeros((2, 3)),
        np.zeros((0, 2)),
        np.array([[1.0, float("nan")], [0.0, 0.0]]),
    ],
)
def test_loader_rejects_unsupported_arrays(tmp_path: Path, velocities: np.ndarray) -> None:
    path = tmp_path / "state.npz"
    digest = save_state(path, velocities)
    with pytest.raises(ValueError):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)


def test_loader_rejects_extra_members_and_non_archives(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    np.savez_compressed(path, velocities=np.zeros((2, 2)), unexpected=np.zeros(1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="only velocities.npy"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)
    path.write_bytes(b"not an archive")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Invalid velocity archive"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)


def test_loader_rejects_forged_large_shape_before_numpy_load(tmp_path: Path, monkeypatch) -> None:
    array = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        array, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 2)}
    )
    path = tmp_path / "state.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("velocities.npy", array.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def no_load(*args, **kwargs):
        pytest.fail("An invalid array header must be rejected before numpy.load")

    monkeypatch.setattr(np, "load", no_load)
    with pytest.raises(ValueError, match="header"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)


def test_loader_rejects_compressed_oversized_payload_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    # Small compressed bytes can claim an excessive uncompressed allocation.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("velocities.npy", b"\0" * (16_000_000 + 20_000))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Uncompressed"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=1)


def test_loader_bounds_file_read_and_rejects_truncated_array_payload(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.npz"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_STATE_BYTES + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        load_velocity_state(oversized, expected_sha256="0" * 64, expected_particles=1)

    array = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        array, {"descr": "<f8", "fortran_order": False, "shape": (2, 2)}
    )
    path = tmp_path / "truncated.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("velocities.npy", array.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="payload"):
        load_velocity_state(path, expected_sha256=digest, expected_particles=2)
