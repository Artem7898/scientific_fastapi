"""Transport estimators respect drift, finite-time bias and correlated frames."""

import hashlib
import io
import json
import math
import zipfile

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.models.langevin_engine import create_langevin_simulator
from scientific_api.research.transport_analysis import (
    MAX_ARCHIVE_BYTES,
    analyze_transport,
    load_trajectory_state,
)


def parameters(**changes):
    return LangevinParameters(
        **{
            "initial_temperature": 1.0,
            "bath_temperature": 1.0,
            "num_particles": 5,
            "time_step": 1.0,
            "num_steps": 20,
            "sample_stride": 1,
            **changes,
        }
    )


def linear_covariance_trajectory(configuration, drift=(0.0, 0.0)):
    steps = np.asarray(configuration.sampling_steps(), dtype=np.int64)
    times = steps * configuration.time_step
    vectors = np.array([[1.0, 0.0], [-0.5, 1.0], [0.0, -2.0], [2.0, 1.0], [-2.5, 0.0]])
    origin = np.asarray(configuration.initial_position, dtype=np.float64)
    coordinates = (
        origin
        + np.sqrt(times)[:, None, None] * vectors[None, :, :]
        + times[:, None, None] ** 2 * np.asarray(drift)[None, None, :]
    )
    return coordinates, steps


def test_raw_and_centered_msd_have_distinct_normalizations_and_drift_behavior():
    configuration = parameters(initial_position=(1.0, -2.0))
    baseline, steps = linear_covariance_trajectory(configuration)
    drifting, _ = linear_covariance_trajectory(configuration, drift=(10.0, -3.0))
    no_drift = analyze_transport(baseline, steps, parameters=configuration)
    observed = analyze_transport(drifting, steps, parameters=configuration)
    np.testing.assert_allclose(observed["centered_msd"], no_drift["centered_msd"], rtol=1e-12)
    assert observed["msd"][-1] > 1000 * observed["centered_msd"][-1]
    displacement = drifting[-1] - drifting[0]
    np.testing.assert_allclose(observed["mean_displacement"][-1], displacement.mean(axis=0))
    assert observed["msd"][-1] == pytest.approx(np.mean(np.sum(displacement**2, axis=1)))
    assert observed["centered_msd"][-1] == pytest.approx(np.var(displacement, axis=0, ddof=1).sum())
    assert observed["msd_standard_error"][-1] == pytest.approx(
        np.std(np.sum(displacement**2, axis=1), ddof=1) / math.sqrt(5)
    )
    assert observed["msd"][0] == observed["centered_msd"][0] == 0.0
    assert observed["diffusion"]["estimate"] == pytest.approx(no_drift["diffusion"]["estimate"])
    assert observed["coordinate_convention"] == "unwrapped"
    json.dumps(observed, allow_nan=False)


def test_particle_jackknife_matches_explicit_leave_one_out_fits():
    configuration = parameters(diffusion_fit_start_step=5, diffusion_fit_end_step=15)
    steps = np.asarray(configuration.sampling_steps(), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(123))
    coordinates = np.zeros((len(steps), configuration.num_particles, 2), dtype=np.float64)
    coordinates[1:] = np.cumsum(rng.normal(size=coordinates[1:].shape), axis=0)
    coordinates += (steps[:, None, None] ** 2) * np.array([0.5, -0.2])
    result = analyze_transport(coordinates, steps, parameters=configuration)
    diffusion = result["diffusion"]
    assert diffusion["status"] == "available"
    selected = (steps >= 5) & (steps <= 15)
    times = steps[selected] * configuration.time_step
    design = np.column_stack([times, np.ones(len(times))])
    displacement = coordinates[selected] - coordinates[0]
    centered = np.var(displacement, axis=1, ddof=1).sum(axis=1)
    slope, intercept = np.linalg.lstsq(design, centered, rcond=None)[0]
    assert diffusion["estimate"] == pytest.approx(slope / 4, rel=1e-12)
    assert diffusion["intercept"] == pytest.approx(intercept, rel=1e-12)
    deleted = []
    for index in range(configuration.num_particles):
        subset = np.delete(displacement, index, axis=1)
        subset_msd = np.var(subset, axis=1, ddof=1).sum(axis=1)
        deleted.append(np.linalg.lstsq(design, subset_msd, rcond=None)[0][0] / 4)
    expected_se = math.sqrt(
        (configuration.num_particles - 1)
        / configuration.num_particles
        * np.sum((np.asarray(deleted) - np.mean(deleted)) ** 2)
    )
    assert diffusion["standard_error"] == pytest.approx(expected_se, rel=1e-12)
    assert "particle" in diffusion["method"]
    assert "bias" in diffusion["uncertainty_scope"]


def test_sampling_more_correlated_frames_does_not_create_independent_particles():
    results = []
    for stride in (1, 2):
        configuration = parameters(sample_stride=stride)
        coordinates, steps = linear_covariance_trajectory(configuration, drift=(2.0, -1.0))
        results.append(analyze_transport(coordinates, steps, parameters=configuration))
    assert results[0]["diffusion"]["sample_count"] > results[1]["diffusion"]["sample_count"]
    assert results[0]["diffusion"]["estimate"] == pytest.approx(results[1]["diffusion"]["estimate"])
    assert results[0]["diffusion"]["standard_error"] == pytest.approx(
        results[1]["diffusion"]["standard_error"], rel=1e-12
    )


def test_short_and_explicit_early_windows_do_not_claim_a_diffusion_measurement():
    short = parameters(num_steps=4)
    coordinates, steps = linear_covariance_trajectory(short)
    result = analyze_transport(coordinates, steps, parameters=short)
    assert result["diffusion"]["status"] == "unavailable"
    assert result["diffusion"]["estimate"] is None
    assert result["diffusion"]["theoretical"] == 1.0
    early = parameters(diffusion_fit_start_step=1)
    coordinates, steps = linear_covariance_trajectory(early)
    result = analyze_transport(coordinates, steps, parameters=early)
    assert result["diffusion"]["status"] == "unavailable"
    assert result["diffusion"]["reason"] == "initial_transient_not_relaxed"


def test_hot_initial_state_requires_adaptive_relaxation_and_records_analytic_fit_bias():
    hot = parameters(initial_temperature=1e6, bath_temperature=1e-6, num_steps=100)
    coordinates, steps = linear_covariance_trajectory(hot)
    result = analyze_transport(coordinates, steps, parameters=hot)
    fit = result["diffusion"]
    assert fit["status"] == "available"
    assert fit["minimum_relaxation_times"] > 30
    assert fit["fit_start_step"] >= math.ceil(fit["minimum_relaxation_times"])
    assert abs(fit["relative_finite_window_bias"]) <= 0.01
    assert fit["expected_finite_window_estimate"] == pytest.approx(fit["theoretical"], rel=0.01)
    early = parameters(
        initial_temperature=1e6,
        bath_temperature=1e-6,
        num_steps=100,
        diffusion_fit_start_step=5,
    )
    coordinates, steps = linear_covariance_trajectory(early)
    rejected = analyze_transport(coordinates, steps, parameters=early)["diffusion"]
    assert rejected["status"] == "unavailable"
    assert rejected["reason"] == "initial_transient_not_relaxed"
    assert rejected["standard_error"] is None
    assert rejected["relative_finite_window_bias"] > 0.01
    assert rejected["expected_finite_window_estimate"] > rejected["theoretical"]


def test_centered_theory_matches_equilibrium_ballistic_and_diffusive_expression():
    configuration = parameters(mass=2.0, friction=0.5, acceleration=(0.2, -0.1))
    coordinates, steps = linear_covariance_trajectory(configuration)
    result = analyze_transport(coordinates, steps, parameters=configuration)
    times = np.asarray(result["times"])
    coefficient = configuration.bath_temperature / (configuration.mass * configuration.friction)
    expected_centered = (
        4
        * coefficient
        * (times - (-np.expm1(-configuration.friction * times)) / configuration.friction)
    )
    np.testing.assert_allclose(result["theory"]["centered_msd"], expected_centered, atol=1e-14)
    expected_mean = (
        (times - (-np.expm1(-configuration.friction * times)) / configuration.friction)[:, None]
        * np.asarray(configuration.acceleration)
        / configuration.friction
    )
    np.testing.assert_allclose(result["theory"]["mean_displacement"], expected_mean, atol=1e-14)
    np.testing.assert_allclose(
        result["theory"]["msd"], expected_centered + np.sum(expected_mean**2, axis=1)
    )


def test_simulated_diffusion_is_recovered_with_particle_uncertainty_in_a_periodic_field():
    configuration = parameters(
        num_particles=1000,
        num_steps=80,
        seed=2026,
        acceleration=(0.2, -0.1),
        boundary="periodic",
        box_size=(1.0, 1.0),
    )
    simulator = create_langevin_simulator(configuration)
    steps = np.asarray(configuration.sampling_steps(), dtype=np.int64)
    coordinates = np.empty((len(steps), configuration.num_particles, 2), dtype=np.float64)
    coordinates[0] = simulator.unwrapped_positions
    for index in range(1, len(steps)):
        simulator.advance()
        coordinates[index] = simulator.unwrapped_positions
    result = analyze_transport(coordinates, steps, parameters=configuration)
    fit = result["diffusion"]
    assert fit["status"] == "available"
    # This checks an independently sampled trajectory against known model D;
    # uncertainty uses independent particles, not the eighty time frames.
    assert abs(fit["estimate"] - fit["theoretical"]) < 6 * fit["standard_error"]
    assert 0 < fit["standard_error"] < 0.2
    assert result["centered_msd"][-1] > 100
    wrapped = simulator.positions
    assert np.all((wrapped >= 0) & (wrapped < 1))
    assert np.mean(np.sum(wrapped**2, axis=1)) < 2


@pytest.mark.parametrize("change", ["dtype", "shape", "nan", "schedule", "origin"])
def test_analysis_rejects_states_that_do_not_match_the_declared_experiment(change):
    configuration = parameters()
    coordinates, steps = linear_covariance_trajectory(configuration)
    if change == "dtype":
        coordinates = coordinates.astype(np.float32)
    elif change == "shape":
        coordinates = coordinates[:, :, :1]
    elif change == "nan":
        coordinates[1, 0, 0] = np.nan
    elif change == "schedule":
        steps[1] += 1
    else:
        coordinates[0, 0, 0] += 1
    with pytest.raises(ValueError):
        analyze_transport(coordinates, steps, parameters=configuration)


def save_state(path, configuration, coordinates=None):
    if coordinates is None:
        coordinates, steps = linear_covariance_trajectory(configuration)
    else:
        steps = np.asarray(configuration.sampling_steps(), dtype=np.int64)
    np.savez_compressed(
        path,
        positions=coordinates,
        sample_steps=steps,
        times=steps.astype(np.float64) * configuration.time_step,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path, configuration, digest):
    return load_trajectory_state(
        path,
        expected_sha256=digest,
        expected_particles=configuration.num_particles,
        expected_sample_steps=configuration.sampling_steps(),
        expected_time_step=configuration.time_step,
    )


def test_saved_trajectory_roundtrip_supports_independent_transport_reanalysis(tmp_path):
    configuration = parameters(initial_position=(1.0, -2.0))
    coordinates, steps = linear_covariance_trajectory(configuration)
    path = tmp_path / "trajectory.npz"
    digest = save_state(path, configuration, coordinates)
    loaded = load_state(path, configuration, digest)
    np.testing.assert_array_equal(loaded["positions"], coordinates)
    np.testing.assert_array_equal(loaded["sample_steps"], steps)
    assert analyze_transport(
        loaded["positions"], loaded["sample_steps"], parameters=configuration
    ) == (analyze_transport(coordinates, steps, parameters=configuration))
    with pytest.raises(ValueError, match="SHA256"):
        load_state(path, configuration, "0" * 64)


def test_loader_infers_uniform_time_step_without_demanding_float_roundtrip_identity(tmp_path):
    configuration = parameters(time_step=0.007, num_steps=100, sample_stride=3)
    path = tmp_path / "trajectory.npz"
    digest = save_state(path, configuration)
    loaded = load_trajectory_state(
        path,
        expected_sha256=digest,
        expected_particles=configuration.num_particles,
        expected_sample_steps=configuration.sampling_steps(),
    )
    np.testing.assert_array_equal(
        loaded["times"], np.asarray(configuration.sampling_steps()) * configuration.time_step
    )


@pytest.mark.parametrize("mutation", ["extra", "dtype", "object", "times", "steps", "fortran"])
def test_trajectory_loader_rejects_malformed_archives(tmp_path, mutation):
    configuration = parameters()
    coordinates, steps = linear_covariance_trajectory(configuration)
    values = {
        "positions": coordinates,
        "sample_steps": steps,
        "times": steps.astype(np.float64),
    }
    if mutation == "extra":
        values["extra"] = np.array([0.0])
    elif mutation == "dtype":
        values["positions"] = coordinates.astype(np.float32)
    elif mutation == "object":
        values["positions"] = coordinates.astype(object)
    elif mutation == "times":
        values["times"] = values["times"].copy()
        values["times"][1] += 0.1
    elif mutation == "steps":
        values["sample_steps"] = values["sample_steps"].copy()
        values["sample_steps"][1] += 1
    else:
        values["positions"] = np.asfortranarray(coordinates)
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_state(path, configuration, digest)


def test_forged_header_is_rejected_before_numpy_allocates(tmp_path, monkeypatch):
    configuration = parameters()
    path = tmp_path / "oversized-shape.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in (
            ("sample_steps", np.asarray(configuration.sampling_steps(), dtype=np.int64)),
            ("times", np.asarray(configuration.sampling_steps(), dtype=np.float64)),
        ):
            value = io.BytesIO()
            np.save(value, array, allow_pickle=False)
            archive.writestr(name + ".npy", value.getvalue())
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {"descr": "<f8", "fortran_order": False, "shape": (10**9, 10**9, 2)},
        )
        archive.writestr("positions.npy", header.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Invalid headers must be rejected before np.load")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ValueError, match="header"):
        load_state(path, configuration, digest)


def test_loader_rejects_symlinks_and_oversized_regular_files(tmp_path):
    configuration = parameters()
    path = tmp_path / "source.npz"
    digest = save_state(path, configuration)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_state(link, configuration, digest)
    oversized = tmp_path / "oversized.npz"
    with oversized.open("wb") as stream:
        stream.seek(MAX_ARCHIVE_BYTES)
        stream.write(b"x")
    with pytest.raises(ValueError, match="128 MiB"):
        load_state(oversized, configuration, digest)
