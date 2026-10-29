"""LJ diagnostics use two-dimensional geometry and respect correlated frames."""

import numpy as np
import pytest

from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.research.lj_analysis import analyze_lj_observables, radial_distribution


def configuration(**changes):
    return LennardJonesParameters(
        **{
            "num_particles": 4,
            "num_steps": 20,
            "sample_stride": 5,
            "equilibration_steps": 7,
            **changes,
        }
    )


def sampled_data(parameters):
    steps = np.asarray(parameters.sampling_steps(), dtype=np.int64)
    initial = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    positions = np.broadcast_to(initial, (len(steps), *initial.shape)).copy()
    kinetic = np.asarray([3.0, 4.0, 5.0, 4.0, 3.0], dtype=np.float64)
    potential = -kinetic - 3.0
    virial = np.asarray([-1.0, -2.0, -3.0, -2.0, -1.0], dtype=np.float64)
    series = {
        "thermal_temperature": kinetic / 3,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": kinetic + potential,
        "virial": virial,
        "pressure": (2 * kinetic + virial) / (2 * np.prod(parameters.box_size)),
    }
    return positions, steps, series


def test_rdf_counts_unordered_pairs_with_finite_n_two_dimensional_normalization():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    observed = radial_distribution(positions, box_size=(10.0, 10.0), bins=10)
    expected_counts = np.zeros(10, dtype=np.int64)
    expected_counts[[2, 4, 6]] = 1
    np.testing.assert_array_equal(observed["counts"], expected_counts)
    edges = np.asarray(observed["bin_edges"])
    expected_uniform = 3 * np.pi * np.diff(edges**2) / 100
    np.testing.assert_allclose(observed["expected_uniform_counts"], expected_uniform)
    np.testing.assert_allclose(observed["g_r"], expected_counts / expected_uniform)
    assert observed["dimensions"] == 2
    assert observed["pair_checks"] == 3
    assert observed["maximum_radius"] == 5.0


def test_rdf_uses_periodic_minimum_images_and_is_invariant_to_unwrapping():
    positions = np.array([[0.1, 0.2], [9.9, 0.2], [0.1, 9.7]], dtype=np.float64)
    shifted = positions + np.array([[20.0, -10.0], [-30.0, 40.0], [10.0, 20.0]])
    baseline = radial_distribution(positions, box_size=(10.0, 10.0), bins=20)
    unwrapped = radial_distribution(shifted, box_size=(10.0, 10.0), bins=20)
    assert baseline["counts"] == unwrapped["counts"]
    assert sum(baseline["counts"]) == 3
    assert baseline["counts"][0] == 1  # Separation 0.2 across the boundary.


def test_uniform_periodic_particles_have_unit_rdf_without_a_3d_shell_factor():
    generator = np.random.Generator(np.random.PCG64(2026))
    positions = generator.uniform(0.0, 20.0, size=(1000, 2)).astype(np.float64)
    observed = radial_distribution(positions, box_size=(20.0, 20.0), bins=20)
    # Pair counts are descriptive; this broad fixed-seed reference is not an
    # uncertainty estimator that treats all pairs as independent samples.
    np.testing.assert_allclose(observed["g_r"], np.ones(20), rtol=0.15, atol=0.0)
    assert sum(observed["counts"]) / sum(observed["expected_uniform_counts"]) == pytest.approx(
        1.0, rel=0.01
    )


def test_signed_energies_production_window_and_nve_energy_normalization_are_preserved():
    parameters = configuration()
    positions, steps, series = sampled_data(parameters)
    series["total_energy"] = np.asarray([-3.0, -2.9, -2.8, -2.7, -2.6])
    series["potential_energy"] = series["total_energy"] - series["kinetic_energy"]
    observed = analyze_lj_observables(positions, steps, series, parameters=parameters)
    assert observed["series"]["potential_energy"][-1] < 0
    assert observed["series"]["total_energy"][-1] < 0
    production = observed["production"]
    assert production["first_sample_step"] == 10
    assert production["last_sample_step"] == 20
    assert production["sample_count"] == 3
    statistics = production["statistics"]["kinetic_energy"]
    assert statistics["mean"] == 4.0
    assert statistics["sample_standard_deviation"] == 1.0
    assert "standard_error" not in statistics
    drift = observed["energy_drift"]
    assert drift["applicable"] is True
    assert drift["reference_scale"] == 4.0  # max(abs(-3), epsilon*N)
    assert drift["maximum_absolute_change"] == pytest.approx(0.4)
    assert drift["maximum_normalized_absolute_change"] == pytest.approx(0.1)
    assert drift["terminal_normalized_change"] == pytest.approx(0.1)


def test_near_zero_initial_energy_has_a_physical_nonzero_normalization_scale():
    parameters = configuration()
    positions, steps, series = sampled_data(parameters)
    series["total_energy"] = np.arange(5, dtype=np.float64) * 0.01
    series["potential_energy"] = series["total_energy"] - series["kinetic_energy"]
    observed = analyze_lj_observables(positions, steps, series, parameters=parameters)
    assert observed["energy_drift"]["reference_scale"] == 4.0
    assert observed["energy_drift"]["maximum_normalized_absolute_change"] == 0.01


def test_nvt_does_not_claim_total_energy_conservation_or_independent_frame_uncertainty():
    parameters = configuration(ensemble="nvt", bath_temperature=1.0)
    positions, steps, series = sampled_data(parameters)
    observed = analyze_lj_observables(positions, steps, series, parameters=parameters)
    drift = observed["energy_drift"]
    assert drift["applicable"] is False
    assert drift["maximum_normalized_absolute_change"] is None
    assert drift["terminal_change"] is None
    assert "thermostat exchanges energy" in drift["interpretation"]
    assert "standard_error" not in str(observed["production"]["statistics"])
    assert "diffusion" not in observed


def test_single_selected_frame_has_no_fabricated_sample_sd():
    parameters = configuration(equilibration_steps=19)
    positions, steps, series = sampled_data(parameters)
    observed = analyze_lj_observables(positions, steps, series, parameters=parameters)
    assert observed["production"]["sample_count"] == 1
    assert all(
        value["sample_standard_deviation"] is None
        for value in observed["production"]["statistics"].values()
    )


@pytest.mark.parametrize(
    "change", ["steps", "shape", "dtype", "nan", "energy", "pressure", "negative", "keys"]
)
def test_saved_analysis_rejects_inconsistent_inputs(change):
    parameters = configuration()
    positions, steps, series = sampled_data(parameters)
    if change == "steps":
        steps = steps + 1
    elif change == "shape":
        positions = positions[:-1]
    elif change == "dtype":
        series["total_energy"] = series["total_energy"].astype(np.float32)
    elif change == "nan":
        positions[0, 0, 0] = np.nan
    elif change == "energy":
        series["total_energy"] = series["total_energy"] + 1
    elif change == "pressure":
        series["pressure"] = series["pressure"] + 1
    elif change == "negative":
        series["kinetic_energy"][0] = -1
    else:
        del series["virial"]
    with pytest.raises(ValueError):
        analyze_lj_observables(positions, steps, series, parameters=parameters)


@pytest.mark.parametrize("bins", [0, 7, 257, True, 8.0])
def test_rdf_rejects_invalid_bin_contracts(bins):
    with pytest.raises(ValueError):
        radial_distribution(np.zeros((3, 2), dtype=np.float64), box_size=(10.0, 10.0), bins=bins)


def test_rdf_rejects_nonfinite_positions_and_invalid_geometry():
    with pytest.raises(ValueError):
        radial_distribution(np.full((3, 2), np.inf), box_size=(10.0, 10.0))
    with pytest.raises(ValueError):
        radial_distribution(np.zeros((3, 2)), box_size=(0.0, 10.0))
