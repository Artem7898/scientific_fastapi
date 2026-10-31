"""Single-trajectory uncertainty uses time dependence, not particle counting."""

import copy
import json
import math

import numpy as np
import pytest

from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.research.lj_analysis import SERIES_NAMES
from scientific_api.research.lj_uncertainty import (
    MIN_SAMPLES,
    _block_means,
    analyze_lj_uncertainty,
    select_lj_production,
)


def parameters(**changes):
    return LennardJonesParameters(
        **{"num_particles": 4, "num_steps": 1000, "sample_stride": 1, **changes}
    )


def artifact(values, configuration=None):
    configuration = parameters() if configuration is None else configuration
    values = np.asarray(values, dtype=np.float64)
    # The source codec validates observable shape/identity, independently of
    # the statistical properties under test. Positive quantities are separate.
    return {
        "model_version": LJ_MODEL_VERSION,
        "sample_steps": list(configuration.sampling_steps()),
        "series": {
            name: (
                np.abs(values) + 1 if name in {"thermal_temperature", "kinetic_energy"} else values
            ).tolist()
            for name in SERIES_NAMES
        },
    }


def ar1(phi, *, count=1001, seed=2026):
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(count, dtype=np.float64)
    values[0] = rng.normal()  # Stationary unit-variance initialization.
    innovation_scale = math.sqrt(1 - phi * phi)
    for index in range(1, count):
        values[index] = phi * values[index - 1] + innovation_scale * rng.normal()
    return values


def analyze(values, configuration=None, **options):
    configuration = parameters() if configuration is None else configuration
    return analyze_lj_uncertainty(
        artifact(values, configuration),
        configuration,
        **{
            "observable_names": ("pressure",),
            "bootstrap_samples": 400,
            "assume_stationary": True,
            **options,
        },
    )


def metric(values, configuration=None, **options):
    return analyze(values, configuration, **options)["metrics"]["pressure"]


def test_iid_mean_sd_iat_and_se_have_known_scaling():
    values = ar1(0)
    report = analyze(values)
    result = report["metrics"]["pressure"]
    assert result["status"] == "available"
    assert result["mean"] == pytest.approx(np.mean(values), rel=1e-13)
    assert result["sample_standard_deviation"] == pytest.approx(np.std(values, ddof=1))
    assert 1 <= result["integrated_autocorrelation_time_samples"] < 1.5
    assert 650 < result["effective_sample_size"] <= len(values)
    assert result["iat_mean_standard_error"] == pytest.approx(
        np.std(values, ddof=1)
        * math.sqrt(result["integrated_autocorrelation_time_samples"] / len(values))
    )
    assert result["correlated_mean_standard_error"] == pytest.approx(1 / math.sqrt(1001), rel=0.2)
    assert result["confidence_interval"]["coverage_guarantee"] is False
    assert result["confidence_interval"]["nominal_level"] == 0.95
    assert report["policy"]["minimum_samples"] == MIN_SAMPLES
    assert "particle" in report["policy"]["effective_sample_size"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("phi", [0.5, 0.7, 0.8])
def test_ar1_correlated_se_and_iat_match_analytic_scale(phi):
    result = metric(ar1(phi))
    theoretical_iat = (1 + phi) / (1 - phi)
    assert result["status"] == "available"
    assert result["integrated_autocorrelation_time_samples"] == pytest.approx(
        theoretical_iat, rel=0.3
    )
    assert result["effective_sample_size"] < 450
    assert result["correlated_mean_standard_error"] > 1.5 / math.sqrt(1001)
    assert result["block_length"] >= math.ceil(
        2 * result["integrated_autocorrelation_time_samples"]
    )
    assert result["diagnostics"]["complete_block_count"] >= 8


def test_default_requires_explicit_stationarity_assumption_and_does_not_bootstrap(monkeypatch):
    from scientific_api.research import lj_uncertainty

    def forbidden(*args, **kwargs):
        raise AssertionError("bootstrap must not run without the assumption")

    monkeypatch.setattr(lj_uncertainty, "_block_means", forbidden)
    report = analyze_lj_uncertainty(artifact(ar1(0)), parameters(), observable_names=("pressure",))
    result = report["metrics"]["pressure"]
    assert result["reason"] == "stationary_assumption_required"
    assert result["confidence_interval"] is None
    assert result["correlated_mean_standard_error"] is None
    assert result["integrated_autocorrelation_time_samples"] >= 1
    assert "no_flags" in result["diagnostics"]["stationarity_screen"]["status"]
    assert any("does not prove" in line for line in report["limitations"])


@pytest.mark.parametrize("value", [0.0, -3.0, 1e-250, 1e250])
def test_constant_series_is_not_given_a_zero_width_confidence_claim(value):
    result = metric(np.full(1001, value))
    assert result["reason"] == "flat_series"
    assert result["mean"] == value
    assert result["sample_standard_deviation"] == 0
    assert result["confidence_interval"] is None
    assert result["integrated_autocorrelation_time_samples"] is None


def test_trend_and_variance_changes_are_explicit_nonstationarity_flags():
    trend = metric(np.linspace(-1, 1, 1001))
    assert trend["reason"] == "nonstationarity_detected"
    assert "linear_drift_detected" in trend["diagnostics"]["stationarity_screen"]["flags"]
    rng = np.random.Generator(np.random.PCG64(43))
    changed = metric(np.concatenate((rng.normal(size=500), 10 * rng.normal(size=501))))
    assert changed["reason"] == "nonstationarity_detected"
    assert (
        "quarter_variance_shift_detected" in changed["diagnostics"]["stationarity_screen"]["flags"]
    )
    assert changed["confidence_interval"] is None


def test_slow_smooth_mode_cannot_be_promoted_to_an_independent_sample_interval():
    result = metric(np.sin(np.linspace(0, 2 * np.pi, 1001)))
    assert result["status"] == "unavailable"
    assert result["reason"] in {
        "nonstationarity_detected",
        "insufficient_effective_samples",
        "insufficient_time_blocks",
    }
    assert result["confidence_interval"] is None


def test_reemergent_slow_positive_lobe_is_not_hidden_by_an_early_negative_pair():
    # Stationary Gaussian sum of damped 90-degree rotations and a slow AR(1).
    # Its analytic infinite-window IAT is ~19.99, while early IPS can be ~1.2.
    rng = np.random.Generator(np.random.PCG64(42))
    rotation = rng.normal(size=2)
    slow = rng.normal()
    values = []
    for _ in range(1001):
        rotation = 0.9 * np.array([-rotation[1], rotation[0]]) + math.sqrt(0.19) * rng.normal(
            size=2
        )
        slow = 0.99 * slow + math.sqrt(1 - 0.99**2) * rng.normal()
        values.append(math.sqrt(0.9) * rotation[0] + math.sqrt(0.1) * slow)
    result = metric(values)
    assert result["reason"] == "unresolved_positive_correlation_tail"
    assert result["diagnostics"]["autocorrelation"]["positive_tail_reemergence"] is True
    assert result["confidence_interval"] is None


def test_irregular_terminal_frame_and_user_burn_in_select_only_saved_uniform_samples():
    configuration = parameters(num_steps=1000, sample_stride=7, equilibration_steps=100)
    values = np.arange(len(configuration.sampling_steps()), dtype=np.float64)
    source = artifact(values, configuration)
    selected = select_lj_production(source, configuration, burn_in_step=102)
    expected_steps = np.arange(105, 995, 7, dtype=np.int64)
    np.testing.assert_array_equal(selected["sample_steps"], expected_steps)
    assert selected["sampling"]["excluded_irregular_terminal_steps"] == [1000]
    assert selected["sampling"]["first_retained_step"] == 105
    assert selected["sampling"]["last_retained_step"] == 994
    assert selected["sampling"]["sample_interval"] == 7 * configuration.time_step
    assert not selected["sample_steps"].flags.writeable
    assert not selected["series"]["pressure"].flags.writeable
    source["series"]["pressure"][15] = -999
    assert selected["series"]["pressure"][0] == 15


def test_burn_in_cannot_shorten_physics_exclusion_and_terminal_is_not_interpolated():
    configuration = parameters(num_steps=1000, sample_stride=7, equilibration_steps=100)
    source = artifact(np.arange(len(configuration.sampling_steps())), configuration)
    with pytest.raises(ValueError, match="burn_in_step"):
        select_lj_production(source, configuration, burn_in_step=99)
    selected = select_lj_production(source, configuration, burn_in_step=999)
    assert len(selected["sample_steps"]) == 0
    result = analyze_lj_uncertainty(source, configuration, burn_in_step=999)["metrics"]["pressure"]
    assert result["reason"] == "insufficient_samples"
    assert result["mean"] is None


def test_minimum_samples_and_eight_blocks_are_hard_gates():
    configuration = parameters(num_steps=126)
    result = metric(ar1(0, count=127), configuration)
    assert result["reason"] == "insufficient_samples"
    result = metric(ar1(0), block_length=126)
    assert result["reason"] == "insufficient_time_blocks"
    assert result["diagnostics"]["complete_block_count"] == 7
    result = metric(ar1(0.8), block_length=2)
    assert result["reason"] == "block_length_too_short_for_correlation"


def test_percentile_resolution_gates_secondary_only_and_leaves_primary_unchanged():
    result = metric(ar1(0), confidence_level=0.99, bootstrap_samples=1000)
    assert result["status"] == "available"
    assert result["confidence_interval"] is not None
    assert result["exploratory_bootstrap_reason"] == "bootstrap_quantile_resolution_insufficient"
    assert result["exploratory_bootstrap_percentile"] is None
    assert result["diagnostics"]["expected_bootstrap_tail_samples"] == pytest.approx(5)
    resolved = metric(ar1(0), confidence_level=0.99, bootstrap_samples=2000)
    assert resolved["confidence_interval"] == result["confidence_interval"]
    assert resolved["correlated_mean_standard_error"] == result["correlated_mean_standard_error"]
    assert resolved["exploratory_bootstrap_percentile"] is not None
    assert (
        metric(ar1(0), confidence_level=0.9, bootstrap_samples=200)[
            "exploratory_bootstrap_percentile"
        ]
        is not None
    )


def test_bootstrap_is_reproducible_seed_isolated_and_does_not_change_the_source():
    source = artifact(ar1(0.5))
    original = copy.deepcopy(source)
    np.random.seed(47)
    global_state = np.random.get_state()
    first = analyze_lj_uncertainty(source, parameters(), assume_stationary=True, bootstrap_seed=99)
    second = analyze_lj_uncertainty(source, parameters(), assume_stationary=True, bootstrap_seed=99)
    after = np.random.get_state()
    assert first == second
    assert source == original
    assert global_state[0] == after[0]
    np.testing.assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]
    changed = analyze_lj_uncertainty(
        source, parameters(), assume_stationary=True, bootstrap_seed=100
    )
    assert (
        first["metrics"]["pressure"]["confidence_interval"]
        == changed["metrics"]["pressure"]["confidence_interval"]
    )
    assert (
        first["metrics"]["pressure"]["exploratory_bootstrap_percentile"]
        != changed["metrics"]["pressure"]["exploratory_bootstrap_percentile"]
    )
    assert first["metrics"]["pressure"]["mean"] == changed["metrics"]["pressure"]["mean"]
    assert (
        first["metrics"]["pressure"]["integrated_autocorrelation_time_samples"]
        == changed["metrics"]["pressure"]["integrated_autocorrelation_time_samples"]
    )


def test_particle_count_never_reduces_one_trajectory_uncertainty():
    first = metric(ar1(0.5), parameters(num_particles=4))
    second = metric(ar1(0.5), parameters(num_particles=64))
    assert first == second


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_extreme_scales_have_the_same_dimensionless_iat_and_relative_uncertainty(scale):
    values = ar1(0.5)
    reference = metric(values)
    result = metric(values * scale)
    assert result["status"] == "available"
    assert result["integrated_autocorrelation_time_samples"] == pytest.approx(
        reference["integrated_autocorrelation_time_samples"], rel=1e-13
    )
    assert result["mean"] / scale == pytest.approx(reference["mean"])
    assert result["correlated_mean_standard_error"] / scale == pytest.approx(
        reference["correlated_mean_standard_error"]
    )
    assert result["confidence_interval"]["lower"] / scale == pytest.approx(
        reference["confidence_interval"]["lower"]
    )
    json.dumps(result, allow_nan=False)


def test_large_offset_uses_local_fluctuations_and_near_roundoff_is_flat():
    result = metric(1e200 + ar1(0.5) * 1e190)
    assert result["status"] == "available"
    assert result["sample_standard_deviation"] == pytest.approx(
        np.std(ar1(0.5), ddof=1) * 1e190, rel=2e-6
    )
    assert result["correlated_mean_standard_error"] < 1e190
    flat = metric(1e200 + ar1(0.5) * 1e185)
    assert flat["reason"] == "flat_series"


def test_unrepresentable_sd_is_unavailable_and_report_remains_finite():
    values = np.tile(np.array([-1e308, 1e308]), 500)
    values = np.append(values, 1e308)
    result = metric(values)
    assert result["status"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_moving_block_bootstrap_matches_explicit_contiguous_blocks_with_truncated_tail():
    values = np.linspace(-2, 3, 129)
    block = 16
    samples = 257  # Exercises two batch boundaries as well as the short final block.
    observed = _block_means(values, block, samples, 77, None)
    rng = np.random.Generator(np.random.PCG64(77))
    expected = []
    for _ in range(samples):
        starts = rng.integers(0, len(values) - block + 1, size=math.ceil(len(values) / block))
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        expected.append(float(np.mean(sample)))
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_bootstrap_endpoint_weighting_bias_matches_all_possible_block_starts():
    values = ar1(0.5)
    block = 37
    result = metric(values, block_length=block)
    assert result["status"] == "available"
    full, remainder = divmod(len(values), block)
    possible_starts = range(len(values) - block + 1)
    expected = (
        full * np.mean([np.sum(values[start : start + block]) for start in possible_starts])
        + np.mean([np.sum(values[start : start + remainder]) for start in possible_starts])
    ) / len(values)
    center = result["diagnostics"]["bootstrap_center"]
    assert center["expected_mean"] == pytest.approx(expected)
    assert center["endpoint_weighting_bias"] == pytest.approx(expected - np.mean(values))


def test_primary_batch_means_t_uses_full_mean_prefix_variance_and_declared_remainder():
    values = ar1(0.5)
    result = metric(values, block_length=37)
    assert result["status"] == "available"
    batches = values[:999].reshape(27, 37).mean(axis=1)
    expected_se = np.std(batches, ddof=1) * math.sqrt(37 / 1001)
    assert result["correlated_mean_standard_error"] == pytest.approx(expected_se)
    interval = result["confidence_interval"]
    assert (interval["upper"] + interval["lower"]) / 2 == pytest.approx(np.mean(values))
    # Independent fixed reference t(.975, df=26); shared table has its own CDF checks.
    assert interval["upper"] - result["mean"] == pytest.approx(2.0555294386428735 * expected_se)
    diagnostics = result["diagnostics"]["batch_means"]
    assert diagnostics["complete_batch_count"] == 27
    assert diagnostics["degrees_of_freedom"] == interval["degrees_of_freedom"] == 26
    assert diagnostics["variance_sample_count"] == 999
    assert diagnostics["variance_excluded_remainder_count"] == 2
    assert diagnostics["mean_sample_count"] == 1001
    assert diagnostics["complete_batches_mean"] == pytest.approx(np.mean(values[:999]))
    assert diagnostics["complete_batches_mean"] != result["mean"]


def test_primary_block_length_policy_requires_eight_iat_and_eight_to_fifty_batches():
    result = metric(ar1(0.5))
    g = result["integrated_autocorrelation_time_samples"]
    assert result["block_length"] == max(
        math.ceil(8 * g), math.ceil(1001 / 50), math.ceil(1001 ** (1 / 3))
    )
    assert 8 <= result["diagnostics"]["complete_block_count"] <= 50
    assert metric(ar1(0.5), block_length=25)["reason"] == "block_length_too_short_for_correlation"
    assert metric(ar1(0), block_length=11)["reason"] == "too_many_time_blocks"


@pytest.mark.parametrize("phi", [0.0, 0.7])
def test_fixed_seed_population_calibration_checks_mean_se_and_block_interval_coverage(phi):
    # Aggregate diagnostics across 80 predetermined stationary Gaussian series,
    # rather than requiring a single random interval to contain the truth.
    means, predicted_variances, covered = [], [], []
    for seed in range(80):
        result = metric(ar1(phi, seed=10000 + seed), bootstrap_seed=seed)
        if result["status"] != "available":
            continue  # Rejections are explicit; do not silently turn them into intervals.
        interval = result["confidence_interval"]
        means.append(result["mean"])
        predicted_variances.append(result["correlated_mean_standard_error"] ** 2)
        covered.append(interval["lower"] <= 0 <= interval["upper"])
    assert len(means) >= 65
    empirical_variance = np.var(means, ddof=1)
    assert 0.65 < np.mean(predicted_variances) / empirical_variance < 1.5
    assert 0.82 < np.mean(covered) <= 1.0


@pytest.mark.parametrize(
    "options",
    [
        {"bootstrap_seed": -1},
        {"bootstrap_seed": True},
        {"bootstrap_seed": 2**64},
        {"bootstrap_samples": 199},
        {"bootstrap_samples": 5001},
        {"confidence_level": 0.8},
        {"confidence_level": True},
        {"assume_stationary": 1},
        {"block_length": 0},
        {"block_length": 1002},
        {"observable_names": ()},
        {"observable_names": ("pressure", "pressure")},
        {"observable_names": ("rdf",)},
        {"observable_names": (["pressure"],)},
    ],
)
def test_invalid_options_reject_before_statistical_work(options):
    with pytest.raises(ValueError):
        analyze_lj_uncertainty(artifact(ar1(0)), parameters(), **options)


@pytest.mark.parametrize(
    "mutation", ["model", "schedule", "extra", "nan", "negative", "object", "shape"]
)
def test_invalid_saved_sources_are_rejected(mutation):
    source = artifact(ar1(0))
    if mutation == "model":
        source["model_version"] = "other"
    elif mutation == "schedule":
        source["sample_steps"][10] = 9
    elif mutation == "extra":
        source["series"]["not_an_observable"] = [0] * 1001
    elif mutation == "nan":
        source["series"]["pressure"][3] = math.nan
    elif mutation == "negative":
        source["series"]["thermal_temperature"][3] = -1
    elif mutation == "object":
        source["series"]["pressure"] = np.array(source["series"]["pressure"], dtype=object)
    else:
        source["sample_steps"] = np.array(5)
    with pytest.raises(ValueError):
        analyze_lj_uncertainty(source, parameters())


def test_cancellation_propagates_during_autocorrelation_and_bootstrap():
    class Cancelled(Exception):
        pass

    for stop in (1, 4, 20):
        calls = 0

        def check():
            nonlocal calls
            calls += 1
            if calls >= stop:
                raise Cancelled

        with pytest.raises(Cancelled):
            analyze(ar1(0), bootstrap_samples=1000, check_cancelled=check)
