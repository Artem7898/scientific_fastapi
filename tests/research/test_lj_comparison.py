"""Whole-replica units, matched sensitivity and finite-window reporting."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import SimulationCancelled
from scientific_api.research.lj_analysis import SERIES_NAMES
from scientific_api.research.lj_comparison import (
    LJReplicaSource,
    _student_t_critical,
    analyze_lj_replicas,
    analyze_lj_sensitivity,
)
from scientific_api.research.student_t import student_t_critical


def replicas(
    count=8,
    *,
    prefix="base",
    shift=0.0,
    slope=1.0,
    num_steps=20,
    time_step=0.002,
    sample_stride=5,
    equilibration_steps=5,
    **parameter_changes,
):
    result = []
    for index in range(count):
        parameters = LennardJonesParameters(
            num_particles=16,
            seed=index,
            num_steps=num_steps,
            time_step=time_step,
            sample_stride=sample_stride,
            equilibration_steps=equilibration_steps,
            **parameter_changes,
        )
        steps = np.asarray(parameters.sampling_steps(), dtype=np.int64)
        # Synthetic physically signed observables with exactly known means;
        # independent replicas differ, while their saved time frames correlate.
        kinetic = np.full(len(steps), 20.0 + shift + slope * index, dtype=np.float64)
        potential = np.full(len(steps), -10.0 + shift + slope * index, dtype=np.float64)
        virial = np.full(len(steps), -2.0 + shift + slope * index, dtype=np.float64)
        series = {
            "thermal_temperature": kinetic / 15,
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "total_energy": kinetic + potential,
            "pressure": (2 * kinetic + virial) / (2 * 12 * 12),
            "virial": virial,
        }
        result.append(
            LJReplicaSource(
                source_id=f"{prefix}-{index}",
                attempt_id=f"{prefix}-owner-{index}",
                status="completed",
                parameters=parameters,
                sample_steps=steps,
                series=series,
            )
        )
    return result


def analyze(sources, **kwargs):
    return analyze_lj_replicas(sources, bootstrap_samples=500, **kwargs)


def test_whole_replica_means_have_analytic_sample_se_and_student_t_interval():
    sources = replicas()
    before = copy.deepcopy(sources)
    report = analyze(sources, observable_names=("kinetic_energy",))
    statistic = report["statistics"]["kinetic_energy"]
    # The eight independent means are 20..27, not 32 independent time frames.
    assert statistic["replica_means"] == list(range(20, 28))
    assert statistic["mean"] == 23.5
    assert statistic["sample_standard_deviation"] == pytest.approx(math.sqrt(6))
    assert statistic["standard_error"] == pytest.approx(math.sqrt(6 / 8))
    interval = statistic["confidence_interval"]
    half_width = 2.364624251592784 * math.sqrt(6 / 8)
    assert interval["lower"] == pytest.approx(23.5 - half_width)
    assert interval["upper"] == pytest.approx(23.5 + half_width)
    assert interval["degrees_of_freedom"] == 7
    assert not interval["coverage_guarantee"]
    assert "approximately normal" in interval["assumption"]
    assert "Exploratory" in statistic["exploratory_bootstrap_percentile"]["method"]
    assert report["independent_unit"] == "whole logical seeded replica"
    assert report["failure_policy"] == "require_all_requested"
    json.dumps(report, allow_nan=False)
    for source, original in zip(sources, before, strict=True):
        assert source.parameters == original.parameters
        np.testing.assert_array_equal(source.sample_steps, original.sample_steps)
        for name in SERIES_NAMES:
            np.testing.assert_array_equal(source.series[name], original.series[name])


def test_repeated_correlated_frames_do_not_shrink_across_replica_uncertainty():
    short = analyze(replicas(), observable_names=("kinetic_energy",))
    longer = analyze(replicas(num_steps=200), observable_names=("kinetic_energy",))
    assert short["statistics"] == longer["statistics"]
    assert short["sampling"]["retained_samples"] < longer["sampling"]["retained_samples"]


@pytest.mark.parametrize("count", [1, 2, 7, 8, 50])
def test_conservative_interval_minimum_does_not_fabricate_small_sample_ci(count):
    report = analyze(replicas(count), observable_names=("kinetic_energy",))
    statistic = report["statistics"]["kinetic_energy"]
    assert (statistic["confidence_interval"] is not None) == (count >= 8)
    assert (statistic["standard_error"] is not None) == (count >= 2)
    assert (statistic["exploratory_bootstrap_percentile"] is not None) == (count >= 8)


def test_identical_replica_means_are_unresolved_not_zero_width_confidence():
    statistic = analyze(replicas(slope=0.0))["statistics"]["kinetic_energy"]
    assert statistic["sample_standard_deviation"] == 0.0
    assert statistic["standard_error"] == 0.0
    assert statistic["confidence_interval"] is None
    assert statistic["exploratory_bootstrap_percentile"] is None
    assert statistic["interval_reason"] == "unresolved_variation"


@pytest.mark.parametrize("level,boundary", [(0.9, 200), (0.95, 400), (0.99, 2000)])
def test_secondary_bootstrap_tail_resolution_never_changes_primary_student_t_ci(level, boundary):
    unresolved = analyze_lj_replicas(
        replicas(),
        confidence_level=level,
        bootstrap_samples=boundary - 1,
        observable_names=("kinetic_energy",),
    )
    resolved = analyze_lj_replicas(
        replicas(),
        confidence_level=level,
        bootstrap_samples=boundary,
        observable_names=("kinetic_energy",),
    )
    first = unresolved["statistics"]["kinetic_energy"]
    second = resolved["statistics"]["kinetic_energy"]
    assert first["confidence_interval"] == second["confidence_interval"]
    assert first["standard_error"] == second["standard_error"]
    assert first["confidence_interval"] is not None
    assert first["exploratory_bootstrap_percentile"] is None
    assert first["exploratory_bootstrap_reason"] == "bootstrap_quantile_resolution_insufficient"
    assert first["bootstrap_expected_tail_samples"] < 10
    assert unresolved["bootstrap"]["expected_tail_samples"] < 10
    assert second["bootstrap_expected_tail_samples"] == 10
    assert second["exploratory_bootstrap_percentile"] is not None
    assert second["exploratory_bootstrap_reason"] == "exploratory_percentile_bootstrap"


def test_unresolved_secondary_tail_quantiles_skip_random_resampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Suppressed exploratory intervals must not resample replicas")

    monkeypatch.setattr(np.random, "Generator", forbidden)
    report = analyze_lj_replicas(
        replicas(), confidence_level=0.99, bootstrap_samples=200, observable_names=("pressure",)
    )
    statistic = report["statistics"]["pressure"]
    assert statistic["confidence_interval"] is not None
    assert statistic["exploratory_bootstrap_percentile"] is None
    assert statistic["bootstrap_expected_tail_samples"] == 1


def test_paired_sensitivity_t_interval_remains_available_when_bootstrap_tails_are_sparse():
    report = analyze_lj_sensitivity(
        [replicas(), replicas(prefix="cutoff", cutoff_radius=3.0, slope=2.0)],
        parameter="cutoff_radius",
        confidence_level=0.99,
        bootstrap_samples=200,
        observable_names=("kinetic_energy",),
    )
    statistic = report["contrasts"][0]["statistics"]["kinetic_energy"]
    assert statistic["confidence_interval"] is not None
    assert statistic["exploratory_bootstrap_percentile"] is None
    assert statistic["exploratory_bootstrap_reason"] == "bootstrap_quantile_resolution_insufficient"


@pytest.mark.parametrize(
    "df,level,expected",
    [
        (7, 0.9, 1.8945786050900062),
        (7, 0.95, 2.364624251592784),
        (7, 0.99, 3.4994832973504924),
        (15, 0.95, 2.131449545559776),
        (31, 0.99, 2.744041919294269),
        (49, 0.9, 1.6765508926168535),
        (49, 0.95, 2.0095752371292392),
        (49, 0.99, 2.679951973631552),
    ],
)
def test_reviewed_student_t_quantiles_match_independent_inverse_cdf(df, level, expected):
    assert _student_t_critical(level, df) == pytest.approx(expected, rel=2e-15)
    assert student_t_critical(level, df) == pytest.approx(expected, rel=2e-15)


def test_shared_student_t_helper_preserves_comparison_lookup_exactly():
    assert student_t_critical is _student_t_critical
    for df in range(7, 50):
        for level in (0.9, 0.95, 0.99):
            assert student_t_critical(level, df) == _student_t_critical(level, df)


@pytest.mark.parametrize(
    "level,df", [(0.95, 6), (0.95, 50), (0.95, 7.0), (0.95, True), (0.8, 7), (float("nan"), 7)]
)
def test_shared_student_t_policy_rejects_unreviewed_quantiles(level, df):
    with pytest.raises(ValueError, match="Unsupported Student-t"):
        student_t_critical(level, df)


def test_all_quantiles_have_expected_tail_and_df_order():
    for df in range(7, 50):
        quantiles = [_student_t_critical(level, df) for level in (0.9, 0.95, 0.99)]
        assert 1.64 < quantiles[0] < quantiles[1] < quantiles[2] < 3.51
    for confidence in (0.9, 0.95, 0.99):
        assert all(
            _student_t_critical(confidence, df) > _student_t_critical(confidence, df + 1)
            for df in range(7, 49)
        )


def test_selected_regular_production_window_excludes_pre_burn_in_and_short_terminal():
    sources = replicas(num_steps=22)
    for source in sources:
        source.series["kinetic_energy"][[0, 1, -1]] = 1e6
    report = analyze(sources, burn_in_time=0.02, observable_names=("kinetic_energy",))
    assert report["sampling"]["sample_steps"] == [10, 15, 20]
    assert report["sampling"]["dropped_irregular_terminal"]
    assert report["statistics"]["kinetic_energy"]["mean"] == 23.5


def test_burn_in_never_reintroduces_archived_excluded_samples():
    report = analyze(replicas(equilibration_steps=10), burn_in_time=0.0)
    assert report["sampling"]["sample_steps"] == [10, 15, 20]


def test_bootstrap_and_report_are_reproducible_and_source_order_independent():
    sources = replicas()
    first = analyze(sources, bootstrap_seed=2**64 - 1)
    assert first == analyze(list(reversed(sources)), bootstrap_seed=2**64 - 1)
    second = analyze(sources, bootstrap_seed=91)
    assert (
        first["statistics"]["kinetic_energy"]["confidence_interval"]
        == second["statistics"]["kinetic_energy"]["confidence_interval"]
    )
    assert (
        first["statistics"]["kinetic_energy"]["exploratory_bootstrap_percentile"]
        != second["statistics"]["kinetic_energy"]["exploratory_bootstrap_percentile"]
    )


@pytest.mark.parametrize("status", ["pending", "running", "failed", "cancelled"])
def test_incomplete_requested_replica_is_rejected_without_survivor_aggregation(status):
    sources = replicas()
    sources[-1] = replace(sources[-1], status=status)
    with pytest.raises(ValueError, match="All requested replicas"):
        analyze(sources)


@pytest.mark.parametrize("duplicate", ["source", "attempt", "seed"])
def test_execution_attempts_and_repeated_seeds_do_not_create_independent_replicas(duplicate):
    sources = replicas()
    changes = {
        "source": {"source_id": sources[0].source_id},
        "attempt": {"attempt_id": sources[0].attempt_id},
        "seed": {"parameters": replace(sources[1].parameters, seed=0)},
    }[duplicate]
    sources[1] = replace(sources[1], **changes)
    with pytest.raises(ValueError, match="[Dd]uplicate|seeds must be distinct"):
        analyze(sources)


@pytest.mark.parametrize(
    "change",
    [
        {"epsilon": 1.1},
        {"backend": "numba"},
        {"interaction_backend": "direct"},
        {"sample_stride": 4},
    ],
)
def test_heterogeneous_physics_backend_or_sampling_does_not_share_replica_ci(change):
    sources = replicas()
    sources[-1] = replace(sources[-1], parameters=replace(sources[-1].parameters, **change))
    with pytest.raises(ValueError, match="protocol must match"):
        analyze(sources)


@pytest.mark.parametrize("bad", ["steps", "dtype", "nonfinite", "missing", "negative"])
def test_invalid_saved_arrays_are_rejected(bad):
    sources = replicas()
    if bad == "steps":
        sources[0] = replace(sources[0], sample_steps=sources[0].sample_steps[:-1])
    elif bad == "dtype":
        sources[0].series["pressure"] = sources[0].series["pressure"].astype(np.float32)
    elif bad == "nonfinite":
        sources[0].series["virial"][0] = np.nan
    elif bad == "missing":
        del sources[0].series["virial"]
    else:
        sources[0].series["thermal_temperature"][0] = -1.0
    with pytest.raises(ValueError):
        analyze(sources)


@pytest.mark.parametrize(
    "options",
    [
        {"burn_in_time": -1.0},
        {"burn_in_time": float("nan")},
        {"burn_in_time": True},
        {"burn_in_time": 1.0},
        {"bootstrap_seed": True},
        {"bootstrap_seed": 2**64},
        {"bootstrap_samples": 99},
        {"bootstrap_samples": 10_001},
        {"confidence_level": 0.8},
        {"confidence_level": True},
        {"observable_names": ()},
        {"observable_names": ("rdf",)},
        {"observable_names": ("pressure", "pressure")},
    ],
)
def test_options_are_bounded_and_supported(options):
    with pytest.raises(ValueError):
        analyze_lj_replicas(replicas(), **options)


def test_cancellation_propagates_during_chunked_bootstrap():
    calls = 0

    def check_cancelled():
        nonlocal calls
        calls += 1
        if calls == 11:
            raise SimulationCancelled("stop bootstrap")

    with pytest.raises(SimulationCancelled, match="stop bootstrap"):
        analyze_lj_replicas(replicas(), check_cancelled=check_cancelled)
    assert calls == 11


def test_matched_cutoff_sensitivity_reports_paired_seed_differences():
    baseline = replicas()
    comparison = replicas(prefix="cutoff", cutoff_radius=3.0, shift=2.0, slope=2.0)
    report = analyze_lj_sensitivity(
        [baseline, list(reversed(comparison))],
        parameter="cutoff_radius",
        bootstrap_samples=100,
        observable_names=("kinetic_energy",),
    )
    assert report["parameter_values"] == [2.5, 3.0]
    assert report["matched_seeds"] == list(range(8))
    assert len(report["groups"]) == 2
    contrast = report["contrasts"][0]["statistics"]["kinetic_energy"]
    assert contrast["replica_means"] == list(range(2, 10))
    assert contrast["mean"] == 5.5
    assert contrast["standard_error"] == pytest.approx(math.sqrt(6 / 8))
    assert contrast["confidence_interval"]["degrees_of_freedom"] == 7
    assert "no true convergence" in report["interpretation"]
    assert any("changes the physical model" in text for text in report["limitations"])
    json.dumps(report, allow_nan=False)


def test_time_step_sensitivity_matches_physical_sampling_window_and_duration():
    baseline = replicas()
    half_step = replicas(
        prefix="half", time_step=0.001, num_steps=40, sample_stride=10, equilibration_steps=10
    )
    report = analyze_lj_sensitivity(
        [baseline, half_step], parameter="time_step", bootstrap_samples=100
    )
    assert (
        report["groups"][0]["sampling"]["sample_times"]
        == report["groups"][1]["sampling"]["sample_times"]
    )
    for statistic in report["contrasts"][0]["statistics"].values():
        assert statistic["mean"] == 0.0
        assert statistic["confidence_interval"] is None


@pytest.mark.parametrize(
    "field", ["sample_stride", "equilibration_steps", "num_steps", "mass", "backend"]
)
def test_time_step_sensitivity_rejects_unmatched_physical_protocols(field):
    baseline = replicas()
    changes = {
        "sample_stride": 5,
        "equilibration_steps": 5,
        "num_steps": 41,
        "mass": 2.0,
        "backend": "numba",
    }
    arguments = dict(
        prefix="half", time_step=0.001, num_steps=40, sample_stride=10, equilibration_steps=10
    )
    arguments[field] = changes[field]
    half_step = replicas(**arguments)
    with pytest.raises(ValueError, match="physical|vary only"):
        analyze_lj_sensitivity([baseline, half_step], parameter="time_step", bootstrap_samples=100)


@pytest.mark.parametrize(
    "bad",
    [
        "unmatched_seed",
        "same_parameter",
        "shared_source",
        "shared_attempt",
        "changed_physics",
        "incomplete",
    ],
)
def test_sensitivity_requires_complete_distinct_matched_physical_seed_groups(bad):
    first = replicas()
    second = replicas(prefix="second", cutoff_radius=3.0)
    if bad == "unmatched_seed":
        second[-1] = replace(second[-1], parameters=replace(second[-1].parameters, seed=99))
    elif bad == "same_parameter":
        second = replicas(prefix="second")
    elif bad == "shared_source":
        second[0] = replace(second[0], source_id=first[0].source_id)
    elif bad == "shared_attempt":
        second[0] = replace(second[0], attempt_id=first[0].attempt_id)
    elif bad == "changed_physics":
        second = replicas(prefix="second", cutoff_radius=3.0, epsilon=2.0)
    else:
        second[-1] = replace(second[-1], status="failed")
    with pytest.raises(ValueError):
        analyze_lj_sensitivity([first, second], parameter="cutoff_radius", bootstrap_samples=100)


def test_sensitivity_groups_preserve_reference_order_and_bound_group_count():
    groups = [
        replicas(prefix="large", cutoff_radius=3.0),
        replicas(prefix="small", cutoff_radius=2.5),
        replicas(prefix="middle", cutoff_radius=2.8),
    ]
    report = analyze_lj_sensitivity(groups, parameter="cutoff_radius", bootstrap_samples=100)
    assert report["parameter_values"] == [3.0, 2.5, 2.8]
    assert [value["reference_parameter_value"] for value in report["contrasts"]] == [3.0, 3.0]
    for invalid in ([], groups[:1], groups * 3):
        with pytest.raises(ValueError, match="2..8"):
            analyze_lj_sensitivity(invalid, parameter="cutoff_radius", bootstrap_samples=100)


def test_real_saved_lj_series_reproduce_explicit_whole_run_mean_estimator(tmp_path: Path):
    from scientific_api.services.simulation_service import run_simulation

    sources = []
    expected = []
    for seed in range(8):
        parameters = LennardJonesParameters(
            num_particles=16,
            num_steps=10,
            sample_stride=2,
            equilibration_steps=2,
            seed=seed,
            initial_temperature=0.1,
        )
        result = run_simulation(
            parameters,
            task_id=f"source-{seed}",
            attempt_id=f"owner-{seed}",
            artifacts_dir=tmp_path,
        )
        observed = json.loads((tmp_path / result["artifacts"]["observables"]).read_bytes())
        series = {
            name: np.asarray(observed["series"][name], dtype=np.float64) for name in SERIES_NAMES
        }
        sources.append(
            LJReplicaSource(
                source_id=f"source-{seed}",
                attempt_id=f"owner-{seed}",
                status="completed",
                parameters=parameters,
                sample_steps=np.asarray(observed["sample_steps"], dtype=np.int64),
                series=series,
            )
        )
        expected.append(float(np.mean(series["pressure"][1:])))
    report = analyze(sources)
    pressure = report["statistics"]["pressure"]
    assert pressure["replica_means"] == pytest.approx(expected, rel=2e-15)
    assert pressure["mean"] == pytest.approx(float(np.mean(expected)), rel=2e-15)
    assert pressure["standard_error"] == pytest.approx(
        float(np.std(expected, ddof=1) / math.sqrt(8)), rel=2e-13
    )
    assert report["sampling"]["sample_steps"] == [2, 4, 6, 8, 10]
