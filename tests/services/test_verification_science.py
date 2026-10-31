"""Independent observables checks, bounded parsing and honest partial coverage."""

import io
import json
import math
import zipfile

import numpy as np
import pytest

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.domain.simulation import (
    MODEL_VERSION,
    SimulationCancelled,
    SimulationParameters,
)
from scientific_api.services import verification_science as science
from scientific_api.services.simulation_service import run_simulation


def configuration(model="ou", backend="numpy", ensemble="nve", interaction="verlet"):
    if model == "ou":
        return SimulationParameters(1.2, 32, num_steps=4, seed=2**64 - 1, backend=backend)
    if model == "langevin":
        return LangevinParameters(
            1.7,
            0.8,
            24,
            time_step=0.1,
            num_steps=100,
            sample_stride=5,
            seed=2**64 - 1,
            backend=backend,
            acceleration=(0.4, -0.1),
            boundary="periodic",
            box_size=(6.0, 8.0),
            initial_position=(5.9, 0.01),
        )
    return LennardJonesParameters(
        num_particles=16,
        box_size=(6.0, 6.0),
        initial_temperature=0.1,
        num_steps=12,
        sample_stride=3,
        seed=42,
        backend=backend,
        ensemble=ensemble,
        interaction_backend=interaction,
        bath_temperature=0.4 if ensemble == "nvt" else None,
    )


def saved(root, parameters):
    result = run_simulation(parameters, task_id="task", attempt_id="owner", artifacts_dir=root)
    artifacts = {name: (root / path).read_bytes() for name, path in result["artifacts"].items()}
    return result, artifacts


def npz(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def checkmap(checks):
    assert len({item["name"] for item in checks}) == len(checks)
    json.dumps(checks, allow_nan=False)
    return {item["name"]: item for item in checks}


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("model", ["ou", "langevin", "lj"])
def test_nominal_saved_results_pass_the_declared_independent_checks(tmp_path, backend, model):
    if backend == "numba":
        pytest.importorskip("numba")
    parameters = configuration(model, backend)
    result, artifacts = saved(tmp_path, parameters)
    checks = science.verify_scientific_result(parameters, result, artifacts)
    assert checks and all(item["status"] == "passed" for item in checks), checks
    checked = checkmap(checks)
    assert checked["terminal_kinetic_energy"]["details"]["rtol"] == science.NUMERICAL_RTOL
    if model == "langevin":
        assert checked["diffusion_particle_jackknife_error"]["status"] == "passed"
    if model == "lj":
        assert checked["terminal_potential_reference"]["status"] == "passed"
        assert checked["rdf_pair_counts"]["status"] == "passed"


@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
@pytest.mark.parametrize("interaction", ["direct", "verlet"])
def test_lj_reference_checks_both_integrators_without_assuming_exact_energy_conservation(
    tmp_path, ensemble, interaction
):
    parameters = configuration("lj", ensemble=ensemble, interaction=interaction)
    result, artifacts = saved(tmp_path, parameters)
    checks = science.verify_scientific_result(parameters, result, artifacts)
    assert all(item["status"] == "passed" for item in checks), checks
    assert all("conservation_tolerance" not in item["name"] for item in checks)


def test_ou_known_snapshot_uses_independent_energy_and_does_not_assert_equilibrium():
    parameters = SimulationParameters(2.0, 2, num_steps=1)
    result = {
        "task_id": "manual",
        "attempt_id": "known",
        "status": "completed",
        "model_version": MODEL_VERSION,
        "seed": 0,
        "num_steps": 1,
        "total_energy": 15.0,
        "final_temperature": 7.5,
        "temperature_standard_error": math.sqrt(2.0),
    }
    manifest = {
        "model": {"version": MODEL_VERSION},
        "outputs": result,
        "parameters": {
            "initial_temperature": 2.0,
            "num_particles": 2,
            "time_step": 0.001,
            "num_steps": 1,
            "seed": 0,
            "backend": "numpy",
        },
    }
    artifacts = {
        "manifest": json.dumps(manifest).encode(),
        "final_state": npz(velocities=np.array([[1.0, 2.0], [3.0, 4.0]])),
    }
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert all(item["status"] == "passed" for item in checked.values())
    assert checked["terminal_kinetic_energy"]["expected"] == 15.0
    assert checked["terminal_temperature"]["expected"] == 7.5
    assert "no equilibrium" in checked["declared_snapshot_standard_error"]["details"]["scope"]


def test_independent_lj_pair_reference_matches_known_shifted_potential_and_2d_histogram():
    parameters = LennardJonesParameters(
        num_particles=3, box_size=(8.0, 8.0), num_steps=1, rdf_bins=8
    )
    positions = np.array([[1.0, 1.0], [2.5, 1.0], [5.0, 5.0]])
    potential, virial, edges, counts, uniform = science._lj_pair_reference(
        parameters, positions, None
    )
    assert potential == pytest.approx(-0.2650202256897745, rel=1e-13)
    assert virial == pytest.approx(-1.678544030390033, rel=1e-13)
    assert counts.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    np.testing.assert_allclose(uniform, 3 * np.pi * np.diff(edges**2) / 64, rtol=1e-15)


def test_langevin_jackknife_matches_explicit_whole_particle_deletions():
    parameters = LangevinParameters(1.0, 1.0, 3, time_step=1.0, num_steps=30, sample_stride=10)
    times = np.array([0.0, 10.0, 20.0, 30.0])
    base = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    positions = times[:, None, None] * base
    centered = np.sum((positions - np.mean(positions, axis=1, keepdims=True)) ** 2, axis=(1, 2)) / 2
    slopes = []
    for removed in range(3):
        remaining = np.delete(positions, removed, axis=1)
        residual = remaining - np.mean(remaining, axis=1, keepdims=True)
        msd = np.sum(residual**2, axis=(1, 2))
        slopes.append(float(np.polyfit(times[1:], msd[1:], 1)[0]) / 4)
    estimate = float(np.polyfit(times[1:], centered[1:], 1)[0]) / 4
    intercept = float(np.polyfit(times[1:], centered[1:], 1)[1])
    error = math.sqrt(2 / 3 * float(np.sum((slopes - np.mean(slopes)) ** 2)))
    theory = 4 * (times[1:] - (1 - np.exp(-times[1:])))
    theoretical_window = float(np.polyfit(times[1:], theory, 1)[0]) / 4
    recorded = {
        "status": "available",
        "reason": None,
        "theoretical": 1.0,
        "estimate": estimate,
        "intercept": intercept,
        "standard_error": error,
        "relative_finite_window_bias": theoretical_window - 1,
        "expected_finite_window_estimate": theoretical_window,
        "fit_start_step": 10,
        "fit_end_step": 30,
        "sample_count": 3,
    }
    checks = science._diffusion_checks(parameters, positions, times, centered, recorded)
    assert all(item["status"] == "passed" for item in checks), checks
    assert checkmap(checks)["diffusion_particle_jackknife_error"]["expected"] == pytest.approx(
        error
    )


@pytest.mark.parametrize(
    "model,field",
    [
        ("ou", "total_energy"),
        ("ou", "temperature_standard_error"),
        ("langevin", "final_centered_msd"),
        ("langevin", "thermal_temperature"),
        ("lj", "potential_energy"),
        ("lj", "pressure"),
    ],
)
def test_coherently_rehashed_wrong_numbers_fail_independent_checks(tmp_path, model, field):
    parameters = configuration(model)
    result, artifacts = saved(tmp_path, parameters)
    result[field] += 0.1 * max(1.0, abs(result[field]))
    manifest = json.loads(artifacts["manifest"])
    manifest["outputs"] = result
    artifacts["manifest"] = json.dumps(manifest).encode()
    checks = science.verify_scientific_result(parameters, result, artifacts)
    assert any(
        item["status"] == "failed" and item["name"] != "source_scientific_contract"
        for item in checks
    ), checks


@pytest.mark.parametrize(
    "target", ["rdf", "production", "series", "transport_msd", "diffusion_error"]
)
def test_saved_research_report_corruption_is_detected(tmp_path, target):
    parameters = configuration(
        "langevin" if target.startswith(("transport", "diffusion")) else "lj"
    )
    result, artifacts = saved(tmp_path, parameters)
    name = "transport" if isinstance(parameters, LangevinParameters) else "observables"
    report = json.loads(artifacts[name])
    if target == "rdf":
        report["rdf"]["counts"][0] += 1
    elif target == "production":
        report["production"]["statistics"]["total_energy"]["mean"] += 1
    elif target == "series":
        report["series"]["pressure"][0] += 1
    elif target == "transport_msd":
        report["msd"][-1] += 1
    else:
        report["diffusion"]["standard_error"] += 1
        result["diffusion"]["standard_error"] += 1
        manifest = json.loads(artifacts["manifest"])
        manifest["outputs"] = result
        artifacts["manifest"] = json.dumps(manifest).encode()
    artifacts[name] = json.dumps(report).encode()
    assert any(
        item["status"] == "failed"
        for item in science.verify_scientific_result(parameters, result, artifacts)
    )


def test_large_pair_work_is_explicitly_unsupported_not_passed(tmp_path, monkeypatch):
    parameters = configuration("lj")
    result, artifacts = saved(tmp_path, parameters)
    monkeypatch.setattr(science, "MAX_PAIR_CHECKS", 1)
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert checked["terminal_pair_reference"]["status"] == "unsupported"
    assert checked["terminal_pair_reference"]["reason"] == "INDEPENDENT_PAIR_BUDGET_EXCEEDED"
    assert "terminal_potential_reference" not in checked


def test_production_analysis_and_force_functions_are_not_used(tmp_path, monkeypatch):
    from scientific_api.models import langevin_engine, lj_forces, physics_engine
    from scientific_api.research import lj_analysis, transport_analysis

    saved_cases = [
        (parameters, *saved(tmp_path / str(index), parameters))
        for index, parameters in enumerate(
            (configuration(), configuration("langevin"), configuration("lj"))
        )
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("Independent verifier called a production scientific routine")

    for module, name in (
        (lj_forces, "compute_forces"),
        (langevin_engine, "langevin_moments"),
        (lj_analysis, "analyze_lj_observables"),
        (lj_analysis, "radial_distribution"),
        (transport_analysis, "analyze_transport"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(physics_engine.IdealGasSimulator, "observables", forbidden)
    for parameters, result, artifacts in saved_cases:
        assert all(
            item["status"] == "passed"
            for item in science.verify_scientific_result(parameters, result, artifacts)
        )


@pytest.mark.parametrize(
    "fault", ["object", "fortran", "huge_header", "duplicate", "nested", "compressed", "nonfinite"]
)
def test_decoder_rejects_invalid_members_before_or_after_bounded_decode(monkeypatch, fault):
    values = np.arange(16, dtype=np.float64).reshape(8, 2)
    name = "velocities.npy"
    payload = io.BytesIO()
    if fault == "object":
        np.save(payload, values.astype(object), allow_pickle=True)
    elif fault == "fortran":
        np.save(payload, np.asfortranarray(values))
    elif fault == "huge_header":
        payload.write(
            np.lib.format.magic(2, 0) + (2**31).to_bytes(4, "little") + b"\0" * values.nbytes
        )
    else:
        if fault == "nonfinite":
            values[0, 0] = np.nan
        np.save(payload, values)
    packed = io.BytesIO()
    with zipfile.ZipFile(
        packed,
        "w",
        compression=zipfile.ZIP_BZIP2 if fault == "compressed" else zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("../" + name if fault == "nested" else name, payload.getvalue())
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(name, payload.getvalue())
    if fault != "nonfinite":
        monkeypatch.setattr(
            science.np,
            "load",
            lambda *a, **k: pytest.fail("Invalid header reached NumPy allocation"),
        )
    with pytest.raises(science.VerificationArtifactError):
        science.decode_verification_npz(
            packed.getvalue(), {"velocities": ((8, 2), np.dtype(np.float64))}
        )


def test_oversized_header_is_not_read_using_its_advertised_length(monkeypatch):
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        archive.writestr(
            "velocities.npy",
            np.lib.format.magic(2, 0) + (2**31).to_bytes(4, "little") + b"\0" * 128,
        )
    original = zipfile.ZipExtFile.read
    requests = []

    def bounded(stream, size=-1):
        requests.append(size)
        assert 0 <= size <= science.MAX_HEADER_BYTES
        return original(stream, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded)
    with pytest.raises(science.VerificationArtifactError):
        science.decode_verification_npz(
            packed.getvalue(), {"velocities": ((8, 2), np.dtype(np.float64))}
        )
    assert requests == [8, 4]


@pytest.mark.parametrize("boundary", ["raw", "decoded"])
def test_decoder_enforces_byte_budgets_before_array_allocation(monkeypatch, boundary):
    raw = npz(velocities=np.zeros((8, 2)))
    monkeypatch.setattr(science, "MAX_NPZ_BYTES" if boundary == "raw" else "MAX_DECODED_BYTES", 1)
    monkeypatch.setattr(
        science.np, "load", lambda *a, **k: pytest.fail("Oversized state allocated arrays")
    )
    with pytest.raises(science.VerificationArtifactError, match="TOO_LARGE"):
        science.decode_verification_npz(raw, {"velocities": ((8, 2), np.dtype(np.float64))})


@pytest.mark.parametrize("raw", [b'{"value":NaN}', b'{"value":1e309}', b'{"value":1,"value":2}'])
def test_invalid_json_is_a_safe_failed_check(tmp_path, raw):
    parameters = configuration()
    result, artifacts = saved(tmp_path, parameters)
    artifacts["manifest"] = raw
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert checked["numerical_artifact_validation"]["status"] == "failed"
    assert "Traceback" not in json.dumps(checked)


def test_cancellation_propagates_including_inside_decoder():
    def cancel():
        raise SimulationCancelled("cancelled")

    raw = npz(velocities=np.zeros((8, 2)))
    with pytest.raises(SimulationCancelled):
        science.decode_verification_npz(
            raw, {"velocities": ((8, 2), np.dtype(np.float64))}, check_cancelled=cancel
        )
    with pytest.raises(SimulationCancelled):
        science.verify_scientific_result(configuration(), {}, {}, check_cancelled=cancel)


def test_decoder_returns_finite_readonly_arrays_from_the_supplied_bytes():
    original = np.arange(16, dtype=np.float64).reshape(8, 2)
    decoded = science.decode_verification_npz(
        npz(velocities=original), {"velocities": ((8, 2), np.dtype(np.float64))}
    )
    np.testing.assert_array_equal(decoded["velocities"], original)
    assert not decoded["velocities"].flags.writeable
    original[:] = 0
    assert np.any(decoded["velocities"] != 0)


def test_strong_drift_cannot_hide_zeroed_centered_msd(tmp_path):
    parameters = LangevinParameters(
        1e-6,
        1e-6,
        8,
        friction=0.001,
        time_step=10.0,
        num_steps=100,
        sample_stride=10,
        acceleration=(1000.0, 0.0),
        seed=7,
    )
    result, artifacts = saved(tmp_path, parameters)
    assert result["final_centered_msd"] > 0.1
    result["final_centered_msd"] = 0.0
    report = json.loads(artifacts["transport"])
    report["centered_msd"] = [0.0] * len(report["centered_msd"])
    artifacts["transport"] = json.dumps(report).encode()
    manifest = json.loads(artifacts["manifest"])
    manifest["outputs"] = result
    artifacts["manifest"] = json.dumps(manifest).encode()
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert checked["centered_msd_series"]["status"] == "failed"
    assert checked["terminal_centered_msd"]["status"] == "failed"


def test_hot_initial_state_cannot_hide_zeroed_cold_terminal_energy_or_temperature(tmp_path):
    parameters = LangevinParameters(
        1e6, 1e-6, 8, friction=1000.0, time_step=0.01, num_steps=2, sample_stride=1, seed=7
    )
    result, artifacts = saved(tmp_path, parameters)
    assert 0 < result["thermal_temperature"] < 1e-6
    result.update(thermal_temperature=0.0, kinetic_energy=0.0)
    manifest = json.loads(artifacts["manifest"])
    manifest["outputs"] = result
    artifacts["manifest"] = json.dumps(manifest).encode()
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert checked["terminal_thermal_temperature"]["status"] == "failed"
    assert checked["terminal_kinetic_energy"]["status"] == "failed"


def test_extensive_lj_energy_scale_cannot_hide_intensive_temperature_error(tmp_path):
    parameters = LennardJonesParameters(
        num_particles=2000,
        box_size=(500.0, 500.0),
        epsilon=1000.0,
        initial_temperature=1e-4,
        num_steps=2,
        time_step=1e-4,
        sample_stride=1,
        seed=7,
    )
    result, artifacts = saved(tmp_path, parameters)
    report = json.loads(artifacts["observables"])
    report["series"]["thermal_temperature"][-1] += 1e-5
    artifacts["observables"] = json.dumps(report).encode()
    checked = checkmap(science.verify_scientific_result(parameters, result, artifacts))
    assert checked["terminal_series_thermal_temperature"]["status"] == "failed"
    assert checked["production_mean_thermal_temperature"]["status"] == "failed"
    assert checked["production_sample_sd_thermal_temperature"]["status"] == "failed"


def test_large_initial_msd_offset_has_isolated_conditioning_verdict_not_false_failure(tmp_path):
    parameters = LangevinParameters(
        1e6,
        1e-6,
        8,
        mass=0.001,
        friction=0.001,
        time_step=10.0,
        num_steps=10_000,
        sample_stride=100,
        seed=7,
    )
    result, artifacts = saved(tmp_path, parameters)
    checks = science.verify_scientific_result(parameters, result, artifacts)
    assert not any(item["status"] == "failed" for item in checks), checks
    checked = checkmap(checks)
    for name in ("diffusion_analytic_window_bias", "diffusion_analytic_window_estimate"):
        assert checked[name]["status"] == "inconclusive"
        assert checked[name]["reason"] == "ANALYTIC_WINDOW_CANCELLATION_LIMIT"
        assert (
            checked[name]["details"]["roundoff_bound"]
            > checked[name]["details"]["max_absolute_error"]
        )
    assert checked["terminal_thermal_temperature"]["status"] == "passed"
    assert checked["diffusion_particle_jackknife_error"]["status"] == "passed"


def test_finite_input_that_overflows_reference_arithmetic_yields_safe_failed_check(tmp_path):
    parameters = configuration()
    result, artifacts = saved(tmp_path, parameters)
    artifacts["final_state"] = npz(velocities=np.full((parameters.num_particles, 2), 1e308))
    with np.errstate(over="ignore", invalid="ignore"):
        checks = science.verify_scientific_result(parameters, result, artifacts)
    assert checkmap(checks)["numerical_artifact_validation"]["status"] == "failed"
    json.dumps(checks, allow_nan=False)
