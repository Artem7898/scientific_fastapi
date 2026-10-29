"""Deterministic replica streams and statistics over logical contributions."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from scientific_api.schemas.ensemble import EnsembleRequest
from scientific_api.services.ensemble_service import (
    aggregate_ensemble,
    ensemble_manifest,
    prepare_ensemble,
)

PAYLOAD = {
    "parameters": {
        "initial_temperature": 300.0,
        "num_particles": 100,
        "time_step": 0.01,
        "num_steps": 3,
    },
    "replicas": 3,
    "base_seed": 42,
}


def snapshot(values=(1.0, 2.0, 3.0)) -> dict:
    children, metadata = prepare_ensemble(
        EnsembleRequest.model_validate({**PAYLOAD, "replicas": len(values)}), ensemble_id="example"
    )
    return {
        "ensemble_id": "example",
        "metadata": metadata,
        "children": [identifier for identifier, _ in children],
        "results": {
            identifier: {
                "task_id": identifier,
                "status": "completed",
                "seed": parameters["seed"],
                "final_temperature": value,
                "total_energy": 100 * value,
            }
            # Deliberately finish in reverse order: aggregation uses replica indices.
            for (identifier, parameters), value in reversed(
                list(zip(children, values, strict=True))
            )
        },
        "child_statuses": {identifier: "completed" for identifier, _ in children},
        "cancel_requested": False,
    }


def test_seed_materialization_and_request_fingerprint_are_independent_of_ids_and_global_rng():
    request = EnsembleRequest.model_validate(PAYLOAD)
    before = np.random.get_state(legacy=True)
    first_children, first_metadata = prepare_ensemble(request, ensemble_id="first")
    second_children, second_metadata = prepare_ensemble(request, ensemble_id="second")
    after = np.random.get_state(legacy=True)
    assert first_metadata == second_metadata
    assert [item[1] for item in first_children] == [item[1] for item in second_children]
    assert first_children[0][0] != second_children[0][0]
    assert first_metadata["seed_policy"]["spawn_keys"] == [[0], [1], [2]]
    expected = [
        int(stream.generate_state(1, dtype=np.uint64)[0])
        for stream in np.random.SeedSequence(42).spawn(3)
    ]
    assert first_metadata["seeds"] == expected
    assert len(set(expected)) == 3
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_backend_is_persisted_and_idempotency_preserves_legacy_numpy_fingerprint():
    request = EnsembleRequest.model_validate(PAYLOAD)
    numpy_children, numpy_metadata = prepare_ensemble(request, ensemble_id="numpy-run")
    legacy_request = request.model_dump()
    del legacy_request["parameters"]["backend"]
    legacy_fingerprint = hashlib.sha256(
        json.dumps(legacy_request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    assert numpy_metadata["request_fingerprint"] == legacy_fingerprint
    assert numpy_metadata["request"]["parameters"]["backend"] == "numpy"
    assert all(parameters["backend"] == "numpy" for _, parameters in numpy_children)
    numba_request = request.model_copy(
        update={"parameters": request.parameters.model_copy(update={"backend": "numba"})}
    )
    numba_children, numba_metadata = prepare_ensemble(numba_request, ensemble_id="numba-run")
    assert numba_metadata["request_fingerprint"] != legacy_fingerprint
    assert numba_metadata["request"]["parameters"]["backend"] == "numba"
    assert all(parameters["backend"] == "numba" for _, parameters in numba_children)
    assert [parameters["seed"] for _, parameters in numpy_children] == [
        parameters["seed"] for _, parameters in numba_children
    ]


def test_statistics_use_sample_sd_and_replica_order_and_preserve_inputs():
    source = snapshot()
    unchanged = copy.deepcopy(source)
    result = aggregate_ensemble(source)
    assert result["status"] == "completed"
    assert [item["task_id"] for item in result["children"]] == source["children"]
    assert [item["replica_index"] for item in result["children"]] == [0, 1, 2]
    temperature = result["statistics"]["final_temperature"]
    assert temperature["n"] == 3
    assert temperature["mean"] == 2
    assert temperature["sample_standard_deviation"] == 1
    assert temperature["standard_error"] == 1 / math.sqrt(3)
    assert result["theoretical_temperature_standard_error"] == 300 / math.sqrt(300)
    assert source == unchanged
    manifest = ensemble_manifest(source)
    assert manifest["seeds"] == source["metadata"]["seeds"]
    assert manifest["outputs"] == result
    assert "confidence interval" in manifest["statistics_method"]["limitations"]
    reordered = {**source, "results": dict(reversed(list(source["results"].items())))}
    assert aggregate_ensemble(reordered) == result


def test_single_replica_has_no_empirical_sample_sd_or_standard_error():
    statistics = aggregate_ensemble(snapshot((2.0,)))["statistics"]["final_temperature"]
    assert statistics == {
        "n": 1,
        "mean": 2.0,
        "sample_standard_deviation": None,
        "standard_error": None,
    }


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
def test_require_all_reports_failure_without_aggregating_only_survivors(terminal_status):
    source = snapshot()
    identifier = source["children"][1]
    source["results"][identifier] = {
        "task_id": identifier,
        "status": terminal_status,
        "error": "A replica did not finish",
    }
    result = aggregate_ensemble(source)
    assert result["status"] == terminal_status
    assert result["counts"][terminal_status] == 1
    assert result["counts"]["completed"] == 2
    assert result["statistics"] is None
    assert result["theoretical_temperature_standard_error"] is None
    assert result["children"][1]["result"]["error"] == "A replica did not finish"
    assert ensemble_manifest(source)["outputs"]["status"] == terminal_status


def test_active_ensemble_is_not_aggregated_or_exposed_as_a_terminal_manifest():
    source = snapshot()
    identifier = source["children"][0]
    del source["results"][identifier]
    source["child_statuses"][identifier] = "running"
    source["cancel_requested"] = True
    result = aggregate_ensemble(source)
    assert result["status"] == "running"
    assert result["cancel_requested"] is True
    assert result["counts"]["running"] == 1
    assert result["statistics"] is None
    with pytest.raises(ValueError, match="only after"):
        ensemble_manifest(source)


@pytest.mark.parametrize("metric", ["final_temperature", "total_energy"])
def test_nonfinite_or_missing_contributions_are_not_counted_as_scientific_success(metric):
    source = snapshot()
    source["results"][source["children"][0]][metric] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        aggregate_ensemble(source)
    source = snapshot()
    del source["results"][source["children"][0]]
    with pytest.raises(ValueError, match="missing a terminal"):
        aggregate_ensemble(source)


@pytest.mark.parametrize(
    "patch",
    [
        {"replicas": 0},
        {"replicas": 51},
        {"replicas": True},
        {"base_seed": -1},
        {"base_seed": 2**64},
        {"base_seed": True},
        {"failure_policy": "drop_failed"},
        {"model": "unversioned-model"},
        {"parameters": {**PAYLOAD["parameters"], "seed": 42}},
        {"parameters": {**PAYLOAD["parameters"], "initial_temperature": "300"}},
        {"parameters": {**PAYLOAD["parameters"], "num_particles": True}},
        {"parameters": {**PAYLOAD["parameters"], "time_step": float("inf")}},
        {"parameters": {**PAYLOAD["parameters"], "num_particles": 1_000_000, "num_steps": 100}},
    ],
)
def test_ensemble_contract_rejects_malformed_inputs_and_excess_total_work(patch):
    with pytest.raises(ValidationError):
        EnsembleRequest.model_validate({**PAYLOAD, **patch})


def langevin_snapshot(*, diffusion_available: bool = True) -> dict:
    request = EnsembleRequest.model_validate(
        {
            "model": "langevin-free-v1",
            "parameters": {
                "initial_temperature": 1e-6,
                "bath_temperature": 1e6,
                "num_particles": 3,
                "friction": 1e-3,
                "time_step": 1e-6,
                "num_steps": 1,
                "sample_stride": 1,
            },
            "replicas": 2,
        }
    )
    children, metadata = prepare_ensemble(request, ensemble_id="langevin")
    return {
        "ensemble_id": "langevin",
        "metadata": metadata,
        "children": [identifier for identifier, _ in children],
        "results": {
            identifier: {
                "task_id": identifier,
                "status": "completed",
                "seed": parameters["seed"],
                "thermal_temperature": 0.002,
                "kinetic_energy": 0.004,
                "final_centered_msd": 0.001,
                "diffusion": {
                    "status": "available" if diffusion_available else "unavailable",
                    "estimate": -0.1 if diffusion_available else None,
                },
            }
            for identifier, parameters in children
        },
    }


def test_langevin_ensemble_transient_temperature_error_is_stable_at_cold_hot_corner():
    source = langevin_snapshot()
    result = aggregate_ensemble(source)
    expected_temperature = 1e-6 * math.exp(-2e-9) + 1e6 * -math.expm1(-2e-9)
    assert result["theoretical_temperature_standard_error"] == expected_temperature / 2.0
    assert result["statistics"]["diffusion_estimate"]["mean"] == -0.1
    assert all(
        "model" in payload
        for _, payload in prepare_ensemble(
            EnsembleRequest.model_validate(source["metadata"]["request"]), ensemble_id="recreated"
        )[0]
    )
    manifest = ensemble_manifest(source)
    assert (
        "T_velocity(t)" in manifest["statistics_method"]["theoretical_temperature_standard_error"]
    )


def test_langevin_ensemble_requires_every_diffusion_fit_before_aggregating_diffusion():
    source = langevin_snapshot()
    child = source["children"][0]
    source["results"][child]["diffusion"] = {"status": "unavailable", "estimate": None}
    result = aggregate_ensemble(source)
    assert result["status"] == "completed"
    assert set(result["statistics"]) == {
        "thermal_temperature",
        "kinetic_energy",
        "final_centered_msd",
    }


@pytest.mark.parametrize(
    "model,parameters",
    [
        ("langevin-free-v1", PAYLOAD["parameters"]),
        ("ideal-gas-ou-v1", {**PAYLOAD["parameters"], "bath_temperature": 300.0}),
    ],
)
def test_ensemble_model_must_match_its_physical_parameter_contract(model, parameters):
    with pytest.raises(ValidationError):
        EnsembleRequest.model_validate({"model": model, "parameters": parameters, "replicas": 2})


def test_langevin_ensemble_caps_aggregate_sampled_trajectory_bytes_before_allocation():
    parameters = {
        "initial_temperature": 1.0,
        "bath_temperature": 1.0,
        "num_particles": 524_288,
        "num_steps": 3,
        "sample_stride": 1,
    }
    # Four frames per child times 16 children is exactly 512 MiB; this
    # contract check constructs no scientific state or trajectory arrays.
    request = EnsembleRequest.model_validate(
        {"model": "langevin-free-v1", "parameters": parameters, "replicas": 16}
    )
    assert request.parameters.num_particles == 524_288
    with pytest.raises(ValidationError, match="512 MiB"):
        EnsembleRequest.model_validate(
            {
                "model": "langevin-free-v1",
                "parameters": {**parameters, "num_particles": 524_289},
                "replicas": 16,
            }
        )
