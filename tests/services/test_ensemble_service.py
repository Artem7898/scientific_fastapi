"""Deterministic replica streams and statistics over logical contributions."""

import copy
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
