"""Queued model dispatch is explicit and compatible with legacy records."""

from dataclasses import asdict

import pytest

from scientific_api.domain.experiments import parameters_from_payload
from scientific_api.domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.domain.simulation import MODEL_VERSION, SimulationParameters


def test_legacy_and_named_ideal_payloads_keep_the_original_parameters():
    parameters = SimulationParameters(2.0, 8, seed=2**64 - 1)
    for payload in (asdict(parameters), {"model": MODEL_VERSION, **asdict(parameters)}):
        original = dict(payload)
        assert parameters_from_payload(payload) == parameters
        assert payload == original


def test_langevin_json_payload_retains_pairs_model_seed_and_sampling():
    parameters = LangevinParameters(
        2.0, 1.0, 8, seed=2**64 - 1, acceleration=(0.5, -0.25), sample_stride=20
    )
    payload = {"model": LANGEVIN_MODEL_VERSION, **asdict(parameters)}
    payload["acceleration"] = [0.5, -0.25]
    assert parameters_from_payload(payload) == parameters


def test_lj_json_payload_retains_ensemble_interactions_and_seed():
    parameters = LennardJonesParameters(
        num_particles=16, seed=2**64 - 1, ensemble="nvt", bath_temperature=1.5
    )
    payload = {"model": LJ_MODEL_VERSION, **asdict(parameters)}
    payload["box_size"] = list(parameters.box_size)
    assert parameters_from_payload(payload) == parameters


@pytest.mark.parametrize("model", [None, True, "jax", "os.system", "unknown-v1", []])
def test_model_names_do_not_select_python_imports(model):
    with pytest.raises(ValueError, match="Unknown scientific model"):
        parameters_from_payload({"model": model})
