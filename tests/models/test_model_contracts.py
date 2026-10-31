"""Scientific registry semantics, schema coverage and dependency-free discovery."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest
from pydantic import ValidationError

from scientific_api.domain.experiments import MODEL_PARAMETER_TYPES, parameters_from_payload
from scientific_api.domain.langevin import LANGEVIN_MODEL_VERSION, LangevinParameters
from scientific_api.domain.lennard_jones import LJ_MODEL_VERSION, LennardJonesParameters
from scientific_api.domain.simulation import MODEL_VERSION, SimulationParameters
from scientific_api.schemas.langevin import LangevinRequest, LangevinResponse
from scientific_api.schemas.lennard_jones import LennardJonesRequest, LennardJonesResponse
from scientific_api.schemas.model_contract import ModelContract
from scientific_api.schemas.simulation import SimulationRequest, SimulationResponse
from scientific_api.services import model_registry


def test_executable_model_versions_have_exactly_one_scientific_contract():
    versions = set(MODEL_PARAMETER_TYPES)
    assert set(model_registry.MODEL_VERSIONS) == versions
    assert set(model_registry._MODEL_METADATA_FACTORIES) == versions
    assert {
        contract.model_version for contract in model_registry.list_model_contracts()
    } == versions
    with pytest.raises(TypeError):
        MODEL_PARAMETER_TYPES["unexpected-v1"] = SimulationParameters


@pytest.mark.parametrize(
    "version,parameters",
    [
        (MODEL_VERSION, SimulationParameters(initial_temperature=2.0, num_particles=8)),
        (
            LANGEVIN_MODEL_VERSION,
            LangevinParameters(initial_temperature=2.0, bath_temperature=1.0, num_particles=8),
        ),
        (LJ_MODEL_VERSION, LennardJonesParameters(num_particles=16)),
    ],
)
def test_catalog_contract_versions_dispatch_representative_valid_parameters(version, parameters):
    assert model_registry.get_model_contract(version).model_version == version
    assert MODEL_PARAMETER_TYPES[version] is type(parameters)
    assert parameters_from_payload({"model": version, **asdict(parameters)}) == parameters


@pytest.mark.parametrize(
    "version,request_model,response_model,artifacts",
    [
        (MODEL_VERSION, SimulationRequest, SimulationResponse, {"manifest", "final_state"}),
        (
            LANGEVIN_MODEL_VERSION,
            LangevinRequest,
            LangevinResponse,
            {"manifest", "final_state", "trajectory", "transport"},
        ),
        (
            LJ_MODEL_VERSION,
            LennardJonesRequest,
            LennardJonesResponse,
            {"manifest", "final_state", "trajectory", "observables"},
        ),
    ],
)
def test_fixed_contracts_cover_current_transport_and_required_artifacts(
    version, request_model, response_model, artifacts
):
    contract = model_registry.get_model_contract(version)
    assert contract.request_schema == request_model.model_json_schema(mode="validation")
    assert contract.result_schema == response_model.model_json_schema(mode="serialization")
    assert contract.result_schema["properties"]["model_version"]["const"] == version
    assert {record.name for record in contract.required_artifacts} == artifacts
    assert {record.name for record in contract.required_artifacts} == set(
        contract.result_schema["properties"]["artifacts"]["propertyNames"]["enum"]
    )
    assert contract.default_backend == "numpy"
    assert {record.backend for record in contract.backends} == {"numpy", "numba"}
    assert all(not record.silent_fallback for record in contract.backends)
    assert all(record.device == "cpu" and record.dtype == "float64" for record in contract.backends)
    assert all(record.unit == "reduced" for record in contract.units)
    assert contract.version_policy.scope == "declarative_metadata"
    assert "source" in contract.version_policy.implementation_identity
    assert "coupled physics" in " ".join(contract.schema_limitations)


def test_scientific_contract_distinguishes_three_models_and_uncertainty_scopes():
    ou, langevin, lj = model_registry.list_model_contracts()
    assert tuple(contract.model_version for contract in (ou, langevin, lj)) == (
        MODEL_VERSION,
        LANGEVIN_MODEL_VERSION,
        LJ_MODEL_VERSION,
    )
    assert any("No particle coordinates" in assumption for assumption in ou.assumptions)
    assert any("not a general Monte Carlo sampler" in limit for limit in ou.limitations)
    assert ou.uncertainty_capabilities[0].method == "T_target / sqrt(N)"
    assert "sqrt(N-1)" in langevin.uncertainty_capabilities[0].method
    jackknife = next(
        item
        for item in langevin.uncertainty_capabilities
        if item.name == "diffusion_standard_error"
    )
    assert jackknife.availability == "conditional"
    assert "particle" in jackknife.method
    assert "all sampled times" in jackknife.independent_units
    assert any("1 percent" in assumption for assumption in jackknife.assumptions)
    assert any("without walls" in assumption for assumption in langevin.assumptions)
    assert any("not a canonical sample" in assumption for assumption in lj.assumptions)
    assert any("no tail correction" in assumption for assumption in lj.assumptions)
    assert any("NVE only" in invariant.applicability for invariant in lj.invariants)
    assert any("U, E and P may be negative" in invariant.definition for invariant in lj.invariants)
    assert all(
        "minimum" not in lj.result_schema["properties"][field]
        for field in ("potential_energy", "total_energy", "pressure")
    )
    unavailable = next(
        item
        for item in lj.uncertainty_capabilities
        if item.name == "single_run_confidence_interval"
    )
    assert unavailable.availability == "unavailable"
    assert "correlated" in unavailable.method
    assert any("not an independent-sample SE" in limit for limit in unavailable.limitations)
    for contract in (ou, langevin, lj):
        ensemble = next(
            item
            for item in contract.uncertainty_capabilities
            if item.name == "ensemble_standard_error"
        )
        assert "require_all" in " ".join(ensemble.assumptions)
        assert "null for M=1" in ensemble.method


def test_contracts_cannot_mutate_future_reads_and_digest_is_availability_independent(monkeypatch):
    contract = model_registry.get_model_contract(MODEL_VERSION)
    digest = model_registry.model_contract_sha256(contract)
    assert len(digest) == 64
    with pytest.raises(ValidationError):
        contract.model_version = "changed"
    with pytest.raises(ValidationError):
        contract.required_artifacts[0].filename = "changed"
    contract.request_schema["properties"]["seed"]["default"] = 999
    contract.result_schema["properties"].clear()
    fresh = model_registry.get_model_contract(MODEL_VERSION)
    assert fresh.request_schema["properties"]["seed"]["default"] == 0
    assert fresh.result_schema["properties"]
    assert model_registry.model_contract_sha256(fresh) == digest
    monkeypatch.setattr(model_registry.importlib.util, "find_spec", lambda name: None)
    assert not any(item.dependency_present for item in model_registry.backend_availability())
    assert model_registry.model_contract_sha256(
        model_registry.get_model_contract(MODEL_VERSION)
    ) == (digest)
    assert (
        len(
            {model_registry.model_contract_sha256(c) for c in model_registry.list_model_contracts()}
        )
        == 3
    )


@pytest.mark.parametrize(
    "field",
    ["units", "assumptions", "invariants", "diagnostics", "backends", "uncertainty_capabilities"],
)
def test_contract_publication_metadata_requires_scientific_declarations(field):
    data = model_registry.get_model_contract(MODEL_VERSION).model_dump()
    data[field] = ()
    with pytest.raises(ValidationError):
        ModelContract.model_validate(data)


def test_contract_rejects_duplicate_declarations_and_missing_manifest():
    original = model_registry.get_model_contract(MODEL_VERSION)
    for changes in (
        {"units": (*original.units, original.units[0])},
        {"required_artifacts": original.required_artifacts[1:]},
        {"backends": tuple(item for item in original.backends if item.backend == "numba")},
    ):
        with pytest.raises(ValidationError):
            ModelContract.model_validate({**original.model_dump(), **changes})


def test_unknown_model_has_no_dynamic_import_or_fallback(monkeypatch):
    def unexpected_import(version):
        pytest.fail("Unknown model must be rejected before schema import")

    monkeypatch.setattr(model_registry, "_json_schemas", unexpected_import)
    with pytest.raises(KeyError):
        model_registry.get_model_contract("user.module:run")


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_model_contract_version_requires_json_integer(value):
    data = model_registry.get_model_contract(MODEL_VERSION).model_dump(mode="json")
    data["schema_version"] = value
    with pytest.raises(ValidationError):
        ModelContract.model_validate_json(json.dumps(data))


def test_availability_is_dependency_discovery_not_execution_readiness(monkeypatch):
    monkeypatch.setattr(
        model_registry.importlib.util,
        "find_spec",
        lambda name: object() if name == "numpy" else None,
    )
    available = model_registry.backend_availability()
    assert [(item.backend, item.dependency_present) for item in available] == [
        ("numpy", True),
        ("numba", False),
    ]
    assert all(item.worker_readiness == "not_checked" for item in available)
    assert all(item.scope == "catalog_process" for item in available)
    assert all("no import" in item.interpretation for item in available)


def test_discovery_never_imports_optional_backends_or_integrations():
    script = """
import json
import sys
from scientific_api.services.model_registry import list_model_contracts, backend_availability
contracts = list_model_contracts()
availability = backend_availability()
optional = {'numba', 'llvmlite', 'jax', 'jaxlib', 'mlflow', 'redis', 'matplotlib', 'opentelemetry'}
print(json.dumps({'models': len(contracts), 'loaded': sorted(optional.intersection(sys.modules))}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert json.loads(result.stdout) == {"models": 3, "loaded": []}
