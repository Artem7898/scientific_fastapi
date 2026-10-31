"""Read-only discovery stays available without a queue or worker lifecycle."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scientific_api.api.v1.endpoints.models import router
from scientific_api.services import model_registry


@pytest.fixture
def catalog_client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client


def test_catalog_and_model_details_describe_all_registered_models(catalog_client):
    response = catalog_client.get("/api/v1/models")
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["schema_version"] == 1
    assert [item["model_version"] for item in catalog["models"]] == list(
        model_registry.MODEL_VERSIONS
    )
    assert {item["backend"] for item in catalog["backend_availability"]} == {"numpy", "numba"}
    for contract in catalog["models"]:
        detail = catalog_client.get(f"/api/v1/models/{contract['model_version']}")
        assert detail.status_code == 200
        assert detail.json() == contract
        assert "backend_availability" not in detail.json()


def test_absent_optional_backend_does_not_remove_scientific_support(catalog_client, monkeypatch):
    monkeypatch.setattr(
        model_registry.importlib.util,
        "find_spec",
        lambda name: object() if name == "numpy" else None,
    )
    response = catalog_client.get("/api/v1/models")
    assert response.status_code == 200
    catalog = response.json()
    numba = next(item for item in catalog["backend_availability"] if item["backend"] == "numba")
    assert numba["dependency_present"] is False
    assert numba["worker_readiness"] == "not_checked"
    assert all("numba" in {b["backend"] for b in model["backends"]} for model in catalog["models"])


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_model_catalog_is_read_only(catalog_client, method):
    response = getattr(catalog_client, method)("/api/v1/models/ideal-gas-ou-v1")
    assert response.status_code == 405


def test_unknown_scientific_model_is_not_found(catalog_client):
    response = catalog_client.get("/api/v1/models/ideal-gas-ou-v999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Scientific model not found"}


def test_generated_openapi_exposes_catalog_typed_responses(catalog_client):
    schema = catalog_client.get("/openapi.json").json()
    assert schema["paths"]["/api/v1/models"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ModelCatalog")
    assert schema["paths"]["/api/v1/models/{model_version}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ModelContract")
