"""Discover fixed scientific contracts without initializing compute backends."""

from fastapi import APIRouter, HTTPException

from ....schemas.model_contract import ModelCatalog, ModelContract
from ....services.model_registry import (
    backend_availability,
    get_model_contract,
    list_model_contracts,
)

router = APIRouter(prefix="/models", tags=["Scientific Models"])


@router.get("", response_model=ModelCatalog, summary="List supported scientific model contracts")
def list_models() -> ModelCatalog:
    return ModelCatalog(models=list_model_contracts(), backend_availability=backend_availability())


@router.get(
    "/{model_version}",
    response_model=ModelContract,
    summary="Read one versioned scientific contract",
)
def read_model(model_version: str) -> ModelContract:
    try:
        return get_model_contract(model_version)
    except KeyError:
        raise HTTPException(404, "Scientific model not found") from None
