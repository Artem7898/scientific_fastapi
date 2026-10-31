"""Read current integrity evidence without changing legacy result responses."""

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response

from ....schemas.result_envelope import ResultEnvelope
from ....services.result_envelope_service import ResultEnvelopeError, read_result_envelope
from .simulations import Manager, get_simulation_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulations", tags=["Result Evidence"])


async def _read(
    task_id: str,
    request: Request,
    response: Response,
    manager: Manager,
    *,
    analysis_id: str | None = None,
) -> ResultEnvelope:
    result = await get_simulation_status(task_id, manager)
    if result["status"] != "completed":
        raise HTTPException(
            409,
            {
                "code": "RESULT_NOT_COMPLETED",
                "message": "Result evidence requires a completed simulation",
            },
            headers={"Cache-Control": "no-store"},
        )
    try:
        envelope = await asyncio.to_thread(
            read_result_envelope,
            result,
            artifacts_dir=Path(request.app.state.settings.artifacts_dir),
            analysis_id=analysis_id,
        )
    except ResultEnvelopeError as error:
        raise HTTPException(
            404 if error.code == "ANALYSIS_NOT_FOUND" else 409,
            {
                "code": error.code,
                "message": "Saved result evidence is unavailable or failed integrity checks",
            },
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        logger.exception("Could not read result evidence for %s", task_id)
        raise HTTPException(
            503,
            {
                "code": "EVIDENCE_STORAGE_UNAVAILABLE",
                "message": "Result evidence storage is unavailable",
            },
            headers={"Cache-Control": "no-store"},
        ) from None
    # A verified response describes bytes checked by this request. Caching it
    # could make a later caller miss a changed or removed server-side artifact.
    response.headers["Cache-Control"] = "no-store"
    return envelope


@router.get(
    "/{task_id}/envelope",
    response_model=ResultEnvelope,
    summary="Read a result envelope and recheck saved artifact integrity",
)
async def get_result_envelope(
    task_id: str,
    request: Request,
    response: Response,
    manager: Manager,
) -> ResultEnvelope:
    return await _read(task_id, request, response, manager)


@router.get(
    "/{task_id}/analyses/{analysis_id}/envelope",
    response_model=ResultEnvelope,
    summary="Read an analysis envelope and recheck saved artifact integrity",
)
async def get_analysis_envelope(
    task_id: str,
    analysis_id: str,
    request: Request,
    response: Response,
    manager: Manager,
) -> ResultEnvelope:
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        raise HTTPException(
            404,
            {"code": "ANALYSIS_NOT_FOUND", "message": "Analysis not found"},
            headers={"Cache-Control": "no-store"},
        )
    return await _read(task_id, request, response, manager, analysis_id=analysis_id)
