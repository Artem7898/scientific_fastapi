"""Generate and download analyses of a completed simulation's saved state."""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ....core.analysis_manager import AnalysisCapacityError, AnalysisManager
from ....research.reporting import ResearchDependencyError
from ....schemas.analysis import AnalysisArtifact, VelocityAnalysisRequest, VelocityAnalysisResponse
from ....services.analysis_service import AnalysisSourceError, read_analysis_artifact
from .simulations import Manager, get_simulation_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulations", tags=["Scientific Analysis"])


def get_analysis_manager(request: Request) -> AnalysisManager:
    return request.app.state.analysis_manager


@router.post(
    "/{task_id}/analyses",
    response_model=VelocityAnalysisResponse,
    status_code=201,
    summary="Analyze saved 2D velocities; optionally generate PNG and PDF",
)
async def create_analysis(
    task_id: str,
    payload: VelocityAnalysisRequest,
    response: Response,
    manager: Manager,
    analyses: Annotated[AnalysisManager, Depends(get_analysis_manager)],
) -> VelocityAnalysisResponse:
    source = await get_simulation_status(task_id, manager)
    if source["status"] != "completed":
        raise HTTPException(409, "Analysis requires a completed simulation")
    try:
        result = await analyses.create(source, bins=payload.bins, formats=tuple(payload.formats))
    except AnalysisCapacityError:
        raise HTTPException(
            429, "Analysis capacity is exhausted; retry later", headers={"Retry-After": "1"}
        ) from None
    except ResearchDependencyError as error:
        raise HTTPException(503, str(error)) from None
    except AnalysisSourceError:
        raise HTTPException(
            409, "Saved simulation artifacts failed integrity or model checks"
        ) from None
    except Exception:
        logger.exception("Could not generate velocity analysis for %s", task_id)
        raise HTTPException(500, "Analysis generation failed; inspect server logs") from None
    prefix = f"/api/v1/simulations/{task_id}/analyses/{result['analysis_id']}/artifacts"
    result["artifacts"] = {name: f"{prefix}/{name}" for name in result["artifacts"]}
    response.headers["Location"] = result["artifacts"]["manifest"]
    return VelocityAnalysisResponse(**result)


@router.get("/{task_id}/analyses/{analysis_id}/artifacts/{artifact_name}")
async def download_analysis_artifact(
    task_id: str,
    analysis_id: str,
    artifact_name: AnalysisArtifact,
    request: Request,
    manager: Manager,
) -> Response:
    source = await get_simulation_status(task_id, manager)
    if source["status"] != "completed":
        raise HTTPException(409, "Analysis requires a completed simulation")
    try:
        data, filename, media_type = await asyncio.to_thread(
            read_analysis_artifact,
            source,
            artifacts_dir=request.app.state.settings.artifacts_dir,
            analysis_id=analysis_id,
            name=artifact_name,
        )
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(404, "Analysis artifact not found or invalid") from None
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
