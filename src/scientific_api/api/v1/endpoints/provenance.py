"""Read archival scientific provenance independently of live task retention."""

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ....schemas.provenance import (
    ProvenanceAnalysis,
    ProvenanceArtifact,
    ProvenanceGraph,
    ProvenanceTask,
    ProvenanceTimeline,
)
from ....services.provenance_service import ProvenanceError, ProvenanceService
from .simulations import Manager, _valid_task_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Scientific Provenance"])


def _not_found(code: str) -> HTTPException:
    return HTTPException(
        404,
        {"code": code, "message": "Archived provenance not found"},
        headers={"Cache-Control": "no-store"},
    )


def _service(request: Request, manager: Manager) -> ProvenanceService:
    return ProvenanceService(manager.backend, Path(request.app.state.settings.artifacts_dir))


async def _read(
    operation: Callable[..., Any], response: Response, *args: Any, **kwargs: Any
) -> Any:
    try:
        result = await asyncio.to_thread(operation, *args, **kwargs)
    except ProvenanceError as error:
        messages = {
            404: "Archived provenance not found",
            409: "Archived evidence failed integrity or consistency checks",
            413: "Provenance export exceeds the supported size or record limits",
            422: "Invalid provenance query",
            503: "Provenance storage is unavailable",
        }
        raise HTTPException(
            error.status_code,
            {
                "code": error.code,
                "message": messages.get(error.status_code, "Could not read archived provenance"),
            },
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        logger.exception("Could not read archived scientific provenance")
        raise HTTPException(
            503,
            {
                "code": "PROVENANCE_STORAGE_UNAVAILABLE",
                "message": "Provenance storage is unavailable",
            },
            headers={"Cache-Control": "no-store"},
        ) from None
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get(
    "/provenance/tasks/{task_id}",
    response_model=ProvenanceTask,
    summary="Read an archived task and its verified evidence references",
)
async def get_task_provenance(
    task_id: str, request: Request, response: Response, manager: Manager
) -> ProvenanceTask:
    if not _valid_task_id(task_id):
        raise _not_found("PROVENANCE_NOT_FOUND")
    return await _read(_service(request, manager).task, response, task_id)


@router.get(
    "/provenance/tasks/{task_id}/graph",
    response_model=ProvenanceGraph,
    summary="Read the recorded entities and activities as a provenance graph",
)
async def get_provenance_graph(
    task_id: str, request: Request, response: Response, manager: Manager
) -> ProvenanceGraph:
    if not _valid_task_id(task_id):
        raise _not_found("PROVENANCE_NOT_FOUND")
    return await _read(_service(request, manager).graph, response, task_id)


@router.get(
    "/provenance/artifacts/{artifact_id}",
    response_model=ProvenanceArtifact,
    summary="Resolve an indexed artifact and verify its archived evidence",
)
async def get_artifact_provenance(
    artifact_id: str, request: Request, response: Response, manager: Manager
) -> ProvenanceArtifact:
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
        raise _not_found("ARTIFACT_NOT_FOUND")
    return await _read(_service(request, manager).artifact, response, artifact_id)


@router.get(
    "/provenance/analyses/{analysis_id}",
    response_model=ProvenanceAnalysis,
    summary="Resolve an indexed analysis and verify its source evidence",
)
async def get_analysis_provenance(
    analysis_id: str, request: Request, response: Response, manager: Manager
) -> ProvenanceAnalysis:
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        raise _not_found("ANALYSIS_NOT_FOUND")
    return await _read(_service(request, manager).analysis, response, analysis_id)


@router.get(
    "/tasks/{task_id}/timeline",
    response_model=ProvenanceTimeline,
    summary="Page through the archival lifecycle journal without SSE truncation",
)
async def get_task_timeline(
    task_id: str,
    request: Request,
    response: Response,
    manager: Manager,
    after_id: Annotated[str | None, Query(max_length=64, pattern=r"^[0-9]+-[0-9]+$")] = None,
    limit: Annotated[int, Query(ge=1, le=256)] = 100,
) -> ProvenanceTimeline:
    if not _valid_task_id(task_id):
        raise _not_found("PROVENANCE_NOT_FOUND")
    return await _read(
        _service(request, manager).timeline, response, task_id, after_id=after_id, limit=limit
    )


@router.get(
    "/provenance/tasks/{task_id}/export",
    response_class=Response,
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}
    },
    summary="Export verified archived evidence as a bounded RO-Crate ZIP",
)
async def export_task_provenance(
    task_id: str,
    request: Request,
    response: Response,
    manager: Manager,
    export_format: Annotated[Literal["ro-crate"], Query(alias="format")] = "ro-crate",
) -> Response:
    if not _valid_task_id(task_id):
        raise _not_found("PROVENANCE_NOT_FOUND")
    from ....services.provenance_export import export_ro_crate

    payload, filename = await _read(export_ro_crate, response, _service(request, manager), task_id)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", filename):
        raise HTTPException(
            503,
            {"code": "PROVENANCE_STORAGE_UNAVAILABLE", "message": "Could not export provenance"},
            headers={"Cache-Control": "no-store"},
        )
    return Response(
        payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
