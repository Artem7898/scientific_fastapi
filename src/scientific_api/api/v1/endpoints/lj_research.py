"""Queue bounded research analyses of pinned Lennard–Jones result archives."""

import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ....core.research_manager import ResearchManager
from ....schemas.lj_research import (
    LJResearchAccepted,
    LJResearchList,
    LJResearchRecord,
    LJResearchReport,
    LJResearchRequest,
)
from ....services.lj_research_service import ResearchError
from .simulations import _valid_task_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Lennard–Jones research"])


def get_research_manager(request: Request) -> ResearchManager:
    return request.app.state.research_manager


Manager = Annotated[ResearchManager, Depends(get_research_manager)]


def _error(code: str, status: int) -> HTTPException:
    messages = {
        404: "Research analysis or source task not found",
        409: "Research analysis is unavailable for these sources or state",
        413: "Research metadata exceeds supported limits",
        429: "Research analysis admission limit reached",
        503: "Research analysis storage or execution is unavailable",
    }
    descriptions = {
        "RESEARCH_SOURCE_NOT_COMPLETED": "Research analysis requires archived completed simulations",
        "RESEARCH_SOURCE_EVIDENCE_UNAVAILABLE": "A source has no committed artifact evidence to analyze",
        "RESEARCH_SOURCE_LIMIT_EXCEEDED": "The primary source has reached its retained research analysis limit",
        "RESEARCH_QUEUE_FULL": "Research analysis queue capacity is exhausted; retry later",
        "RESEARCH_REPORT_NOT_READY": "The report is available after the research analysis completes",
        "RESEARCH_REPORT_INTEGRITY_FAILED": "Saved research report failed integrity checks",
    }
    headers = {"Cache-Control": "no-store"}
    if status == 429 and code != "RESEARCH_SOURCE_LIMIT_EXCEEDED":
        headers["Retry-After"] = "1"
    return HTTPException(
        status,
        {
            "code": code,
            "message": descriptions.get(
                code, messages.get(status, "Could not read research analysis")
            ),
        },
        headers=headers,
    )


async def _resolve(computation: Awaitable[Any]) -> Any:
    try:
        return await computation
    except ResearchError as error:
        raise _error(error.code, error.status_code) from None
    except Exception:
        logger.exception("Could not submit or read Lennard–Jones research analysis")
        raise _error("RESEARCH_STORAGE_UNAVAILABLE", 503) from None


def _validate_analysis(analysis_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        raise _error("RESEARCH_NOT_FOUND", 404)


@router.post(
    "/research/lj/analyses",
    response_model=LJResearchAccepted,
    status_code=202,
    summary="Queue a research analysis of archived Lennard–Jones simulations",
)
async def submit_lj_research(
    payload: LJResearchRequest, response: Response, manager: Manager
) -> LJResearchAccepted:
    accepted = await _resolve(manager.submit(payload))
    response.headers.update(
        {
            "Location": accepted["check_url"],
            "Retry-After": "1",
            "Cache-Control": "no-store",
        }
    )
    return LJResearchAccepted.model_validate(accepted)


@router.get(
    "/research/analyses/{analysis_id}",
    response_model=LJResearchRecord,
    summary="Read research execution status separately from statistical availability",
)
async def get_research_analysis(
    analysis_id: str, response: Response, manager: Manager
) -> LJResearchRecord:
    _validate_analysis(analysis_id)
    record = await _resolve(asyncio.to_thread(manager.get, analysis_id))
    response.headers["Cache-Control"] = "no-store"
    return LJResearchRecord.model_validate(record)


@router.get(
    "/research/analyses/{analysis_id}/report",
    response_model=LJResearchReport,
    response_class=JSONResponse,
    summary="Download the immutable research report and its SHA256",
)
async def get_research_report(analysis_id: str, manager: Manager) -> Response:
    _validate_analysis(analysis_id)
    raw, digest = await _resolve(asyncio.to_thread(manager.report, analysis_id))
    return Response(
        raw,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="lj-analysis-{analysis_id}.json"',
            "X-Content-SHA256": digest,
            "Cache-Control": "no-store",
        },
    )


@router.get(
    "/simulations/{task_id}/research-analyses",
    response_model=LJResearchList,
    summary="List analyses indexed by this source as their primary anchor",
)
async def list_source_research(
    task_id: str, response: Response, manager: Manager
) -> LJResearchList:
    if not _valid_task_id(task_id):
        raise _error("RESEARCH_SOURCE_NOT_FOUND", 404)
    records = await _resolve(asyncio.to_thread(manager.list, task_id))
    response.headers["Cache-Control"] = "no-store"
    return LJResearchList.model_validate(records)
