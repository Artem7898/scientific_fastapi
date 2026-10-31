"""Submit independent verification without changing the source scientific result."""

import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ....core.verification_manager import VerificationManager
from ....schemas.verification import (
    VerificationAccepted,
    VerificationCertificate,
    VerificationList,
    VerificationRecord,
    VerificationRequest,
)
from ....services.verification_service import VerificationError
from .simulations import _valid_task_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Independent result verification"])


def get_verification_manager(request: Request) -> VerificationManager:
    return request.app.state.verification_manager


Manager = Annotated[VerificationManager, Depends(get_verification_manager)]


def _error(code: str, status: int) -> HTTPException:
    messages = {
        404: "Verification or source task not found",
        409: "Verification is unavailable for this source or state",
        429: "Verification admission limit reached",
        503: "Verification storage or execution is unavailable",
    }
    headers = {"Cache-Control": "no-store"}
    if status == 429 and code != "VERIFICATION_SOURCE_LIMIT_EXCEEDED":
        headers["Retry-After"] = "1"
    descriptions = {
        "VERIFICATION_SOURCE_NOT_COMPLETED": "Verification requires an archived completed simulation",
        "VERIFICATION_SOURCE_EVIDENCE_UNAVAILABLE": "The source has no committed artifact evidence to verify",
        "VERIFICATION_SOURCE_LIMIT_EXCEEDED": "This source has reached its retained verification limit",
        "VERIFICATION_QUEUE_FULL": "Verification queue capacity is exhausted; retry later",
        "VERIFICATION_CERTIFICATE_NOT_READY": "The certificate is available after the verification job completes",
        "VERIFICATION_CERTIFICATE_INTEGRITY_FAILED": "Saved verification certificate failed integrity checks",
    }
    return HTTPException(
        status,
        {
            "code": code,
            "message": descriptions.get(code, messages.get(status, "Could not read verification")),
        },
        headers=headers,
    )


async def _resolve(computation: Awaitable[Any]) -> Any:
    try:
        return await computation
    except VerificationError as error:
        raise _error(error.code, error.status_code) from None
    except Exception:
        logger.exception("Could not submit or read independent result verification")
        raise _error("VERIFICATION_STORAGE_UNAVAILABLE", 503) from None


def _validate_task(task_id: str) -> None:
    if not _valid_task_id(task_id):
        raise _error("VERIFICATION_SOURCE_NOT_FOUND", 404)


def _validate_verification(verification_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", verification_id):
        raise _error("VERIFICATION_NOT_FOUND", 404)


@router.post(
    "/simulations/{task_id}/verifications",
    response_model=VerificationAccepted,
    status_code=202,
    summary="Queue an independent verification of an archived completed result",
)
async def submit_verification(
    task_id: str,
    payload: VerificationRequest,
    response: Response,
    manager: Manager,
) -> VerificationAccepted:
    _validate_task(task_id)
    accepted = await _resolve(manager.submit(task_id, rerun=payload.rerun))
    response.headers.update(
        {
            "Location": accepted["check_url"],
            "Retry-After": "1",
            "Cache-Control": "no-store",
        }
    )
    return VerificationAccepted.model_validate(accepted)


@router.get(
    "/verifications/{verification_id}",
    response_model=VerificationRecord,
    summary="Read verification execution status separately from its scientific verdict",
)
async def get_verification(
    verification_id: str, response: Response, manager: Manager
) -> VerificationRecord:
    _validate_verification(verification_id)
    record = await _resolve(asyncio.to_thread(manager.get, verification_id))
    response.headers["Cache-Control"] = "no-store"
    return VerificationRecord.model_validate(record)


@router.get(
    "/simulations/{task_id}/verifications",
    response_model=VerificationList,
    summary="List recorded verification attempts independently of the source live TTL",
)
async def list_verifications(
    task_id: str, response: Response, manager: Manager
) -> VerificationList:
    _validate_task(task_id)
    records = await _resolve(asyncio.to_thread(manager.list, task_id))
    response.headers["Cache-Control"] = "no-store"
    return VerificationList.model_validate(records)


@router.get(
    "/verifications/{verification_id}/certificate",
    response_model=VerificationCertificate,
    response_class=JSONResponse,
    summary="Download the immutable unsigned verification certificate and its SHA256",
)
async def get_verification_certificate(verification_id: str, manager: Manager) -> Response:
    _validate_verification(verification_id)
    raw, digest = await _resolve(asyncio.to_thread(manager.certificate, verification_id))
    return Response(
        raw,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="verification-{verification_id}.json"',
            "X-Content-SHA256": digest,
            "Cache-Control": "no-store",
        },
    )
