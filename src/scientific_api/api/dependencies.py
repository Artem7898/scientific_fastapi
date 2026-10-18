"""Resolve per-application resources without creating import-time singletons."""

from fastapi import Request

from ..core.queue_manager import QueueManager


def get_queue_manager(request: Request) -> QueueManager:
    return request.app.state.queue_manager
