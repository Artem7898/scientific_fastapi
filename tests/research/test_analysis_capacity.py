"""Cancelling a request must not free an executor's still-running capacity."""

import asyncio
from threading import Event

import pytest

from scientific_api.core import analysis_manager
from scientific_api.core.analysis_manager import AnalysisCapacityError, AnalysisManager


@pytest.mark.asyncio
async def test_cancelled_request_holds_slot_until_its_thread_finishes(tmp_path, monkeypatch):
    entered, release, finished = Event(), Event(), Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"status": "completed"}

    monkeypatch.setattr(analysis_manager, "create_velocity_analysis", blocked)
    manager = AnalysisManager(tmp_path, workers=1)
    task = asyncio.create_task(manager.create({}, bins=12, formats=()))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(AnalysisCapacityError):
            await manager.create({}, bins=12, formats=())
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await asyncio.sleep(0.01)
        assert await manager.create({}, bins=12, formats=()) == {"status": "completed"}
    finally:
        release.set()
        await manager.stop()
