"""Bound concurrent report generation without tying capacity to HTTP lifetime."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from ..services.analysis_service import create_velocity_analysis


class AnalysisCapacityError(RuntimeError):
    """No capacity is available for another analysis request."""


class AnalysisManager:
    def __init__(self, artifacts_dir: Path, workers: int = 2) -> None:
        self.artifacts_dir = artifacts_dir
        self._slots = BoundedSemaphore(workers)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="AnalysisWorker"
        )
        self._futures: set[asyncio.Future[Any]] = set()
        self._closed = False

    async def create(
        self, result: dict[str, Any], *, bins: int, formats: tuple[str, ...]
    ) -> dict[str, Any]:
        if self._closed or not self._slots.acquire(blocking=False):
            raise AnalysisCapacityError("Analysis capacity is exhausted")
        function = partial(
            create_velocity_analysis,
            deepcopy(result),
            artifacts_dir=self.artifacts_dir,
            bins=bins,
            formats=formats,
        )

        def execute() -> dict[str, Any]:
            try:
                return function()
            finally:
                # Cancelling an HTTP coroutine does not stop its OS thread.
                self._slots.release()

        try:
            future = asyncio.get_running_loop().run_in_executor(self._executor, execute)
        except BaseException:
            self._slots.release()
            raise
        self._futures.add(future)

        def finished(done: asyncio.Future[Any]) -> None:
            self._futures.discard(done)
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    async def stop(self, timeout: float = 5.0) -> None:
        self._closed = True
        if self._futures:
            await asyncio.wait(self._futures, timeout=timeout)
        self._executor.shutdown(wait=False, cancel_futures=False)
