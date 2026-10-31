"""Standalone Redis-backed LJ research workers: ``python -m scientific_api.research_worker``.

The process serves no HTTP. SIGINT/SIGTERM stop admission and allow the independent
research manager to finish or cooperatively release active work during shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from .core.config import Settings

logger = logging.getLogger(__name__)


class WorkerConfigurationError(ValueError):
    """Standalone execution has no configured shared task backend."""


def _worker_settings(configuration: Settings | None) -> Settings:
    current = configuration if configuration is not None else Settings()
    if not current.redis_uri:
        raise WorkerConfigurationError(
            "Standalone research workers require REDIS_URI and the scientific-api[redis] extra"
        )
    return Settings(**{**current.model_dump(), "execution_mode": "research-worker"})


def _create_app(configuration: Settings) -> Any:
    # Importing this CLI or asking for --help must not initialize the HTTP app.
    from .main import create_app

    return create_app(configuration)


async def run_worker(
    configuration: Settings | None = None, *, stop_event: asyncio.Event | None = None
) -> None:
    """Run standalone consumers until stopped; fail if a local consumer exits.

    Supplying an event lets a supervisor or test control shutdown without
    changing process signal handlers. Redis outages are retried by ResearchManager;
    an unexpectedly finished local worker loop is a process failure instead.
    """
    settings = _worker_settings(configuration)
    stopping = stop_event if stop_event is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handlers: dict[signal.Signals, Any] = {}
    try:
        if stop_event is None:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.getsignal(signum)
                try:
                    loop.add_signal_handler(signum, stopping.set)
                except (NotImplementedError, RuntimeError):
                    # Embedded callers in non-main threads should provide their
                    # own stop_event; the CLI uses the main POSIX event loop.
                    continue
                previous_handlers[signum] = previous
        application = _create_app(settings)
        async with application.router.lifespan_context(application):
            logger.info("Standalone result-research workers started")
            while not stopping.is_set():
                if not application.state.research_manager.ready:
                    raise RuntimeError("A standalone research worker loop stopped unexpectedly")
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=0.25)
                except TimeoutError:
                    pass
            logger.info("Stopping standalone result-research workers")
    finally:
        for signum, previous in previous_handlers.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_worker())
    except ValidationError:
        # Pydantic's input repr can include configured URLs or credentials.
        logger.error(
            "Invalid research worker configuration: set REDIS_URI and valid environment settings"
        )
        return 2
    except WorkerConfigurationError:
        logger.error(
            "Standalone research workers require REDIS_URI and the scientific-api[redis] extra"
        )
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.error("Standalone research worker startup or execution failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
