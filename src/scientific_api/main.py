import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from .api.v1.endpoints.simulations import router as sim_router
from .core.config import settings
from .core.queue_manager import queue_manager

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.getLogger(__name__).info("Initializing computational resources...")
    worker_task = asyncio.create_task(queue_manager.process_tasks())

    yield

    # Shutdown
    logging.getLogger(__name__).info("Releasing computational resources...")

    # 1. Сигнализируем воркеру остановиться и УБИВАЕМ синхронные потоки
    queue_manager.shutdown()

    # 2. Ждем завершения асинхронной обертки воркера (теперь она не зависнет)
    try:
        await asyncio.wait_for(worker_task, timeout=2.0)
    except asyncio.TimeoutError:
        worker_task.cancel()


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(sim_router, prefix="/api/v1")