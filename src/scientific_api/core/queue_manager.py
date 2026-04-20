import queue
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from ..schemas.simulation import SimulationRequest
from ..services.simulation_service import run_simulation
from ..core.task_store import task_store
from ..api.dependencies import get_experiment_tracker

logger = logging.getLogger(__name__)


class QueueManager:
    def __init__(self):
        # Используем обычную queue.Queue. Она не привязана к Event Loop
        # и идеально работает для передачи данных между потоками ОС.
        self._queue: queue.Queue[tuple[str, SimulationRequest]] = queue.Queue()
        self._is_shutting_down = False

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SciWorker")

    async def add_task(self, task_id: str, payload: SimulationRequest) -> None:
        if self._is_shutting_down:
            raise RuntimeError("Server is shutting down, not accepting new tasks")

        # put() синхронен и потокобезопасен. httpx может вызывать его из своего потока.
        self._queue.put((task_id, payload))
        logger.info(f"Task {task_id} added to background queue. Queue size: {self._queue.qsize()}")

    async def process_tasks(self) -> None:
        """
        Long-running coroutine that consumes tasks from the thread-safe queue.
        """
        logger.info("Background task processor started.")

        try:
            while not self._is_shutting_down:
                try:
                    # МАГИЯ ЗДЕСЬ: мы выносим блокирующее queue.get() в пул потоков.
                    # Это не блокирует Event Loop! Мы просто ждем, пока в очереди появится задача.

                    task_id, payload = await asyncio.get_running_loop().run_in_executor(
                        None, self._queue.get, True, 0.05
                    )
                except queue.Empty:
                    # Таймаут истек, просто проверяем флаг _is_shutting_down
                    continue

                logger.info(f"Worker picked up task {task_id}")

                try:
                    tracker = get_experiment_tracker()
                    loop = asyncio.get_running_loop()

                    # Выполняем тяжелую физику
                    result = await loop.run_in_executor(
                        self._executor,
                        run_simulation,
                        payload,
                        tracker
                    )

                    task_store.set(task_id, result.model_dump())
                    logger.info(f"Task {task_id} completed successfully.")

                except asyncio.CancelledError:
                    logger.warning(f"Task {task_id} cancelled. Forcing executor shutdown.")
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:
                    task_store.set(task_id, {"status": "failed", "error": str(e)})
                    logger.error(f"Task {task_id} failed: {e}")
                finally:
                    self._queue.task_done()
        finally:
            logger.info("Background task processor stopped.")

    def shutdown(self) -> None:
        logger.info("Initiating queue manager shutdown...")
        self._is_shutting_down = True
        self._executor.shutdown(wait=False, cancel_futures=True)


queue_manager = QueueManager()