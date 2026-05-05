import json
import redis
from ..core.config import settings

class TaskStore:
    """
    In-memory/Redis store for background task results.
    Acts as a simple key-value cache for task statuses and payloads.
    """
    def __init__(self):
        # Подключаемся к Redis, если URI указан, иначе используем in-memory dict
        if settings.redis_uri:
            self._redis = redis.from_url(settings.redis_uri, decode_responses=True)
        else:
            self._redis = None
            self._memory: dict[str, str] = {}

    def set(self, task_id: str, data: dict) -> None:
        serialized = json.dumps(data)
        if self._redis:
            # Храним результат 1 час (3600 секунд)
            self._redis.setex(f"task:{task_id}", 3600, serialized)
        else:
            self._memory[task_id] = serialized

    def get(self, task_id: str) -> dict | None:
        if self._redis:
            data = self._redis.get(f"task:{task_id}")
            if data is not None:
                return json.loads(str(data))
            return None
        else:
            raw = self._memory.get(task_id, "null")
            return json.loads(raw)

# Глобальный синглтон
task_store = TaskStore()