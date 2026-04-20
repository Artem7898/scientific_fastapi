
---

### 2. Документация для разработчиков (`docs/architecture.md`)

Создай папку `docs/` в корне проекта. Это внутренняя документация для твоей команды.

```markdown
# Architecture & Design Decisions (ADR)

## 1. Why Pydantic `strict=True`?
In standard web development, coercing `"300"` to `300.0` is a convenience. In scientific computing, it is a silent source of error. If an instrument sends a malformed payload, we *must* reject it (HTTP 422) rather than proceeding with mutated data. 

## 2. Synchronous vs Asynchronous Services
Notice that `simulation_service.py` calls the physics engine synchronously, even though the endpoint is `async def`. 
**Rule:** CPU-bound tasks (NumPy, SciPy, C++ bindings) **must not** run inside Python's `asyncio` event loop. They will block the loop and freeze the server.
* **Current state:** For micro-simulations (< 1 sec), synchronous execution is acceptable.
* **Scaling path:** For real workloads, you must offload execution using:
  1. `asyncio.to_thread(run_simulation, payload)` (good for I/O or very light CPU tasks).
  2. A task queue (Celery + Redis / RQ) returning a `task_id` and polling via a separate endpoint (required for HPC/GPU jobs).

## 3. Directory Structure Boundaries
* `src/scientific_api/models/`: The only place allowed to import `numpy`, `jax`, or `torch`. 
* `src/scientific_api/schemas/`: The only place allowed to import `pydantic`. 
* `src/scientific_api/services/`: The glue. It maps Pydantic schemas to Python dataclasses/dicts required by the models.