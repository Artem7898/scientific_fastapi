# Scientific API Gateway

[Русский](README.md) · Python 3.12+ · [MIT](LICENSE)

`scientific-api` is a scientific service example built with FastAPI: an independent NumPy computational core, strict contracts, background jobs, and persisted artifacts. Version 2.0.0 makes reproducibility and result validation part of the workflow.

The `ideal-gas-ou-v1` model describes two-dimensional velocities of independent particles coupled to an Ornstein–Uhlenbeck thermostat. It uses **dimensionless reduced units** with `m = k_B = 1` and friction `γ = 1`. This is an educational stochastic dynamics model; its inputs do not represent Kelvin or femtoseconds.

## Quick start

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/). In a cloud task, use the existing isolated checkout; an additional Git worktree is unnecessary.

```bash
git clone https://github.com/Artem7898/scientific_fastapi.git
cd scientific_fastapi
uv sync --frozen
uv run --frozen uvicorn scientific_api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI is available at `http://localhost:8000/docs`, ReDoc at `/redoc`. Health endpoints are `/health/live` and `/health/ready`.

Submit a job:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/monte-carlo \
  -H 'Content-Type: application/json' \
  -d '{"initial_temperature":300.0,"num_particles":1000,"time_step":0.001,"num_steps":100,"seed":42}'
```

The service responds with `202 Accepted`:

```json
{
  "task_id": "a1b2c3d4",
  "status": "pending",
  "check_url": "/api/v1/simulations/status/a1b2c3d4"
}
```

Poll `check_url`. The same `task_id` passes through `pending`, `running`, and `completed` or `failed`/`cancelled`. A completed response includes temperature, energy, temperature standard error, model version, seed, and artifact references. See the [API reference](docs/api_reference.md) for the complete contract and limits.

## Analyze a saved simulation

After `status="completed"`, analyze the saved velocities without running another simulation. JSON analysis works with the minimal installation:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/analyses \
  -H 'Content-Type: application/json' \
  -d '{"bins":32,"formats":[]}'
```

The request waits for generation in a bounded thread pool and returns **201 Created** with `analysis_id`, the committed `attempt_id`, and `artifacts.analysis` and `artifacts.manifest` URLs. For PNG and PDF, install `uv sync --frozen --extra research`, then request `{"bins":32,"formats":["png","pdf"]}`. The response adds `artifacts.png` and `artifacts.pdf`; open these URLs in a browser or download them:

```bash
curl -o analysis.json http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/analysis
curl -o velocity_distribution.png http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/png
curl -o report.pdf http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/pdf
```

In `/docs`, select **Scientific Analysis → POST /api/v1/simulations/{task_id}/analyses → Try it out**, enter a completed task ID and request body, then click **Execute**. The PNG has three panels for `v_x`, `v_y`, and speed against the 2D Gaussian/Rayleigh reference; the PDF adds context and limitations. Moment and visual agreement do not prove correctness or equilibration: the current model already starts in its equilibrium distribution.

Two simultaneous analyses are allowed per API process by default; a busy service returns `429` with `Retry-After: 1`. This phase waits for completion over HTTP and has no persistent analysis queue. A disconnected request leaves its thread running and occupying capacity until completion; the client may lose its result URLs, and retrying POST creates a new analysis. See the [API reference](docs/api_reference.md) for source integrity checks, availability limits, and formats.

## Progress and cancellation

Server-Sent Events expose task progress. Existing polling through `check_url` continues to work:

```bash
curl -N http://localhost:8000/api/v1/simulations/TASK_ID/events
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/cancel
```

Cancellation returns `202` to acknowledge the request; check the eventual state through polling or SSE. Event history is bounded. On reconnect, send the last received `id` in `Last-Event-ID`; if replay history is unavailable, the stream reports a reset and the current state. Progress belongs to an execution attempt: recovery restarts from the saved seed, without a checkpoint.

## Ensembles of independent replicas

`POST /api/v1/ensembles` accepts one model and its parameters without `seed`, a replica count from `1..50`, and `base_seed`. For example:

```bash
curl -X POST http://localhost:8000/api/v1/ensembles \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: temperature-study-001' \
  -d '{"model":"ideal-gas-ou-v1","parameters":{"initial_temperature":300.0,"num_particles":1000,"num_steps":100},"replicas":4,"base_seed":42,"failure_policy":"require_all"}'
```

The `202` response includes `ensemble_id` and `check_url`. Read status through `GET /api/v1/ensembles/ENSEMBLE_ID`, request cancellation of remaining work through `POST /api/v1/ensembles/ENSEMBLE_ID/cancel`, and download `/api/v1/ensembles/ENSEMBLE_ID/manifest` after completion. These operations appear under **Ensembles** in `/docs`.

All seeds are saved before execution, and admission reserves capacity for every replica atomically. The total budget counts `replicas * num_particles * num_steps`; existing workers execute the jobs. An optional `Idempotency-Key` returns the same ensemble on retry; conflicting parameters produce `409`. With `require_all`, failures and cancellations remain explicit and no survivor-only aggregate is reported. Sample SD (`ddof=1`) and the mean's SE require all replicas to succeed and `M >= 2`; both are `null` for one replica. These estimates across independent replicas differ from the single-snapshot theoretical error `T/sqrt(N)`.

## Reproducibility

- Each calculation uses a local `numpy.random.Generator` with an explicit seed and PCG64 algorithm; the request specifies the number of steps.
- The API saves a run manifest and `final_state.npz` in `ARTIFACTS_DIR/<task_id>/<attempt_id>/`. `task_id` stays stable; `attempt_id` distinguishes execution attempts during queue recovery. The result references the committed attempt. The manifest records inputs, model, results, environment versions, and the commit when available.
- `uv.lock` fixes resolved dependencies; `uv sync --frozen` installs without regenerating the lockfile. A `pylock.toml` export supports tools implementing PEP 751.
- Tests check repeatability in a fixed environment, analytical properties of the model, the API contract, and research artifact generation.

A seed supports repeatability within a fixed software and hardware environment. Bitwise equality across NumPy versions, platforms, or computational backends is not guaranteed. The standard error `T / sqrt(N)` describes a terminal snapshot of independent particles in this model; it does not include physical model error or measurement uncertainty.

## Optional dependencies

The minimal installation runs the API and NumPy model. Install integrations explicitly:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest, Ruff, Pyright, package validation tools |
| `redis` | Redis status storage and queue |
| `tracking` | MLflow adapter |
| `research` | saved-velocity PNG/PDF reports and research figures |
| `notebooks` | Jupyter and notebook tooling |
| `observability` | OpenTelemetry and telemetry export |
| `benchmark` | additional computational backends for comparisons |

For example: `uv sync --frozen --extra redis --extra observability`. Redis, MLflow, and OpenTelemetry adapters load when their integration is enabled. An empty `MLFLOW_TRACKING_URI` disables tracking.

## Execution and telemetry

The default `EXECUTION_MODE=embedded` combines HTTP and computation with a bounded in-memory queue for local development in one process. Jobs and statuses are lost on restart, and independent processes do not share state. Enable `REDIS_URI` for shared state; acknowledgment, retries, and retention are described in the [architecture guide](docs/architecture.md).

For separate execution, use `EXECUTION_MODE=api` for HTTP without simulation queue consumers and start independent processes with `EXECUTION_MODE=worker python -m scientific_api.worker`. Both roles require Redis and access to the same artifacts. API readiness checks its backend; it does not establish that a worker is running. Saved-state analysis still runs in the API's bounded pool. `WORKER_COUNT` is the consumer count per process, not a cluster-wide limit. Start the local Compose configuration with a separate worker using:

When upgrading an existing deployment, stop admission and all old embedded API/worker processes using the Redis namespace first: running the old and new protocols together is unsupported. The [upgrade procedure](docs/development.md) preserves Redis and artifact volumes. For a new or already stopped deployment:

```bash
docker compose -f docker-compose.yml -f compose.worker.yml up --build
```

Use one Uvicorn worker with the in-memory queue. Multiple machines also need shared artifact storage: Redis does not share local files.

After installing `observability`, set `OTEL_ENABLED=true` to connect HTTP requests, queue waiting, and calculations in traces. Metrics describe queue waiting, execution duration, and errors. Exporter configuration is covered in the [development guide](docs/development.md).

## Development

```bash
uv sync --frozen --extra dev --extra research --extra redis --extra observability
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright src/
uv run --no-sync pytest
```

CI installs from the lockfile and checks code, reproducibility, and the built package. The [development guide](docs/development.md) covers research artifacts, benchmarking, builds, and PEP 751 export. Benchmarks use a fixed workload and report warmup separately. Measure process, Numba, JAX, and free-threaded Python performance on the target machine before claiming a speedup.

## Documentation and citation

- [Architecture and limitations](docs/architecture.md)
- [API reference](docs/api_reference.md)
- [Development, artifacts, and benchmarks](docs/development.md)
- [Citation metadata](CITATION.cff)

Author: Artem Alimpiev · [ORCID](https://orcid.org/0009-0007-6740-7242) · [Zenodo](https://doi.org/10.5281/zenodo.20057106).

Licensed under [MIT](LICENSE).
