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
| `research` | research figures |
| `notebooks` | Jupyter and notebook tooling |
| `observability` | OpenTelemetry and telemetry export |
| `benchmark` | additional computational backends for comparisons |

For example: `uv sync --frozen --extra redis --extra observability`. Redis, MLflow, and OpenTelemetry adapters load when their integration is enabled. An empty `MLFLOW_TRACKING_URI` disables tracking.

## Execution and telemetry

The default bounded in-memory queue supports local development in one process. Jobs and statuses are lost on restart, and independent processes do not share state. Enable `REDIS_URI` for shared state; acknowledgment, retries, and retention are described in the [architecture guide](docs/architecture.md).

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
