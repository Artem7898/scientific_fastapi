<div align="center">

# 🔬 Scientific API Gateway

**A high-performance, strongly typed API framework for computational physics and scientific machine learning**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square)](https://docs.pydantic.dev/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-261230?style=flat-square)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 🎯 Why is this necessary?

Most scientific APIs are built as fragile wrappers around Jupyter Notebooks. They do not have strict data validation, they crash with `malformed` input (distorted data) and cannot be reproducibly deployed on HPC clusters or cloud infrastructure.

This template solves this problem by imposing **Domain-Driven Design (DDD)** and **Strong typing** at the architecture level:

| Principle | Description |
|---------|----------|
| 🔒 **Default reproducibility** | Each request is strictly validated via Pydantic v2 (`strict=True`). A string cannot be implicitly converted to a floating-point number. Physical constraints (for example, `T>0`) are enforced at the API boundary. |
| 🏗️ **Separation of areas of responsibility** | Scientific algorithms (`models/`) they don't know anything about HTTP. They can be tested in isolation, exported as CLI utilities, or used in simulations for an article without changing a single line of code. |
| 🚀 **Modern Python (2025+)** | No legacy ballast. Built on PEP 621, controlled via `uv`, typed via `pyright`, formatted via `ruff'. |

---

## 🏛️ Architecture Overview

``
Client Request


[API Layer - FastAPI] ──► Strict Pydantic validation
         │
         ▼
[Service Layer] ──────► Business Logic, experiment Tracking (MLflow/W&B)


[Scientific Core] ─────► Pure NumPy/JAX/C++ Computing (Stateless)
``

---

## 🚀 Quick Start

> **Requires an installed [uv] (https://docs.astral.sh/uv /)** is the fastest Python package manager.

```bash
# Cloning the repository
git clone https://github.com/Artem7898/scientific_fastapi
cd scientific-fastapi

# Creating a virtual environment and installing dependencies
uv venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Running strict type checking and linting
ruff check .
pyright src/

# Running tests
pytest

# Launching
the uvicorn src API server.scientific_api.main:app --reload
```

, Go to `http://localhost:8000/redoc ` for the interactive API specification.

---

## , Documentation

- [Architecture and Design Solutions](docs/architecture.md) — ADR (Architecture Decision Records)
- [API Reference](docs/api-reference.md) — Complete endpoint documentation
- [Development Guide](docs/development.md) — Standards for code and testing

---

## 🧪 Usage example

### Running a Monte Carlo simulation

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/simulations/monte-carlo",
    json={
        "initial_temperature": 300.0,  # K
        "num_particles": 1000,
        "time_step": 0.001  # fs
    }
)

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Final Temperature: {result['final_temperature']} K")
```

---

## 📋 API Reference (v1)

**Base URL:** `http://localhost:8000/api/v1`

### Running the simulation

Performs a simulation of stochastic dynamics based on approximations of an ideal gas.

**Endpoint:** `POST /simulations/monte-carlo`  
**Content-Type:** `application/json`

#### Request Body parameters

| Parameter | Type | Restrictions | Description |
|----------|-----|-------------|----------|
| `initial_temperature` | float | > 0.0, <= 1 000 000 | The initial temperature of the system in Kelvin (K) |
| `num_particles` | integer | > 0, <= 1 000 000 | Number of particles in the simulation volume (N) |
| `time_step` | float | > 0.0 | The integration step is in femtoseconds (fs). The default value is 0.001 |

#### Response scheme (200 OK)

```json
{
  "task_id": "a1b2c3d4...",
  "final_temperature": 298.4521,
  "total_energy": 1245.887,
  "status": "completed"
}
```

#### Error codes

| Code | Description |
|-----|----------|
| `422 Unprocessable Entity` | Violated physical restrictions or incorrect JSON types |
| `500 Internal Server Error` | Numerical instability (e.g. NaN overflow) in the computing core |

---

## 🏗️ Project structure

````
scientific_fastapi/
├── pyproject.toml                    # Metadata, dependencies (PEP 621)
├── Dockerfile                        # Multi-stage build for minimal image
├── src/
│   └── scientific_api/
│       ├── __init__.py
│       ├── main.py                    # Entry point, lifespan, middleware
│       ├── api/
│       │   ├── __init__.py
│       │   ├── v1/
│       │   │   ├── __init__.py
│       │   │   └── endpoints/
│       │   │   ├──__init__.py ,
|       |   |   |── simulations.py      # Routers
│       │   └── dependencies.py         # Shared Dependencies (DI)
│       ├── schemas/                    # Pydantic models (contracts)
│       │   ├──__init__.py ,
|       |   |── simulation.py ,
|       |── services/                   # Business logic
│       │   ├── __init__.py
│       │   └── simulation_service.py
|       |── core/                        # Settings, logging
│       │   ├── __init__.py
│       │   ├── config.py               # Pydantic Settings
│       │   └── logging.py
│       └── models/                     # Pure Science (NumPy/JAX/PyTorch)
│           ├──__init__.py
|           |── physics_engine.py ,
|── tests/
    |──__init__.py ,
    |── conftest.py                   # Fixtures for pytest
    └── api/
        └── test_simulations.py       # API Tests (httpx.AsyncClient)
```



## 🔧 Requirements

- **Python:** 3.12+
- **Package Manager:** [uv](https://docs.astral.sh/uv/)
- **Linter:** Ruff
- **Type Checker:** Pyright
- **Testing:** pytest

---

## 🤝 Contribution to the project

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature')
3. Commit the changes (`git commit -m 'Add amazing feature')
4. Send to the branch ('git push origin feature/amazing-feature')
5. Open the Pull Request

---
  The Author of the project is Artem Alimpiev
https://orcid.org/0009-0007-6740-7242

 📄 License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.

---



