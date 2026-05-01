<div align="center">

# 🔬 Scientific API Gateway

**Высокопроизводительный, строго типизированный API-фреймворк для вычислительной физики и научного машинного обучения**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square)](https://docs.pydantic.dev/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-261230?style=flat-square)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 🎯 Зачем это нужно?

Большинство научных API строятся как хрупкие обертки вокруг Jupyter Notebooks. В них нет строгой валидации данных, они падают при `malformed`-вводе (искаженных данных) и их невозможно воспроизводимо развернуть на HPC-кластерах или облачной инфраструктуре.

Этот шаблон решает эту проблему, навязывая **Domain-Driven Design (DDD)** и **Строгую типизацию** на уровне архитектуры:

| Принцип | Описание |
|---------|----------|
| 🔒 **Воспроизводимость по умолчанию** | Каждый запрос строго валидируется через Pydantic v2 (`strict=True`). Строка не может быть неявно приведена к числу с плавающей точкой. Физические ограничения (например, `T>0`) enforced (принудительно выполняются) на границе API. |
| 🏗️ **Разделение зон ответственности** | Научные алгоритмы (`models/`) ничего не знают про HTTP. Они могут тестироваться изолированно, экспортироваться как CLI-утилиты или использоваться в симуляциях для статьи без изменения единой строки кода. |
| 🚀 **Современный Python (2025+)** | Никакого legacy-балласта. Построен на PEP 621, управляется через `uv`, типизируется через `pyright`, форматируется через `ruff`. |

---

## 🏛️ Обзор архитектуры

```
Client Request (Запрос клиента)
         │
         ▼
[API Layer - FastAPI] ──► Строгая валидация Pydantic
         │
         ▼
[Service Layer] ───────► Бизнес-логика, трекинг экспериментов (MLflow/W&B)
         │
         ▼
[Scientific Core] ─────► Чистые вычисления NumPy/JAX/C++ (Stateless / без состояния)
```

---

## 🚀 Быстрый старт

> **Требуется установленный [uv](https://docs.astral.sh/uv/)** — самый быстрый пакетный менеджер Python.

```bash
# Клонирование репозитория
git clone https://github.com/Artem7898/scientific_fastapi
cd scientific-fastapi

# Создание виртуального окружения и установка зависимостей
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Запуск строгой проверки типов и линтинга
ruff check .
pyright src/

# Запуск тестов
pytest

# Запуск API-сервера
uvicorn src.scientific_api.main:app --reload
```

📚 Перейдите по адресу `http://localhost:8000/redoc` для интерактивной спецификации API.

---

## 📖 Документация

- [Архитектура и проектные решения](docs/architecture.md) — ADR (Architecture Decision Records)
- [Справочник API](docs/api-reference.md) — Полная документация эндпоинтов
- [Руководство по разработке](docs/development.md) — Стандарты кода и тестирования

---

## 🧪 Пример использования

### Запуск симуляции Монте-Карло

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

## 📋 Справочник API (v1)

**Base URL:** `http://localhost:8000/api/v1`

### Запуск симуляции

Выполняет симуляцию стохастической динамики на основе приближений идеального газа.

**Эндпоинт:** `POST /simulations/monte-carlo`  
**Content-Type:** `application/json`

#### Параметры тела запроса

| Параметр | Тип | Ограничения | Описание |
|----------|-----|-------------|----------|
| `initial_temperature` | float | > 0.0, <= 1 000 000 | Начальная температура системы в Кельвинах (K) |
| `num_particles` | integer | > 0, <= 1 000 000 | Количество частиц в объеме симуляции (N) |
| `time_step` | float | > 0.0 | Шаг интегрирования в фемтосекундах (fs). По умолчанию 0.001 |

#### Схема ответа (200 OK)

```json
{
  "task_id": "a1b2c3d4...",
  "final_temperature": 298.4521,
  "total_energy": 1245.887,
  "status": "completed"
}
```

#### Коды ошибок

| Код | Описание |
|-----|----------|
| `422 Unprocessable Entity` | Нарушены физические ограничения или неверные типы JSON |
| `500 Internal Server Error` | Численная нестабильность (например, переполнение NaN) в вычислительном ядре |

---

## 🏗️ Структура проекта

```
scientific_fastapi/
├── pyproject.toml          # Метаданные, зависимости (PEP 621)
├── Dockerfile              # Multi-stage build для минимального образа
├── src/
│   └── scientific_api/
│       ├── __init__.py
│       ├── main.py         # Точка входа, lifespan, middleware
│       ├── api/
│       │   ├── __init__.py
│       │   ├── v1/
│       │   │   ├── __init__.py
│       │   │   └── endpoints/
│       │   │       ├── __init__.py
│       │   │       └── simulations.py  # Роутеры
│       │   └── dependencies.py         # Общие зависимости (DI)
│       ├── schemas/                    # Pydantic модели (контракты)
│       │   ├── __init__.py
│       │   └── simulation.py
│       ├── services/                   # Бизнес-логика
│       │   ├── __init__.py
│       │   └── simulation_service.py
│       ├── core/                       # Настройки, логирование
│       │   ├── __init__.py
│       │   ├── config.py               # Pydantic Settings
│       │   └── logging.py
│       └── models/                     # Чистая наука (NumPy/JAX/PyTorch)
│           ├── __init__.py
│           └── physics_engine.py
└── tests/
    ├── __init__.py
    ├── conftest.py                     # Фикстуры для pytest
    └── api/
        └── test_simulations.py         # Тесты API (httpx.AsyncClient)
```



## 🔧 Требования

- **Python:** 3.12+
- **Пакетный менеджер:** [uv](https://docs.astral.sh/uv/)
- **Линтер:** Ruff
- **Type Checker:** Pyright
- **Тестирование:** pytest

---

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте feature-ветку (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

---
Автор проекта: Артем Алимпиев
#### https://orcid.org/0009-0007-6740-7242

## 📄 Лицензия

Распространяется под лицензией MIT. См. [LICENSE](LICENSE) для подробностей.

---

<div align="center">

**[⬆ Наверх](#-scientific-api-gateway)**

Made with 🔬 for Science

</div>
