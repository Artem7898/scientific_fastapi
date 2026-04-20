# Stage 1: Dependencies
FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir .

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/

EXPOSE 8000
# Используем команду без кавычек, чтобы shell корректно передал сигналы остановки (Ctrl+C)
CMD uvicorn src.scientific_api.main:app --host 0.0.0.0 --port 8000