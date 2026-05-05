FROM python:3.12-slim

WORKDIR /app

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# зависимости
COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir .

# код
COPY src/ src/

# безопасность
RUN useradd -m appuser
USER appuser

# запуск
EXPOSE 8000
# Замени src.main на src.scientific_api.main
CMD ["uvicorn", "src.scientific_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]