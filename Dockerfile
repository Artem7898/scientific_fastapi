FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
ARG INSTALL_NUMBA=false
WORKDIR /app

# Install locked dependencies before source changes invalidate the build cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=ca_bundle \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    case "$INSTALL_NUMBA" in true) set -- --extra numba ;; false) set -- ;; *) echo "INSTALL_NUMBA must be true or false" >&2; exit 2 ;; esac; \
    uv sync --frozen --no-dev --no-install-project --extra redis --extra tracking --extra observability "$@"

COPY README.EN.md LICENSE ./
COPY src/scientific_api/ src/scientific_api/
RUN --mount=type=secret,id=ca_bundle \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    case "$INSTALL_NUMBA" in true) set -- --extra numba ;; false) set -- ;; *) echo "INSTALL_NUMBA must be true or false" >&2; exit 2 ;; esac; \
    uv sync --frozen --no-dev --no-editable --extra redis --extra tracking --extra observability "$@"

FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

RUN useradd --uid 10001 --create-home appuser
RUN mkdir -p /data/runs && chown -R appuser:appuser /data
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/data/runs
WORKDIR /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
CMD ["uvicorn", "scientific_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
