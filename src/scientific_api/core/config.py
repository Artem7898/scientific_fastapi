"""Validated application configuration; resources are created by the lifespan."""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Scientific API"
    debug: bool = False
    artifacts_dir: Path = Path("artifacts/runs")
    max_simulation_iterations: int = Field(default=10_000, ge=1, le=10_000)
    max_particle_steps: int = Field(default=100_000_000, ge=1, le=100_000_000)
    max_queue_size: int = Field(default=100, ge=1, le=100_000)
    worker_count: int = Field(default=2, ge=1, le=32)
    task_ttl_seconds: int = Field(default=3600, ge=1)
    task_lease_seconds: int = Field(default=30, ge=3, le=3600)
    max_task_attempts: int = Field(default=3, ge=1, le=100)
    shutdown_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    redis_uri: str = ""
    redis_namespace: str = Field(default="scientific-api", pattern=r"^[A-Za-z0-9:_-]+$")
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "computational_physics"

    otel_enabled: bool = False
    otel_service_name: str = "scientific-api"
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_console: bool = False

    @field_validator("redis_uri")
    @classmethod
    def validate_redis_uri(cls, value: str) -> str:
        if value:
            parsed = urlsplit(value)
            if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
                raise ValueError("REDIS_URI must be a redis:// or rediss:// URL")
        return value

    @field_validator("otel_exporter_otlp_endpoint")
    @classmethod
    def validate_otel_endpoint(cls, value: str) -> str:
        if value:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT must be an HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Use OTLP exporter headers for credentials, not the endpoint URL")
        return value
