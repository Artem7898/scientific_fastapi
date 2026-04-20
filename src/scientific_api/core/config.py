from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses strict validation to fail fast on misconfigurations.
    """
    model_config = SettingsConfigDict(
        env_file=".env_local_backup",
        env_file_encoding="utf-8",
        strict=True,  # Запрещает неявные преобразования типов в .env_local_backup
    )

    app_name: str = "Scientific API"
    debug: bool = False
    max_simulation_iterations: int = 10000

    # MLflow Configuration
    mlflow_tracking_uri: str = ""  # Оставь пустым, чтобы отключить логирование
    mlflow_experiment_name: str = "computational_physics"

    redis_uri: str = ""


settings = Settings()