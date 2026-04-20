import logging
from typing import Any

import mlflow


logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Concrete implementation of ExperimentTracker for MLflow.
    Acts as an anti-corruption layer between our domain and MLflow's API.
    """

    def __init__(self, tracking_uri: str, experiment_name: str):
        if not tracking_uri:
            logger.warning("MLflow tracking URI is not set. Experiment tracking is disabled.")
            self._disabled = True
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._disabled = False
        logger.info(f"MLflow initialized. URI: {tracking_uri}, Experiment: {experiment_name}")

    def start_run(self, run_name: str) -> None:
        if self._disabled:
            return
        try:
            mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            # MLflow restricts parameter keys to alphanumeric and underscores
            sanitized_params = {k.replace(".", "_"): v for k, v in params.items()}
            mlflow.log_params(sanitized_params)
        except Exception as e:
            logger.error(f"Failed to log MLflow params: {e}")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._disabled:
            return
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to log MLflow metrics: {e}")

    def end_run(self) -> None:
        if self._disabled:
            return
        try:
            mlflow.end_run()
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")