"""An optional MLflow adapter with per-run state and explicit failures."""

from contextvars import ContextVar
from importlib import import_module
from pathlib import Path
from typing import Any

from ..core.experiment_tracker import ExperimentTracker, NullTracker


class MLflowTracker:
    """Use a dedicated client instead of changing MLflow's global active run."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        if not tracking_uri:
            raise ValueError(
                "MLflowTracker requires a tracking URI; use build_tracker to disable it"
            )
        try:
            module = import_module("mlflow.tracking")
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is configured but unavailable; install scientific-api[tracking]"
            ) from exc
        try:
            self._client = module.MlflowClient(tracking_uri=tracking_uri)
            experiment = self._client.get_experiment_by_name(experiment_name)
            self._experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self._client.create_experiment(experiment_name)
            )
        except Exception as exc:
            raise RuntimeError("Unable to initialize the configured MLflow experiment") from exc
        # Context-local run IDs allow parallel worker threads/tasks to share the client.
        self._run_id: ContextVar[str | None] = ContextVar("scientific_mlflow_run", default=None)
        self._run_status: ContextVar[str] = ContextVar(
            "scientific_mlflow_status", default="FINISHED"
        )

    @property
    def run_id(self) -> str | None:
        return self._run_id.get()

    def start_run(self, run_name: str) -> None:
        run = self._client.create_run(self._experiment_id, tags={"mlflow.runName": run_name})
        self._run_id.set(run.info.run_id)
        self._run_status.set("FINISHED")

    def _require_run(self) -> str:
        run_id = self.run_id
        if run_id is None:
            raise RuntimeError("An MLflow run must be started before logging")
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        run_id = self._require_run()
        for key, value in params.items():
            self._client.log_param(run_id, key.replace(".", "_"), value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        run_id = self._require_run()
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

    def end_run(self) -> None:
        run_id = self._require_run()
        try:
            self._client.set_terminated(run_id, status=self._run_status.get())
        finally:
            self._run_id.set(None)
            self._run_status.set("FINISHED")

    def mark_failed(self) -> None:
        """Mark this context's run as failed before the common ``end_run`` path."""
        self._require_run()
        self._run_status.set("FAILED")

    def log_artifact(self, path: str | Path) -> None:
        self._client.log_artifact(self._require_run(), str(path))


def build_tracker(tracking_uri: str, experiment_name: str) -> ExperimentTracker:
    """Keep the default app lightweight; fail at startup when configured but broken."""
    if not tracking_uri:
        return NullTracker()
    return MLflowTracker(tracking_uri, experiment_name)
