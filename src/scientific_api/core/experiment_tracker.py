from abc import abstractmethod
from pathlib import Path
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    """
    Protocol defining the contract for scientific experiment tracking.
    Using Protocol instead of ABC allows for structural subtyping (duck typing).
    """

    @abstractmethod
    def start_run(self, run_name: str) -> None:
        """Initialize a new experiment run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log input parameters (e.g., temperature, num_particles)."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log output metrics (e.g., final temperature, total energy)."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """Finalize the experiment run."""
        ...


class NullTracker:
    """The default tracker has no optional imports and performs no I/O."""

    run_id: str | None = None

    def start_run(self, run_name: str) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def end_run(self) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass

    def mark_failed(self) -> None:
        pass
