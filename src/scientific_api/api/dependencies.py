from ..core.config import settings
from ..adapters.mlflow_adapter import MLflowTracker

# Инициализация при импорте модуля (Application Startup)
# Если settings.mlflow_tracking_uri пуст, трекер будет "заглушкой"
_tracker_instance = MLflowTracker(
    tracking_uri=settings.mlflow_tracking_uri,
    experiment_name=settings.mlflow_experiment_name
)

def get_experiment_tracker():
    """FastAPI dependency provider."""
    return _tracker_instance