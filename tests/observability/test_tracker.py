from concurrent.futures import ThreadPoolExecutor
from itertools import count
from threading import Barrier
from types import SimpleNamespace

import pytest

from scientific_api.adapters.mlflow_adapter import build_tracker


def test_tracking_client_isolated_runs_in_parallel_workers(monkeypatch) -> None:
    sequence = count()
    calls = []

    class Client:
        def __init__(self, tracking_uri):
            assert tracking_uri == "https://mlflow.example"

        def get_experiment_by_name(self, name):
            assert name == "science"
            return SimpleNamespace(experiment_id="exp")

        def create_run(self, experiment_id, tags):
            assert experiment_id == "exp"
            run_id = str(next(sequence))
            calls.append((run_id, "start", tags["mlflow.runName"]))
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

        def log_param(self, run_id, key, value):
            calls.append((run_id, key, value))

        def log_metric(self, run_id, key, value):
            calls.append((run_id, key, value))

        def set_terminated(self, run_id, status):
            calls.append((run_id, "end", status))

    monkeypatch.setattr(
        "scientific_api.adapters.mlflow_adapter.import_module",
        lambda name: SimpleNamespace(MlflowClient=Client),
    )
    tracker = build_tracker("https://mlflow.example", "science")
    barrier = Barrier(2)

    def run(seed):
        tracker.start_run(f"seed-{seed}")
        barrier.wait(timeout=5)
        tracker.log_params({"run.seed": seed})
        tracker.log_metrics({"temperature": float(seed)})
        if seed == 2:
            tracker.mark_failed()
        tracker.end_run()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, seed) for seed in (1, 2)]
        for future in futures:
            future.result(timeout=5)
    for run_id in ("0", "1"):
        events = {key: value for current, key, value in calls if current == run_id}
        seed = int(events["start"].split("-")[1])
        assert events["run_seed"] == seed
        assert events["temperature"] == float(seed)
        assert events["end"] == ("FAILED" if seed == 2 else "FINISHED")


def test_configured_tracking_failure_is_not_silently_disabled(monkeypatch) -> None:
    class BrokenClient:
        def __init__(self, tracking_uri):
            raise ConnectionError("unreachable")

    monkeypatch.setattr(
        "scientific_api.adapters.mlflow_adapter.import_module",
        lambda name: SimpleNamespace(MlflowClient=BrokenClient),
    )
    with pytest.raises(RuntimeError, match="initialize the configured MLflow experiment"):
        build_tracker("https://mlflow.example", "science")
