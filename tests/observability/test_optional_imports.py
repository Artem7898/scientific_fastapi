"""The default runtime must not need monitoring or tracking extras."""

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scientific_api.adapters.mlflow_adapter import build_tracker
from scientific_api.core.experiment_tracker import NullTracker
from scientific_api.core.telemetry import Telemetry


def test_disabled_adapters_do_not_import_optional_dependencies() -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    script = """
import sys
from scientific_api.core.telemetry import Telemetry
from scientific_api.adapters.mlflow_adapter import build_tracker
t = Telemetry()
with t.span('request'):
    assert t.inject() == {}
t.observe_queue_wait(1.0)
t.observe_execution(2.0)
t.record_task('completed')
t.record_http('GET', '/health', 200, 0.1)
t.shutdown()
tracker = build_tracker('', 'test')
tracker.start_run('test')
tracker.log_params({'seed': 42})
tracker.log_metrics({'temperature': 1.0})
tracker.end_run()
assert not any(name.split('.')[0] in {'opentelemetry', 'mlflow'} for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_missing_telemetry_extra_has_an_actionable_startup_error(monkeypatch) -> None:
    original = builtins.__import__

    def no_otel(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("optional dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_otel)
    with pytest.raises(RuntimeError, match=r"scientific-api\[observability\]"):
        Telemetry(enabled=True)


def test_missing_tracking_extra_has_an_actionable_startup_error(monkeypatch) -> None:
    def missing_module(name):
        raise ImportError("optional dependency absent")

    monkeypatch.setattr("scientific_api.adapters.mlflow_adapter.import_module", missing_module)
    assert isinstance(build_tracker("", "test"), NullTracker)
    with pytest.raises(RuntimeError, match=r"scientific-api\[tracking\]"):
        build_tracker("http://tracking.example", "test")
