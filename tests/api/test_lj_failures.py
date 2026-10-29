"""Physical/workload failures expose useful bounded messages without internal details."""

import pytest
from fastapi.testclient import TestClient

from scientific_api.core import queue_manager
from scientific_api.main import create_app
from scientific_api.models.lj_forces import (
    NeighborCapacityError,
    PairBudgetExceeded,
    ParticleOverlapError,
)

from .test_lj import PAYLOAD, terminal
from .test_simulations import configuration


@pytest.mark.parametrize(
    "exception,expected",
    [
        (ParticleOverlapError, "within 0.5 sigma"),
        (PairBudgetExceeded, "pair-work budget exhausted"),
        (NeighborCapacityError, "neighbor capacity exhausted"),
    ],
)
def test_known_lj_failure_is_terminal_and_preserves_actionable_reason(
    tmp_path, monkeypatch, exception, expected
):
    def fail(*args, **kwargs):
        raise exception("internal-sensitive-location-must-not-appear")

    monkeypatch.setattr(queue_manager, "run_simulation", fail)
    with TestClient(create_app(configuration(tmp_path))) as client:
        accepted = client.post("/api/v1/simulations/lennard-jones", json=PAYLOAD)
        assert accepted.status_code == 202
        result = terminal(client, accepted.json()["check_url"])
        assert result["status"] == "failed"
        assert expected in result["error"]
        assert "internal-sensitive" not in result["error"]
        response = client.get(f"/api/v1/simulations/{result['task_id']}/artifacts/observables")
        assert response.status_code == 409
