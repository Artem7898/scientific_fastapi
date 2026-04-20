import time
from fastapi.testclient import TestClient
from scientific_api.main import app


def test_simulation_async_workflow():
    """
    Tests the complete asynchronous workflow.
    'with TestClient' is MANDATORY to trigger FastAPI lifespan (startup/shutdown).
    """
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: with TestClient(app) as client:
    with TestClient(app) as client:
        payload = {
            "initial_temperature": 300.0,
            "num_particles": 1000,
            "time_step": 0.001
        }

        # Шаг 1: Отправляем задачу
        response = client.post("/api/v1/simulations/monte-carlo", json=payload)

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert "check_url" in data

        task_id = data["task_id"]

        # Шаг 2: Опрашиваем статус
        result = None
        for _ in range(10):
            poll_response = client.get(f"/api/v1/simulations/status/{task_id}")

            if poll_response.status_code == 200:
                result = poll_response.json()
                if result.get("status") == "completed":
                    break

            time.sleep(0.05)

        # Шаг 3: Проверяем научные данные
        assert result is not None, "Task did not complete in time"
        assert result["status"] == "completed"
        assert isinstance(result["final_temperature"], float)
        assert isinstance(result["total_energy"], float)


def test_simulation_rejects_string_temperature():
    """Test that strict types prevent implicit string-to-float coercion."""
    with TestClient(app) as client:
        payload = {
            "initial_temperature": "300",
            "num_particles": 1000
        }
        response = client.post("/api/v1/simulations/monte-carlo", json=payload)
        assert response.status_code == 422


def test_simulation_rejects_negative_temperature():
    """Test physical constraints validation."""
    with TestClient(app) as client:
        payload = {
            "initial_temperature": -10.0,
            "num_particles": 1000
        }
        response = client.post("/api/v1/simulations/monte-carlo", json=payload)
        assert response.status_code == 422