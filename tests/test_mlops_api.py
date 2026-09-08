from fastapi.testclient import TestClient

from backend.main import app


def test_mlops_status_handles_scheduler_without_global_next_run_time():
    client = TestClient(app)

    response = client.get("/api/mlops/status")

    assert response.status_code == 200
    payload = response.json()
    assert "scheduler_running" in payload
    assert "jobs" in payload
    assert "next_run" in payload
