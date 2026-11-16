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


def test_retrain_endpoint_schedules_real_retrain_job():
    from backend.scheduler import scheduler

    client = TestClient(app)
    response = client.post("/api/mlops/retrain", params={"model_version": "manual:ghost", "force": True})

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id == "manual_retrain_manual_ghost"
    assert scheduler.get_job(job_id) is not None
    scheduler.remove_job(job_id)
    assert scheduler.get_job(job_id) is None
