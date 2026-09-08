import io
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.api.training import TrainingJob, _run_training_job, training_datasets, training_jobs
from backend.main import app


def _training_csv(rows: int = 80) -> tuple[str, io.BytesIO, str]:
    rng = np.random.default_rng(7)
    score = rng.normal(0, 1, rows)
    age = rng.integers(20, 70, rows)
    city = rng.choice(["北京", "上海", "深圳"], rows)
    label = ((score + age / 100 + rng.normal(0, 0.35, rows)) > 0.85).astype(int)
    df = pd.DataFrame(
        {
            "user_id": range(1, rows + 1),
            "age": age,
            "score": score,
            "city": city,
            "label": label,
        }
    )
    return "training.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 12.0) -> dict:
    deadline = time.time() + timeout
    last_status = {}
    while time.time() < deadline:
        response = client.get(f"/api/training/jobs/{job_id}")
        assert response.status_code == 200
        last_status = response.json()
        if last_status["status"] in {"completed", "failed", "cancelled"}:
            return last_status
        time.sleep(0.25)
    raise AssertionError(f"Training job did not finish in time: {last_status}")


def test_manual_training_full_lifecycle_and_save():
    client = TestClient(app)
    file_name, file_body, content_type = _training_csv()

    upload_response = client.post(
        "/api/training/datasets",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "label"},
    )
    assert upload_response.status_code == 201
    dataset = upload_response.json()
    assert dataset["dataset_id"]
    assert dataset["target_col"] == "label"
    assert dataset["rows"] == 80

    # Simulate a fresh worker/process cache: metadata and dataframe must reload
    # from durable runtime state rather than disappearing with the module dict.
    training_datasets.pop(dataset["dataset_id"])
    restored_dataset = client.get(f"/api/training/datasets/{dataset['dataset_id']}")
    assert restored_dataset.status_code == 200
    assert restored_dataset.json()["rows"] == 80

    config_response = client.get(f"/api/training/datasets/{dataset['dataset_id']}/field-config")
    assert config_response.status_code == 200
    field_config = config_response.json()["fields"]
    assert any(item["field_name"] == "label" and item["field_type"] == "label" for item in field_config)
    assert any(item["field_name"] == "user_id" and item["field_type"] == "key" for item in field_config)

    eda_response = client.post("/api/training/eda", json={"dataset_id": dataset["dataset_id"]})
    assert eda_response.status_code == 200
    assert eda_response.json()["summary"]["rows"] == 80

    job_response = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": dataset["dataset_id"],
            "algorithm": "logistic_regression",
            "params": {"C": 1.0, "solver": "liblinear", "max_iter": 1000},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert job_response.status_code == 201
    job_id = job_response.json()["job_id"]

    final_status = _wait_for_job(client, job_id)
    assert final_status["status"] == "completed"
    assert final_status["progress"] == 100

    training_jobs.pop(job_id)
    result_response = client.get(f"/api/training/jobs/{job_id}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    assert result["algorithm"] == "logistic_regression"
    assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0
    assert result["confusion_matrix"]
    assert isinstance(result["feature_importance"], list)

    save_response = client.post(f"/api/training/jobs/{job_id}/save")
    assert save_response.status_code == 200
    assert save_response.json()["artifact_path"].endswith(".joblib")


def test_training_rejects_invalid_dataset_and_job_ids():
    client = TestClient(app)

    dataset_response = client.get("/api/training/datasets/..%2Fsecret")
    job_response = client.get("/api/training/jobs/..%2Fsecret")

    assert dataset_response.status_code == 400
    assert job_response.status_code == 400


def _upload_dataset(client: TestClient, rows: int = 80) -> str:
    file_name, file_body, content_type = _training_csv(rows)
    response = client.post(
        "/api/training/datasets",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "label"},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def _create_job(client: TestClient, dataset_id: str) -> str:
    response = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": dataset_id,
            "algorithm": "logistic_regression",
            "params": {"solver": "liblinear", "max_iter": 500},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert response.status_code == 201
    return response.json()["job_id"]


def test_preset_cancel_event_marks_job_cancelled():
    """预置 cancel_event 后同步执行训练：任务应在阶段边界转为 cancelled 且无结果。"""
    client = TestClient(app)
    dataset_id = _upload_dataset(client, 60)

    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    job = TrainingJob(
        job_id=job_id,
        dataset_id=dataset_id,
        algorithm="logistic_regression",
        params={"solver": "liblinear", "max_iter": 200},
        status="queued",
        progress=0,
        created_at=now,
        updated_at=now,
    )
    job.cancel_event.set()
    training_jobs[job_id] = job

    _run_training_job(job_id, 0.25, 42)

    assert job.status == "cancelled"
    assert job.result is None


def test_cancel_endpoint_unknown_job_404_and_terminal_noop():
    client = TestClient(app)

    missing = client.post("/api/training/jobs/doesnotexist/cancel")
    assert missing.status_code == 404

    dataset_id = _upload_dataset(client, 60)
    job_id = _create_job(client, dataset_id)
    final_status = _wait_for_job(client, job_id)
    assert final_status["status"] == "completed"

    cancel_response = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "completed"


def test_training_result_feature_importance_uses_pipeline_names():
    """Pipeline 化后特征重要性必须使用展开后的列名（num__/cat__ 前缀）。"""
    client = TestClient(app)
    dataset_id = _upload_dataset(client, 80)
    job_id = _create_job(client, dataset_id)
    final_status = _wait_for_job(client, job_id)
    assert final_status["status"] == "completed"

    result = client.get(f"/api/training/jobs/{job_id}/result").json()
    names = [row["feature"] for row in result["feature_importance"]]
    assert names, "feature_importance 不应为空"
    assert any(name.startswith("num__") for name in names)
    assert any(name.startswith("cat__city_") for name in names)
