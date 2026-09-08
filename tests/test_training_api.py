import io
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import backend.api.training as training_api
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


def _upload_regression_dataset(client: TestClient, rows: int = 60) -> str:
    x = np.linspace(-5.0, 5.0, rows)
    frame = pd.DataFrame({"x": x, "price": 3.0 * x + 7.0})
    response = client.post(
        "/api/training/datasets",
        files={"file": ("prices.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "auto"},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def _upload_low_cardinality_numeric_dataset(client: TestClient, *, problem_type: str = "auto") -> str:
    frame = pd.DataFrame({"x": range(60), "score": np.tile([0, 1, 2], 20)})
    response = client.post(
        "/api/training/datasets",
        files={"file": ("scores.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "score", "problem_type": problem_type},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_training_capabilities_expose_regression():
    """The manual-training client can discover the canonical regression contract."""
    response = TestClient(app).get("/api/training/capabilities")

    assert response.status_code == 200
    assert response.json()["problem_types"]["regression"] == {
        "algorithms": ["lightgbm", "xgboost", "catboost", "random_forest", "linear_regression"],
        "metrics": ["rmse", "mae", "r2"],
        "default_metric": "rmse",
        "metric_directions": {"rmse": "minimize", "mae": "minimize", "r2": "maximize"},
    }


def test_training_upload_auto_detects_regression():
    """A continuous numeric target is retained as a regression dataset snapshot."""
    client = TestClient(app)
    df = pd.DataFrame({"x": range(40), "price": np.linspace(10.1, 80.9, 40)})

    response = client.post(
        "/api/training/datasets",
        files={"file": ("prices.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "auto"},
    )

    assert response.status_code == 201
    assert response.json()["problem_spec"]["resolved_type"] == "regression"


def test_training_upload_rejects_unknown_problem_type_with_stable_detail():
    """Unsupported task labels are rejected by the shared contract, not coerced locally."""
    client = TestClient(app)
    df = pd.DataFrame({"x": range(4), "price": [1.1, 2.2, 3.3, 4.4]})

    response = client.post(
        "/api/training/datasets",
        files={"file": ("prices.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "clustering"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "unsupported_problem_type",
        "field": "problem_type",
        "message": "不支持的任务类型：clustering",
        "received": "clustering",
        "allowed": ["auto", "binary_classification", "multiclass_classification", "regression"],
    }


def test_restoring_legacy_job_without_dataset_uses_classification_compatibility():
    """A pre-problem-spec job remains inspectable even after its dataset has expired."""
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    training_api.runtime_state.put(
        training_api.JOB_NAMESPACE,
        job_id,
        {
            "job_id": job_id,
            "dataset_id": "expireddataset",
            "algorithm": "logistic_regression",
            "params": {},
            "status": "completed",
            "progress": 100,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_path": None,
            "test_size": 0.3,
            "random_state": 42,
            "has_result": True,
        },
    )
    training_api.runtime_state.save_blob(training_api.RESULT_BLOB_NAMESPACE, job_id, {"job_id": job_id, "metrics": {}})
    try:
        restored = training_api._restore_job(job_id)
    finally:
        training_api.runtime_state.delete(training_api.JOB_NAMESPACE, job_id, (training_api.RESULT_BLOB_NAMESPACE,))

    assert restored is not None
    assert restored.problem_spec["resolved_type"] == "binary_classification"
    assert restored.problem_spec["source"] == "legacy_compatibility"
    assert restored.result is not None
    assert restored.result["problem_spec"]["resolved_type"] == "binary_classification"


def test_job_without_problem_type_reuses_explicit_dataset_override():
    """Omitting a job override retains the task type selected while uploading the dataset."""
    client = TestClient(app)
    dataset_id = _upload_low_cardinality_numeric_dataset(client, problem_type="regression")

    created = client.post(
        "/api/training/jobs",
        json={"dataset_id": dataset_id, "algorithm": "linear_regression", "params": {}, "test_size": 0.25},
    )

    assert created.status_code == 201
    assert created.json()["problem_spec"]["requested_type"] == "regression"
    assert created.json()["problem_spec"]["resolved_type"] == "regression"
    assert _wait_for_job(client, created.json()["job_id"])["status"] == "completed"


def test_training_rejects_incompatible_algorithm_before_enqueue():
    """A classification-only algorithm never creates a job for a regression target."""
    client = TestClient(app)
    regression_dataset_id = _upload_regression_dataset(client)

    response = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": regression_dataset_id,
            "problem_type": "regression",
            "algorithm": "logistic_regression",
            "params": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "incompatible_algorithm",
        "field": "algorithm",
        "message": "算法 logistic_regression 不适用于 regression",
        "received": "logistic_regression",
        "allowed": ["lightgbm", "xgboost", "catboost", "random_forest", "linear_regression"],
    }


def test_manual_multiclass_training_keeps_default_threshold_calibration():
    """Classification clients retain the default threshold payload for multiclass jobs."""
    client = TestClient(app)
    dataset_id = _upload_low_cardinality_numeric_dataset(client)
    created = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": dataset_id,
            "algorithm": "logistic_regression",
            "params": {"solver": "lbfgs"},
            "test_size": 0.25,
        },
    )

    assert created.status_code == 201
    job = _wait_for_job(client, created.json()["job_id"])
    assert job["status"] == "completed"
    result = client.get(f"/api/training/jobs/{job['job_id']}/result").json()
    expected_calibration = {"threshold": 0.5, "source": "default", "calibration_size": 0}
    assert result["threshold_calibration"] == expected_calibration
    saved = client.post(f"/api/training/jobs/{job['job_id']}/save")
    artifact = joblib.load(Path(__file__).parents[1] / saved.json()["artifact_path"])
    assert artifact["decision_threshold"] == 0.5
    assert artifact["threshold_calibration"] == expected_calibration


def test_manual_linear_regression_completes_with_regression_metrics():
    """Manual regression trains continuously and saves a schema-v3 regression artifact."""
    client = TestClient(app)
    regression_dataset_id = _upload_regression_dataset(client)

    created = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": regression_dataset_id,
            "problem_type": "regression",
            "algorithm": "linear_regression",
            "params": {},
            "test_size": 0.25,
        },
    )

    assert created.status_code == 201
    assert created.json()["problem_spec"]["resolved_type"] == "regression"
    job = _wait_for_job(client, created.json()["job_id"])
    assert job["status"] == "completed"
    assert job["problem_spec"]["resolved_type"] == "regression"

    result_response = client.get(f"/api/training/jobs/{job['job_id']}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    assert set(result["metrics"]) == {"rmse", "mae", "r2"}
    assert result["metrics"]["rmse"] < 1e-10
    assert result["metrics"]["mae"] < 1e-10
    assert result["metrics"]["r2"] > 0.999999999
    assert result["confusion_matrix"] is None
    assert result["threshold_calibration"] is None
    assert result["problem_spec"]["resolved_type"] == "regression"

    saved = client.post(f"/api/training/jobs/{job['job_id']}/save")
    assert saved.status_code == 200
    artifact = joblib.load(Path(__file__).parents[1] / saved.json()["artifact_path"])
    assert artifact["artifact_schema_version"] == 3
    assert artifact["problem_type"] == "regression"
    assert artifact["problem_spec"]["resolved_type"] == "regression"
    assert artifact["metric_directions"] == {"rmse": "minimize", "mae": "minimize", "r2": "maximize"}


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


def _persisted_job_for_missing_dataset() -> TrainingJob:
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    job = TrainingJob(
        job_id=job_id,
        dataset_id="expireddataset",
        algorithm="logistic_regression",
        params={},
        status="queued",
        progress=0,
        created_at=now,
        updated_at=now,
        problem_spec=training_api._legacy_classification_problem_spec(),
    )
    training_jobs[job_id] = job
    training_api._persist_job(job)
    return job


def test_worker_persists_missing_dataset_failure():
    """A dataset disappearance becomes a durable failed job rather than an escaping worker error."""
    job = _persisted_job_for_missing_dataset()
    try:
        _run_training_job(job.job_id, 0.25, 42)
        training_jobs.pop(job.job_id)
        restored = training_api._get_job(job.job_id)
    finally:
        training_jobs.pop(job.job_id, None)
        training_api.runtime_state.delete(training_api.JOB_NAMESPACE, job.job_id, (training_api.RESULT_BLOB_NAMESPACE,))

    assert restored.status == "failed"
    assert restored.error == "404: 未找到训练数据集"


def test_training_lifecycle_logs_safe_job_context(caplog):
    """Completion and failure logs identify the job context without emitting dataset contents."""
    client = TestClient(app)
    dataset_id = _upload_regression_dataset(client)
    dataset = training_api._get_dataset(dataset_id)
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    completed_job = TrainingJob(
        job_id=job_id,
        dataset_id=dataset_id,
        algorithm="linear_regression",
        params={},
        status="queued",
        progress=0,
        created_at=now,
        updated_at=now,
        problem_spec=dataset.problem_spec,
    )
    training_jobs[job_id] = completed_job
    failed_job = _persisted_job_for_missing_dataset()
    caplog.set_level(logging.INFO, logger="backend.api.training")
    try:
        _run_training_job(completed_job.job_id, 0.25, 42)
        _run_training_job(failed_job.job_id, 0.25, 42)
    finally:
        training_jobs.pop(completed_job.job_id, None)
        training_jobs.pop(failed_job.job_id, None)
        training_api.runtime_state.delete(training_api.JOB_NAMESPACE, completed_job.job_id, (training_api.RESULT_BLOB_NAMESPACE,))
        training_api.runtime_state.delete(training_api.JOB_NAMESPACE, failed_job.job_id, (training_api.RESULT_BLOB_NAMESPACE,))

    records = [record.getMessage() for record in caplog.records]
    assert any(
        f"job_id={completed_job.job_id}" in message
        and "problem_type=regression" in message
        and "algorithm=linear_regression" in message
        and "metric=rmse" in message
        for message in records
    )
    assert any(
        f"job_id={failed_job.job_id}" in message
        and "problem_type=binary_classification" in message
        and "algorithm=logistic_regression" in message
        and "metric=roc_auc" in message
        for message in records
    )


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
