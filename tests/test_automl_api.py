import io
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.api.automl import AutoMLJob, _run_job, automl_jobs
from backend.main import app


def _csv_upload(rows: int = 64) -> tuple[str, io.BytesIO, str]:
    rng = np.random.default_rng(42)
    age = rng.integers(18, 70, rows)
    score = rng.normal(0, 1, rows)
    income = rng.normal(50000, 8000, rows)
    label = ((score + age / 80 + rng.normal(0, 0.4, rows)) > 0.85).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": range(1, rows + 1),
            "age": age,
            "score": score,
            "income": income,
            "label": label,
        }
    )
    return "sample.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def _wait_for_completion(client: TestClient, job_id: str, timeout: float = 12.0) -> dict:
    deadline = time.time() + timeout
    last_status: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/automl/jobs/{job_id}")
        assert response.status_code == 200
        last_status = response.json()
        if last_status["status"] in {"completed", "failed", "cancelled"}:
            return last_status
        time.sleep(0.25)
    raise AssertionError(f"AutoML job did not finish in time: {last_status}")


def test_automl_job_runs_full_lifecycle_and_saves_result():
    client = TestClient(app)
    file_name, file_body, content_type = _csv_upload()

    create_response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "n_trials": "2",
            "cv_folds": "2",
            "candidate_models": '["logistic_regression", "random_forest"]',
            "optimization_metric": "roc_auc",
            "use_llm": "false",
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["job_id"]
    assert created["status"] in {"queued", "running"}
    assert created["decision_summary"]["target_col"] == "label"
    assert "logistic_regression" in created["decision_summary"]["candidate_models"]

    final_status = _wait_for_completion(client, created["job_id"])
    assert final_status["status"] == "completed"
    assert final_status["progress"] == 100

    automl_jobs.pop(created["job_id"])
    result_response = client.get(f"/api/automl/jobs/{created['job_id']}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    assert result["best_model"] in {"logistic_regression", "random_forest"}
    assert isinstance(result["best_params"], dict)
    assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0
    assert len(result["trial_history"]) >= 1

    save_response = client.post(f"/api/automl/jobs/{created['job_id']}/save")
    assert save_response.status_code == 200
    saved = save_response.json()
    assert saved["status"] == "saved"
    assert saved["artifact_path"].endswith(".joblib")
    assert ".." not in saved["artifact_path"]


def test_automl_create_job_rejects_missing_target_column():
    client = TestClient(app)
    file_name, file_body, content_type = _csv_upload()

    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "not_a_column", "n_trials": "1"},
    )

    assert response.status_code == 400
    assert "目标列" in response.json()["detail"]


def test_automl_job_id_validation_blocks_path_traversal():
    client = TestClient(app)

    response = client.get("/api/automl/jobs/..%2Fsecret")

    assert response.status_code == 400
    assert "job_id" in response.json()["detail"]


def _multiclass_csv(rows: int = 90) -> tuple[str, io.BytesIO, str]:
    rng = np.random.default_rng(11)
    score = rng.normal(0, 1, rows)
    label = pd.cut(score, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, rows),
            "score": score,
            "label": label,
        }
    )
    return "multiclass.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def _categorical_csv(rows: int = 80) -> tuple[str, io.BytesIO, str]:
    rng = np.random.default_rng(5)
    score = rng.normal(0, 1, rows)
    city = rng.choice(["北京", "上海", "深圳"], rows)
    label = ((score + rng.normal(0, 0.4, rows)) > 0.6).astype(int)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, rows),
            "score": score,
            "city": city,
            "label": label,
        }
    )
    return "categorical.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def test_automl_multiclass_lightgbm_completes():
    """三分类 + lightgbm 端到端完成（锁 num_class 修复与 Pipeline-in-CV 迁移）。"""
    client = TestClient(app)
    file_name, file_body, content_type = _multiclass_csv()

    create_response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "n_trials": "2",
            "cv_folds": "2",
            "candidate_models": '["lightgbm"]',
            "optimization_metric": "roc_auc",
            "use_llm": "false",
        },
    )
    assert create_response.status_code == 201
    job_id = create_response.json()["job_id"]

    final_status = _wait_for_completion(client, job_id, timeout=60.0)
    assert final_status["status"] == "completed"

    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    assert result["best_model"] == "lightgbm"
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


def test_automl_result_includes_feature_importance_with_pipeline_names():
    """Pipeline 化后 AutoML 结果必须带展开列名（num__/cat__ 前缀）的特征重要性。"""
    client = TestClient(app)
    file_name, file_body, content_type = _categorical_csv()

    create_response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "n_trials": "1",
            "cv_folds": "2",
            "candidate_models": '["random_forest"]',
            "optimization_metric": "accuracy",
            "use_llm": "false",
        },
    )
    assert create_response.status_code == 201
    job_id = create_response.json()["job_id"]

    final_status = _wait_for_completion(client, job_id, timeout=60.0)
    assert final_status["status"] == "completed"

    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    names = [row["feature"] for row in result["feature_importance"]]
    assert names, "feature_importance 不应为空"
    assert any(name.startswith("num__") for name in names)
    assert any(name.startswith("cat__city_") for name in names)


def test_automl_preset_cancel_event_marks_job_cancelled():
    """预置 cancel_event 后同步执行：搜索应停止并转为 cancelled 且无结果。"""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, 60),
            "score": rng.normal(0, 1, 60),
            "label": rng.integers(0, 2, 60),
        }
    )
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    job = AutoMLJob(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        n_trials=5,
        cv_folds=2,
        optimization_metric="roc_auc",
        target_col="label",
        decision_summary={"key_columns": [], "candidate_models": ["logistic_regression"]},
    )
    job.cancel_event.set()
    automl_jobs[job_id] = job

    _run_job(job_id, df, ["logistic_regression"])

    assert job.status == "cancelled"
    assert job.result is None
