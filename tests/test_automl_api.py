import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.api.automl as automl_api
from backend.api.automl import AutoMLJob, _run_job, automl_jobs
from backend.main import app
from backend.runtime_state import RuntimeStateStore


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


def _regression_csv(rows: int = 80) -> tuple[str, io.BytesIO, str]:
    x = np.linspace(-10.0, 10.0, rows)
    frame = pd.DataFrame({"x": x, "price": 4.5 * x + 12.0})
    return "regression.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv"


def _create_regression_automl_job(client: TestClient, metric: str = "rmse") -> str:
    file_name, file_body, content_type = _regression_csv()
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "price",
            "problem_type": "auto",
            "optimization_metric": metric,
            "candidate_models": '["linear_regression"]',
            "n_trials": 1,
            "cv_folds": 3,
        },
    )
    assert response.status_code == 201
    return response.json()["job_id"]


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


def test_automl_regression_auto_metric_and_problem_spec():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=80)
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "price",
            "problem_type": "auto",
            "optimization_metric": "auto",
            "candidate_models": '["linear_regression"]',
            "n_trials": 1,
            "cv_folds": 3,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["problem_spec"]["resolved_type"] == "regression"
    assert payload["optimization_metric"] == "rmse"


def test_automl_rejects_regression_with_classification_metric():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=40)
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={"problem_type": "regression", "optimization_metric": "accuracy"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "optimization_metric"


def test_automl_rejects_incompatible_candidate_before_creating_job():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=40)
    existing_ids = {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)}

    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "problem_type": "regression",
            "optimization_metric": "rmse",
            "candidate_models": '["logistic_regression"]',
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
    assert {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)} == existing_ids


def test_automl_rejects_infeasible_cv_before_creating_job():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=5)
    existing_ids = {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)}

    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "problem_type": "regression",
            "optimization_metric": "rmse",
            "candidate_models": '["linear_regression"]',
            "cv_folds": 5,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "cv_folds"
    assert {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)} == existing_ids


def test_automl_rejects_rare_class_with_stable_detail_before_creating_job():
    client = TestClient(app, raise_server_exceptions=False)
    frame = pd.DataFrame({"x": range(20), "label": [0] * 19 + [1]})
    existing_persisted = {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)}
    existing_memory = set(automl_jobs)
    input_blob_dir = automl_api.runtime_state.blob_dir / automl_api.INPUT_BLOB_NAMESPACE
    existing_input_blobs = {path.name for path in input_blob_dir.glob("*.joblib")}

    response = client.post(
        "/api/automl/jobs",
        files={"file": ("rare.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={
            "target_col": "label",
            "problem_type": "binary_classification",
            "optimization_metric": "accuracy",
            "candidate_models": '["logistic_regression"]',
            "cv_folds": 2,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "insufficient_classification_split",
        "field": "test_size",
        "message": "训练集和测试集必须覆盖全部类别",
    }
    assert {item["job_id"] for item in automl_api.runtime_state.list(automl_api.JOB_NAMESPACE)} == existing_persisted
    assert set(automl_jobs) == existing_memory
    assert {path.name for path in input_blob_dir.glob("*.joblib")} == existing_input_blobs


def test_automl_regression_completes_with_raw_rmse():
    client = TestClient(app)
    job_id = _create_regression_automl_job(client, metric="rmse")

    final = _wait_for_completion(client, job_id)

    assert final["status"] == "completed"
    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    assert result["best_value"] >= 0
    assert set(result["metrics"]) == {"rmse", "mae", "r2"}
    assert result["threshold_calibration"] is None
    assert result["problem_spec"]["capabilities"]["metric_directions"]["rmse"] == "minimize"
    assert all(event["value"] >= 0 for event in result["trial_history"])


@pytest.mark.parametrize(
    ("metric", "direction", "expected_value"),
    [("mae", "minimize", 0.0), ("r2", "maximize", 1.0)],
)
def test_automl_regression_uses_raw_metric_direction(metric, direction, expected_value):
    client = TestClient(app)
    job_id = _create_regression_automl_job(client, metric=metric)

    final = _wait_for_completion(client, job_id)

    assert final["status"] == "completed"
    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    restored_spec = automl_api.ProblemSpec.from_dict(result["problem_spec"])
    assert automl_api._optimization_direction(restored_spec, metric) == direction
    assert result["best_value"] >= 0
    assert result["best_value"] == pytest.approx(expected_value, abs=1e-12)
    assert all(event["value"] >= 0 for event in result["trial_history"])
    assert all(event["value"] == pytest.approx(expected_value, abs=1e-12) for event in result["trial_history"])


@pytest.mark.parametrize(
    ("resolved_type", "expected_test_indices"),
    [
        ("regression", [[0, 1], [2, 5], [3, 4]]),
        ("binary_classification", [[0, 5], [1, 3], [2, 4]]),
    ],
)
def test_automl_cross_validator_is_task_aware_and_deterministic(resolved_type, expected_test_indices):
    X = np.arange(6).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])

    splitter = automl_api._cross_validator(resolved_type, cv_folds=3)
    actual_test_indices = [test_indices.tolist() for _, test_indices in splitter.split(X, y)]

    assert actual_test_indices == expected_test_indices


def test_automl_reports_non_finite_regression_metric_as_failed_job():
    client = TestClient(app)
    frame = pd.DataFrame({"x": range(8), "price": [1.0] * 7 + [2.0]})
    response = client.post(
        "/api/automl/jobs",
        files={"file": ("constant-test-target.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={
            "target_col": "price",
            "problem_type": "regression",
            "optimization_metric": "mae",
            "candidate_models": '["linear_regression"]',
            "n_trials": 1,
            "cv_folds": 2,
        },
    )
    assert response.status_code == 201

    final = _wait_for_completion(client, response.json()["job_id"])

    assert final["status"] == "failed"
    assert final["error"] == "R² 无法计算：测试集目标值没有方差，无法生成有限分数"
    result = client.get(f"/api/automl/jobs/{final['job_id']}/result")
    assert result.status_code == 400
    assert result.json()["detail"] == "任务尚未完成，当前状态：failed"


def test_automl_regression_snapshot_events_and_schema_v3_artifact_survive_reload():
    client = TestClient(app)
    job_id = _create_regression_automl_job(client)
    assert _wait_for_completion(client, job_id)["status"] == "completed"

    with client.stream("GET", f"/api/automl/jobs/{job_id}/stream") as response:
        events = [json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ")]
    assert events
    assert all(event["problem_spec"]["resolved_type"] == "regression" for event in events)

    automl_jobs.pop(job_id)
    restored = client.get(f"/api/automl/jobs/{job_id}")
    assert restored.status_code == 200
    assert restored.json()["problem_spec"]["resolved_type"] == "regression"

    saved = client.post(f"/api/automl/jobs/{job_id}/save")
    assert saved.status_code == 200
    artifact_path = automl_api.PROJECT_ROOT / saved.json()["artifact_path"]
    artifact = joblib.load(artifact_path)
    assert artifact["artifact_schema_version"] == 3
    assert artifact["problem_type"] == "regression"
    assert artifact["problem_spec"]["resolved_type"] == "regression"
    assert artifact["metric_directions"] == {"rmse": "minimize", "mae": "minimize", "r2": "maximize"}


def test_restoring_legacy_automl_job_has_classification_problem_spec():
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    automl_api.runtime_state.put(
        automl_api.JOB_NAMESPACE,
        job_id,
        {
            "job_id": job_id,
            "status": "failed",
            "created_at": now,
            "updated_at": now,
            "n_trials": 1,
            "cv_folds": 2,
            "optimization_metric": "roc_auc",
            "target_col": "label",
            "decision_summary": {"key_columns": [], "candidate_models": ["logistic_regression"]},
            "candidate_models": ["logistic_regression"],
            "progress": 0,
            "current_trial": 0,
            "error": "interrupted",
            "artifact_path": None,
            "events": [],
            "has_result": False,
        },
    )
    try:
        restored = automl_api._restore_job(job_id)
    finally:
        automl_api.runtime_state.delete(automl_api.JOB_NAMESPACE, job_id)

    assert restored is not None
    assert restored.problem_spec["resolved_type"] == "binary_classification"
    assert restored.problem_spec["source"] == "legacy_compatibility"


def test_restart_recovery_event_retains_problem_spec(tmp_path, monkeypatch):
    store = RuntimeStateStore(tmp_path / "automl-recovery")
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    problem_spec = {
        "schema_version": 1,
        "requested_type": "regression",
        "resolved_type": "regression",
        "source": "user_override",
        "reason": "user_selected_regression",
        "explanation": "用户显式选择了regression",
        "target": {
            "column": "price",
            "dtype": "float64",
            "valid_rows": 20,
            "unique_count": 20,
            "unique_ratio": 1.0,
            "integer_like": False,
        },
        "capabilities": {
            "algorithms": ["lightgbm", "xgboost", "catboost", "random_forest", "linear_regression"],
            "metrics": ["rmse", "mae", "r2"],
            "default_metric": "rmse",
            "metric_directions": {"rmse": "minimize", "mae": "minimize", "r2": "maximize"},
        },
    }
    store.put(
        automl_api.JOB_NAMESPACE,
        job_id,
        {
            "job_id": job_id,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "n_trials": 1,
            "cv_folds": 3,
            "optimization_metric": "rmse",
            "target_col": "price",
            "decision_summary": {"key_columns": [], "candidate_models": ["linear_regression"]},
            "problem_spec": problem_spec,
            "candidate_models": ["linear_regression"],
            "progress": 50,
            "current_trial": 0,
            "error": None,
            "artifact_path": None,
            "events": [],
            "has_result": False,
        },
    )
    monkeypatch.setattr(automl_api, "runtime_state", store)

    assert automl_api.recover_interrupted_automl_jobs() == 1
    recovered = store.get(automl_api.JOB_NAMESPACE, job_id)

    assert recovered is not None
    assert recovered["status"] == "failed"
    assert recovered["events"][-1]["problem_spec"]["resolved_type"] == "regression"


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


def test_automl_early_stopping_halts_search():
    """大 min_improvement + 1 轮早停：AUC 不可能再改进 0.5，搜索应在第 2 个 Trial 后停止。"""
    client = TestClient(app)
    file_name, file_body, content_type = _csv_upload()
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "n_trials": "10",
            "cv_folds": "2",
            "candidate_models": '["logistic_regression"]',
            "optimization_metric": "roc_auc",
            "use_llm": "false",
            "early_stopping_rounds": "1",
            "min_improvement": "0.5",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]

    final = _wait_for_completion(client, job_id)

    assert final["status"] == "completed"
    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    assert result["early_stopped"] is True
    assert 0 < len(result["trial_history"]) <= 2


def test_automl_rejects_sampling_for_non_binary_task():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=40)

    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "problem_type": "regression",
            "optimization_metric": "rmse",
            "enable_sampling": "true",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "enable_sampling"


def _imbalanced_csv(rows: int = 200, positives: int = 20) -> tuple[str, io.BytesIO, str]:
    rng = np.random.default_rng(7)
    score = rng.normal(0, 1, rows)
    label = np.zeros(rows, dtype=int)
    label[rng.choice(rows, size=positives, replace=False)] = 1
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, rows),
            "score": score,
            "label": label,
        }
    )
    return "imbalanced.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def test_automl_sampling_reduces_training_rows_only():
    """智能采样：训练集负样本被缩减，测试集保持原始分布，任务正常完成。"""
    client = TestClient(app)
    file_name, file_body, content_type = _imbalanced_csv()
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "problem_type": "binary_classification",
            "n_trials": "1",
            "cv_folds": "2",
            "candidate_models": '["logistic_regression"]',
            "optimization_metric": "roc_auc",
            "use_llm": "false",
            "enable_sampling": "true",
            "sampling_negative_ratio": "0.5",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]

    final = _wait_for_completion(client, job_id)

    assert final["status"] == "completed"
    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    sampling = result["sampling"]
    assert sampling["applied"] is True
    assert sampling["sampled_rows"] < sampling["original_rows"]
    assert sampling["kept_negative_rows"] < sampling["negative_rows"]
    assert sampling["positive_rows"] > 0


def test_automl_feature_selection_reduces_features_and_artifact_predicts():
    """特征选择：选择的特征数减少，selector 随 artifact 持久化且推理一致。"""
    from backend.services.datasets import predict_with_artifact

    client = TestClient(app)
    file_name, file_body, content_type = _categorical_csv()
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={
            "target_col": "label",
            "n_trials": "1",
            "cv_folds": "2",
            "candidate_models": '["random_forest"]',
            "optimization_metric": "accuracy",
            "use_llm": "false",
            "enable_feature_selection": "true",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]

    final = _wait_for_completion(client, job_id, timeout=60.0)

    assert final["status"] == "completed"
    result = client.get(f"/api/automl/jobs/{job_id}/result").json()
    selection = result["feature_selection"]
    assert selection["applied"] is True
    assert 0 < selection["selected_feature_count"] < selection["original_feature_count"]

    saved = client.post(f"/api/automl/jobs/{job_id}/save")
    assert saved.status_code == 200
    artifact_path = automl_api.PROJECT_ROOT / saved.json()["artifact_path"]
    artifact = joblib.load(artifact_path)
    assert artifact["feature_selector"] is not None

    predictions = predict_with_artifact(artifact, [{"age": 30, "score": 0.5, "city": "北京"}])
    assert len(predictions) == 1
    assert predictions[0]["prediction"] in (0, 1)
