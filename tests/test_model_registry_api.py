import io
import time

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from backend.api._dataset import prepare_train_pipeline
from backend.main import app
import backend.api.models as model_registry
from core.automl.problem_spec import ProblemSpec, resolve_problem_spec


def test_model_registry_lists_saved_models_with_sources():
    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert isinstance(payload["models"], list)
    assert payload["models"]
    assert {"model_id", "source", "artifact_path", "metrics"}.issubset(payload["models"][0])
    assert any(item["source"] in {"automl", "manual_training", "agent_workflow", "legacy"} for item in payload["models"])


def test_model_registry_rejects_path_traversal_ids():
    response = TestClient(app).get("/api/models/..%2Fsecret")

    assert response.status_code == 400


def test_model_registry_predicts_with_supported_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LogisticRegression().fit(pd.DataFrame({"age": [20, 40, 60, 80]}), [0, 0, 1, 1])
    joblib.dump(
        {
            "model": model,
            "feature_columns": ["age"],
            "metrics": {"roc_auc": 0.9},
        },
        artifact_dir / "manual_training_test.joblib",
    )

    response = TestClient(app).post(
        "/api/models/manual%3Amanual_training_test/predict",
        json={"features": {"age": 70}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "manual:manual_training_test"
    assert isinstance(payload["prediction"], float)
    assert isinstance(payload["probability"], float)


def test_model_registry_batch_predicts_rows_with_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LogisticRegression().fit(pd.DataFrame({"age": [20, 40, 60, 80]}), [0, 0, 1, 1])
    joblib.dump(
        {
            "model": model,
            "feature_columns": ["age"],
            "metrics": {"roc_auc": 0.9},
        },
        artifact_dir / "manual_training_test.joblib",
    )

    response = TestClient(app).post(
        "/api/models/manual%3Amanual_training_test/batch-predict",
        json={"rows": [{"age": 30}, {"age": 75}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "manual:manual_training_test"
    assert payload["summary"]["count"] == 2
    assert len(payload["rows"]) == 2
    assert {"index", "features", "prediction", "probability"}.issubset(payload["rows"][0])


def test_registry_exposes_regression_metadata_and_null_class_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])
    joblib.dump(
        {
            "artifact_schema_version": 3,
            "problem_type": "regression",
            "problem_spec": resolve_problem_spec(pd.Series([1.5, 3.5, 5.5]), "target", "regression").to_dict(),
            "model": model,
            "feature_columns": ["x"],
            "metrics": {"rmse": 0.1, "mae": 0.08, "r2": 0.99},
        },
        artifact_dir / "manual_training_reg.joblib",
    )

    client = TestClient(app)
    listed = client.get("/api/models").json()["models"][0]
    assert listed["problem_type"] == "regression"
    assert listed["problem_spec"]["resolved_type"] == "regression"
    assert listed["legacy"] is False

    response = client.post(
        "/api/models/manual%3Amanual_training_reg/batch-predict",
        json={"rows": [{"x": 3}, {"x": 4}]},
    )

    assert response.status_code == 200
    batch = response.json()
    assert batch["summary"] == {"count": 2, "positive_rate": None, "average_probability": None}
    assert batch["rows"][0]["prediction"] == pytest.approx(7.5)
    assert batch["rows"][0]["encoded_prediction"] is None


def test_registry_treats_integer_schema_v3_as_non_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])
    joblib.dump(
        {
            "artifact_schema_version": np.int64(3),
            "problem_spec": resolve_problem_spec(pd.Series([1.5, 3.5, 5.5]), "target", "regression").to_dict(),
            "model": model,
            "feature_columns": ["x"],
        },
        artifact_dir / "manual_training_numpy_schema.joblib",
    )

    listed = TestClient(app).get("/api/models").json()["models"][0]

    assert listed["legacy"] is False
    assert listed["problem_type"] == "regression"


def test_registry_batch_summary_matches_normalized_v3_regression_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])
    joblib.dump(
        {
            "artifact_schema_version": 3,
            "problem_type": "REGRESSION",
            "problem_spec": resolve_problem_spec(pd.Series([1.5, 3.5, 5.5]), "target", "regression").to_dict(),
            "model": model,
            "feature_columns": ["x"],
        },
        artifact_dir / "manual_training_normalized_regression.joblib",
    )

    client = TestClient(app)
    listed = client.get("/api/models").json()["models"][0]
    assert listed["problem_type"] == "regression"

    response = client.post(
        "/api/models/manual%3Amanual_training_normalized_regression/batch-predict",
        json={"rows": [{"x": 3}, {"x": 4}]},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == {"count": 2, "positive_rate": None, "average_probability": None}


@pytest.mark.parametrize(
    ("task_metadata", "expected_code", "expected_field"),
    [
        ({"artifact_schema_version": "3", "problem_type": "regression"}, "invalid_artifact_schema_version", "artifact_schema_version"),
        ({"artifact_schema_version": 3}, "invalid_artifact_problem_type", "problem_type"),
        ({"artifact_schema_version": 3, "problem_type": "auto"}, "invalid_artifact_problem_type", "problem_type"),
        ({"artifact_schema_version": 3, "problem_type": "clustering"}, "invalid_artifact_problem_type", "problem_type"),
        ({"artifact_schema_version": 3, "problem_spec": {"resolved_type": 17}}, "invalid_artifact_problem_type", "problem_type"),
    ],
)
def test_registry_exposes_invalid_v3_metadata_and_matches_batch_error(
    tmp_path, monkeypatch, task_metadata, expected_code, expected_field
):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])
    joblib.dump(
        {
            "model": model,
            "feature_columns": ["x"],
            **task_metadata,
        },
        artifact_dir / "manual_training_invalid_v3.joblib",
    )

    client = TestClient(app)
    listed = client.get("/api/models").json()["models"][0]
    assert listed["legacy"] is False
    assert listed["problem_type"] is None
    assert listed["problem_spec"] is None
    assert listed["metadata_error"]["code"] == expected_code
    assert listed["metadata_error"]["field"] == expected_field

    detail = client.get("/api/models/manual%3Amanual_training_invalid_v3").json()
    assert detail == listed

    response = client.post(
        "/api/models/manual%3Amanual_training_invalid_v3/batch-predict",
        json={"rows": [{"x": 3}]},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == listed["metadata_error"]


def test_registry_and_batch_accept_nested_persisted_problem_spec_object(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])
    spec = resolve_problem_spec(pd.Series([1.5, 3.5, 5.5]), "target", "regression")
    assert isinstance(spec, ProblemSpec)
    joblib.dump(
        {
            "artifact_schema_version": 3,
            "problem_spec": spec,
            "model": model,
            "feature_columns": ["x"],
        },
        artifact_dir / "manual_training_problem_spec_object.joblib",
    )

    client = TestClient(app)
    listed = client.get("/api/models").json()["models"][0]
    assert listed["legacy"] is False
    assert listed["problem_type"] == "regression"
    assert listed["problem_spec"] == spec.to_dict()

    response = client.post(
        "/api/models/manual%3Amanual_training_problem_spec_object/batch-predict",
        json={"rows": [{"x": 3}, {"x": 4}]},
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {"count": 2, "positive_rate": None, "average_probability": None}


@pytest.mark.parametrize("task_metadata", [{}, {"artifact_schema_version": 2}, {"artifact_schema_version": np.int64(2)}])
def test_registry_marks_only_missing_or_older_valid_schema_as_legacy(tmp_path, monkeypatch, task_metadata):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    joblib.dump(
        {"model": LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1]), "feature_columns": ["x"], **task_metadata},
        artifact_dir / "manual_training_legacy.joblib",
    )

    listed = TestClient(app).get("/api/models").json()["models"][0]
    assert listed["legacy"] is True
    assert listed["metadata_error"] is None


def test_registry_safely_lists_non_dict_artifacts_as_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True)
    joblib.dump(["not", "an", "artifact", "dictionary"], artifact_dir / "manual_training_non_dict.joblib")

    listed = TestClient(app).get("/api/models").json()["models"][0]

    assert listed["legacy"] is True
    assert listed["problem_type"] is None
    assert listed["problem_spec"] is None
    assert listed["metadata_error"] is None


def _train_v2_artifact(labels=("正常", "违约"), n: int = 60):
    """训练最小 v2 artifact（preprocessor + label_encoder + 原始 feature_columns）。"""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n).astype(float),
            "city": rng.choice(["北京", "上海"], n),
            "label": rng.choice(list(labels), n),
        }
    )
    bundle = prepare_train_pipeline(df, "label", [])
    preprocessor = bundle.preprocessor.fit(bundle.X_train)
    model = LogisticRegression(max_iter=200).fit(preprocessor.transform(bundle.X_train), bundle.y_train)
    return {
        "artifact_version": 2,
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": bundle.label_encoder,
        "feature_columns": bundle.feature_columns,
    }


def _save_v2_artifact(tmp_path, monkeypatch, artifact) -> str:
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    artifact_dir = tmp_path / "saved" / "manual_training"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_dir / "manual_training_v2.joblib")
    return "manual%3Amanual_training_v2"


def test_registry_predict_v2_returns_original_string_label(tmp_path, monkeypatch):
    artifact = _train_v2_artifact()
    model_id = _save_v2_artifact(tmp_path, monkeypatch, artifact)

    response = TestClient(app).post(
        f"/api/models/{model_id}/predict",
        json={"features": {"age": 45, "city": "北京"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"正常", "违约"}
    assert isinstance(payload["encoded_prediction"], float)
    # 与真实管线对拍：prediction 是反变换的原始标签，概率是预测类别的概率
    enc = int(payload["encoded_prediction"])
    frame = pd.DataFrame([{"age": 45, "city": "北京"}])
    proba = artifact["model"].predict_proba(artifact["preprocessor"].transform(frame))[0]
    assert payload["probability"] == pytest.approx(float(proba[enc]))
    assert payload["prediction"] == artifact["label_encoder"].inverse_transform([enc])[0]


def test_registry_predict_v2_returns_diagnostics(tmp_path, monkeypatch):
    model_id = _save_v2_artifact(tmp_path, monkeypatch, _train_v2_artifact())

    response = TestClient(app).post(
        f"/api/models/{model_id}/predict",
        json={"features": {"city": "亚特兰蒂斯", "aeg": 1}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostics"]["unknown_input_keys"] == ["aeg"]
    assert payload["diagnostics"]["missing_columns_imputed"] == ["age"]
    assert payload["prediction"] in {"正常", "违约"}


def test_batch_summary_binary_positive_rate_uses_encoded_prediction(tmp_path, monkeypatch):
    model_id = _save_v2_artifact(tmp_path, monkeypatch, _train_v2_artifact())

    response = TestClient(app).post(
        f"/api/models/{model_id}/batch-predict",
        json={"rows": [{"age": 25, "city": "北京"}, {"age": 60, "city": "上海"}, {"age": 40, "city": "北京"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    rows = payload["rows"]
    expected_positive = sum(1 for row in rows if row["encoded_prediction"] == 1.0) / len(rows)
    assert payload["summary"]["positive_rate"] == pytest.approx(expected_positive)
    assert payload["summary"]["average_probability"] == pytest.approx(
        sum(row["probability"] for row in rows) / len(rows)
    )


def test_batch_summary_multiclass_positive_rate_is_none(tmp_path, monkeypatch):
    artifact = _train_v2_artifact(labels=("低风险", "中风险", "高风险"), n=90)
    model_id = _save_v2_artifact(tmp_path, monkeypatch, artifact)

    response = TestClient(app).post(
        f"/api/models/{model_id}/batch-predict",
        json={"rows": [{"age": 25, "city": "北京"}, {"age": 60, "city": "上海"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["count"] == 2
    assert payload["summary"]["positive_rate"] is None
    assert {row["prediction"] for row in payload["rows"]} <= {"低风险", "中风险", "高风险"}


def test_training_to_registry_end_to_end_string_labels():
    """端到端：字符串标签训练 → save → registry predict 返回原始标签。"""
    client = TestClient(app)
    rng = np.random.default_rng(9)
    rows = 80
    score = rng.normal(0, 1, rows)
    label = np.where(score + rng.normal(0, 0.3, rows) > 0.3, "违约", "正常")
    df = pd.DataFrame({"age": rng.integers(20, 65, rows), "score": score, "label": label})
    csv_body = io.BytesIO(df.to_csv(index=False).encode("utf-8"))

    upload = client.post(
        "/api/training/datasets",
        files={"file": ("string_labels.csv", csv_body, "text/csv")},
        data={"target_col": "label"},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    job_response = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": dataset_id,
            "algorithm": "logistic_regression",
            "params": {"solver": "liblinear", "max_iter": 500},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert job_response.status_code == 201
    job_id = job_response.json()["job_id"]

    deadline = time.time() + 12.0
    final_status = {}
    while time.time() < deadline:
        final_status = client.get(f"/api/training/jobs/{job_id}").json()
        if final_status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.25)
    assert final_status["status"] == "completed"

    save_response = client.post(f"/api/training/jobs/{job_id}/save")
    assert save_response.status_code == 200

    predict_response = client.post(
        f"/api/models/manual%3Amanual_training_{job_id}/predict",
        json={"features": {"age": 50, "score": 0.8}},
    )
    assert predict_response.status_code == 200
    payload = predict_response.json()
    assert payload["prediction"] in {"正常", "违约"}
    assert 0.0 <= payload["probability"] <= 1.0
