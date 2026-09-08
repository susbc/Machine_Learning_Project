"""workflow.py 修复测试：optimize callbacks 透传 + 预测端点走共享推理。"""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import backend.api.workflow as workflow
from backend.main import app
from core.automl.automl_optimizer import AutoMLOptimizer
from core.llm.agent_schemas import FeatureAction, FeatureStrategy
from model_code.feature_executor import FeatureEngineeringExecutor


def test_optimizer_forwards_callbacks_to_study_optimize():
    """optimize(callbacks=[...]) 必须透传给 study.optimize，SSE 进度才不再是回放。"""
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series(rng.integers(0, 2, 60))
    config = {
        "task_type": "binary_classification",
        "optimization_metric": "accuracy",
        "candidate_models": ["logistic_regression"],
        "n_trials": 2,
        "search_spaces": {},
    }
    optimizer = AutoMLOptimizer(X, y, config, cv_splits=2, n_jobs=1)

    seen: list[int] = []

    def probe_callback(study, trial):
        seen.append(trial.number)

    optimizer.optimize(n_trials=2, show_progress=False, callbacks=[probe_callback])

    assert len(seen) == 2


def _save_executor_artifact(tmp_path) -> str:
    """构造 executor 形态 artifact（字符串标签 + label_encoder），返回 task_id。"""
    df = pd.DataFrame(
        {
            "age": [22, 35, 48, 51, 26, 44, 33, 57],
            "is_fraud": ["正常", "正常", "违约", "违约", "正常", "违约", "正常", "违约"],
        }
    )
    executor = FeatureEngineeringExecutor(target_col="is_fraud")
    strategy = FeatureStrategy(
        actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
    )
    transformed = executor.fit_transform(df, strategy)
    feature_columns = [c for c in transformed.columns if c != "is_fraud"]
    label_encoder = LabelEncoder().fit(transformed["is_fraud"])
    model = LogisticRegression(max_iter=200).fit(
        transformed[feature_columns], label_encoder.transform(transformed["is_fraud"])
    )
    task_id = "task_ab12cd34"
    joblib.dump(
        {
            "feature_executor": executor,
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "best_value": 0.9,
        },
        tmp_path / f"model_{task_id}.pkl",
    )
    return task_id


def test_workflow_predict_uses_shared_inference_and_returns_original_label(tmp_path, monkeypatch):
    """predict 端点走共享推理：executor 不再二次 get_dummies，标签反变换为原始字符串。"""
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path)
    task_id = _save_executor_artifact(tmp_path)

    response = TestClient(app).post(
        "/api/workflow/predict",
        json={"model_version": task_id, "features": {"age": 55}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["prediction"] in {"正常", "违约"}
    assert isinstance(payload["encoded_prediction"], float)
    assert payload["probability"] is None or 0.0 <= payload["probability"] <= 1.0
