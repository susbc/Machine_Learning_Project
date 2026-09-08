"""Workflow optimizer, task reconciliation, persistence, and inference tests."""
import json
import logging
import math
import threading
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import backend.api.workflow as workflow
from backend.main import app
from core.automl.automl_agent import AutoMLAgent, extract_data_features
from core.automl.automl_optimizer import AutoMLOptimizer, OptimizationCancelled
from core.automl.problem_spec import resolve_problem_spec
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


def _prepare_workflow_task(tmp_path) -> str:
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    x = np.linspace(-8.0, 8.0, 80)
    frame = pd.DataFrame(
        {
            "age": np.linspace(20.0, 60.0, 80),
            "city_code": ["北京", "上海"] * 40,
            "order_fee": 5.0 * x + 100.0,
        }
    )
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps({"target": "order_fee", "user_query": "预测订单金额"}, ensure_ascii=False),
        encoding="utf-8",
    )
    strategy = FeatureStrategy(
        actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
    )
    (task_dir / "feature_strategy.json").write_text(
        strategy.model_dump_json(), encoding="utf-8"
    )
    return task_id


def _read_workflow_sse(
    task_id: str, problem_type: str, n_trials: int, cv_folds: int
) -> list[dict]:
    events = []
    with TestClient(app) as client:
        with client.stream(
            "GET",
            "/api/workflow/step3_train_stream",
            params={
                "task_id": task_id,
                "problem_type": problem_type,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
            },
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line.removeprefix("data:").strip()))
    return events


def test_workflow_regression_corrects_agent_suggestion_and_predicts_float(
    tmp_path, monkeypatch, caplog
):
    """Catches an Agent suggestion overriding the canonical target-derived task."""
    task_id = _prepare_workflow_task(tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow,
        "recommend_automl_config",
        lambda data_features, scenario: {
            "task_type": "binary_classification",
            "optimization_metric": "roc_auc",
            "candidate_models": ["logistic_regression"],
            "n_trials": 10,
            "search_spaces": {},
            "rationale": "test mismatch",
        },
    )

    with caplog.at_level(logging.INFO, logger="backend.api.workflow"):
        events = _read_workflow_sse(task_id, problem_type="auto", n_trials=10, cv_folds=3)

    assert all(event["status"] != "ERROR" for event in events)
    running = [event for event in events if event["status"] == "RUNNING"]
    assert running
    assert all(event["metric"] == "rmse" and event["value"] >= 0 for event in running)
    done = events[-1]
    assert done["status"] == "DONE"
    assert done["problem_spec"]["resolved_type"] == "regression"
    assert done["suggested_type"] == "binary_classification"
    assert done["best_metric"] == "rmse"
    assert done["best_value"] >= 0
    assert done["metric"] == "rmse"
    assert done["value"] == done["best_value"]
    assert done["agent_corrections"]
    assert set(done["metrics"]) == {"rmse", "mae", "r2"}
    assert all(math.isfinite(value) for value in done["metrics"].values())
    assert done["metrics"]["r2"] >= 0.95
    assert done["metrics_evaluation"] == {
        "scope": "holdout",
        "rows": 20,
        "train_rows": 60,
        "random_state": 42,
    }
    assert done["cv_folds"] == 3

    artifact = joblib.load(tmp_path / "saved" / f"model_{task_id}.pkl")
    assert artifact["artifact_schema_version"] == 3
    assert artifact["problem_type"] == "regression"
    assert artifact["problem_spec"] == done["problem_spec"]
    assert artifact["label_encoder"] is None
    assert artifact["best_metric"] == "rmse"
    assert artifact["best_value"] == done["best_value"]
    assert artifact["metrics"] == done["metrics"]
    assert artifact["metrics_evaluation"] == done["metrics_evaluation"]
    assert artifact["selected_algorithm"] == "linear_regression"

    completion_logs = [
        record.getMessage()
        for record in caplog.records
        if "workflow_training_completed" in record.getMessage()
    ]
    assert completion_logs == [
        "workflow_training_completed task_id=task_ab12cd34 "
        "resolved_type=regression selected_algorithm=linear_regression metric=rmse"
    ]

    with TestClient(app) as client:
        status = client.get(f"/api/workflow/task/{task_id}/status").json()
        prediction = client.post(
            "/api/workflow/predict",
            json={
                "model_version": task_id,
                "features": {"age": 40, "city_code": "北京"},
            },
        ).json()
    assert status["problem_spec"]["resolved_type"] == "regression"
    assert status["best_metric"] == "rmse"
    assert status["best_value"] == done["best_value"]
    assert status["metrics"] == done["metrics"]
    assert status["metrics_evaluation"] == done["metrics_evaluation"]
    assert "best_auc" not in status
    assert isinstance(prediction["prediction"], float)
    assert prediction["probability"] is None
    assert prediction["encoded_prediction"] is None


class _FailingProvider:
    def generate(self, prompt):
        raise RuntimeError("offline")


class _TextProvider:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt):
        return self.response


@pytest.mark.parametrize(
    "provider",
    [
        None,
        _FailingProvider(),
        _TextProvider("not json"),
        _TextProvider('{"task_type": "regression"}'),
    ],
)
def test_automl_agent_uses_task_aware_regression_fallback(provider):
    """Catches every fallback path reverting a regression task to classification."""
    features = {
        "n_samples": 80,
        "n_features": 3,
        "n_categorical": 1,
        "categorical_ratio": 1 / 3,
        "class_imbalance_ratio": None,
        "task_type": "regression",
        "target_col": "order_fee",
    }

    config = AutoMLAgent(llm_provider=provider).generate_config(features)

    assert config["task_type"] == "regression"
    assert config["optimization_metric"] == "rmse"
    assert config["candidate_models"] == ["random_forest", "linear_regression"]
    assert config["search_spaces"]["linear_regression"] == {}


def test_extract_data_features_uses_canonical_native_dtype_inference():
    """Catches reintroducing a unique-count-only Agent task heuristic."""
    frame = pd.DataFrame(
        {
            "feature": np.arange(30),
            "numeric_looking_label": [str(value) for value in range(30)],
        }
    )

    features = extract_data_features(frame, "numeric_looking_label")

    assert features["task_type"] == "multiclass_classification"


def test_automl_agent_uses_empty_search_space_for_linear_classification_baseline():
    """Catches LightGBM-only parameters leaking into logistic regression."""
    provider = _TextProvider(
        json.dumps(
            {
                "task_type": "binary_classification",
                "optimization_metric": "roc_auc",
                "candidate_models": ["logistic_regression"],
                "n_trials": 10,
                "rationale": "linear baseline",
                "search_spaces": {},
            }
        )
    )
    features = {
        "n_samples": 80,
        "n_features": 2,
        "n_categorical": 0,
        "categorical_ratio": 0.0,
        "class_imbalance_ratio": 1,
        "task_type": "binary_classification",
        "target_col": "label",
    }

    config = AutoMLAgent(llm_provider=provider).generate_config(features)

    assert config["search_spaces"]["logistic_regression"] == {}


def test_workflow_status_preserves_best_auc_for_schema_v2_classification(tmp_path, monkeypatch):
    """Catches dropping the legacy AUC alias merely because a v2 marker exists."""
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    joblib.dump(
        {
            "artifact_schema_version": 2,
            "problem_type": "binary_classification",
            "best_value": 0.82,
            "config": {"optimization_metric": "roc_auc"},
            "feature_columns": ["age"],
        },
        saved_dir / f"model_{task_id}.pkl",
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)

    status = TestClient(app).get(f"/api/workflow/task/{task_id}/status").json()

    assert status["best_metric"] == "roc_auc"
    assert status["best_value"] == 0.82
    assert status["best_auc"] == 0.82


def test_workflow_discards_search_space_from_mismatched_agent_task():
    """Catches classifier-only tuning parameters leaking into a shared regressor."""
    spec = resolve_problem_spec(
        pd.Series(np.linspace(1.0, 20.0, 40)), "amount", "auto"
    )
    suggested = {
        "task_type": "binary_classification",
        "optimization_metric": "roc_auc",
        "candidate_models": ["random_forest"],
        "search_spaces": {
            "random_forest": {
                "class_weight": {"type": "categorical", "choices": ["balanced"]}
            }
        },
        "rationale": "wrong task",
    }

    config, _, corrections = workflow._coordinate_automl_config(suggested, spec, 10)

    assert config["candidate_models"] == ["random_forest"]
    assert config["search_spaces"]["random_forest"] == {}
    assert any(item["field"] == "search_spaces" for item in corrections)


@pytest.mark.parametrize(
    ("candidate", "space", "correction_field"),
    [
        (
            "linear_regression",
            {"n_estimators": {"type": "int", "low": 10, "high": 20}},
            "search_spaces.linear_regression.n_estimators",
        ),
        (
            "random_forest",
            {"n_estimators": {"type": "int", "low": "many", "high": 20}},
            "search_spaces.random_forest.n_estimators",
        ),
        (
            "random_forest",
            {"n_estimators": {"type": "int", "low": -10, "high": 20}},
            "search_spaces.random_forest.n_estimators",
        ),
        (
            "random_forest",
            {
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "unsupported"],
                }
            },
            "search_spaces.random_forest.max_features",
        ),
        (
            "random_forest",
            {"n_estimators": {"type": "float", "low": 100.0, "high": 200.0}},
            "search_spaces.random_forest.n_estimators",
        ),
        (
            "random_forest",
            {
                "n_estimators": {
                    "type": "loguniform",
                    "low": 100.0,
                    "high": 200.0,
                }
            },
            "search_spaces.random_forest.n_estimators",
        ),
        (
            "random_forest",
            {
                "max_depth": {
                    "type": "categorical",
                    "choices": [10, 10.5],
                }
            },
            "search_spaces.random_forest.max_depth",
        ),
    ],
)
def test_workflow_drops_algorithm_invalid_search_rules(candidate, space, correction_field):
    """Catches same-task unknown parameters and malformed Optuna rules reaching trials."""
    spec = resolve_problem_spec(pd.Series(np.linspace(1.0, 20.0, 40)), "amount", "auto")
    suggested = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": [candidate],
        "search_spaces": {candidate: space},
        "rationale": "invalid search rule",
    }

    config, _, corrections = workflow._coordinate_automl_config(suggested, spec, 10)

    assert config["search_spaces"][candidate] == {}
    assert any(item["field"] == correction_field for item in corrections)


def test_workflow_preserves_valid_custom_search_space_within_canonical_domain():
    """Catches domain validation replacing valid bounded custom tuning rules."""
    spec = resolve_problem_spec(pd.Series(np.linspace(1.0, 20.0, 40)), "amount", "auto")
    valid_space = {
        "n_estimators": {"type": "int", "low": 150, "high": 300},
        "max_depth": {"type": "int", "low": 4, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", None]},
    }
    suggested = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["random_forest"],
        "search_spaces": {"random_forest": valid_space},
        "rationale": "valid custom search",
    }

    config, _, corrections = workflow._coordinate_automl_config(suggested, spec, 10)

    assert config["search_spaces"]["random_forest"] == valid_space
    assert not any(item["field"].startswith("search_spaces.random_forest") for item in corrections)


@pytest.mark.parametrize("choices", [[5, 10], [5.0, 10.0]])
def test_workflow_preserves_integer_categorical_space_as_integer_values(choices):
    """Catches integral categorical choices reaching integer estimators as floats."""
    spec = resolve_problem_spec(pd.Series(np.linspace(1.0, 20.0, 40)), "amount", "auto")
    suggested = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["random_forest"],
        "search_spaces": {
            "random_forest": {
                "max_depth": {"type": "categorical", "choices": choices}
            }
        },
        "rationale": "valid integer categorical search",
    }

    config, _, corrections = workflow._coordinate_automl_config(suggested, spec, 10)

    rule = config["search_spaces"]["random_forest"]["max_depth"]
    assert rule == {"type": "categorical", "choices": [5, 10]}
    assert all(type(value) is int for value in rule["choices"])
    assert not corrections


def test_workflow_explicit_regression_override_precedes_agent_suggestion(tmp_path, monkeypatch):
    """Catches an Agent classification suggestion overriding an explicit user request."""
    task_id = _prepare_workflow_task(tmp_path)
    task_dir = tmp_path / "output" / task_id
    frame = pd.read_parquet(task_dir / "raw_data.parquet")
    frame["order_fee"] = np.tile([1, 2, 3, 4], 20)
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow,
        "recommend_automl_config",
        lambda data_features, scenario: {
            "task_type": "multiclass_classification",
            "optimization_metric": "accuracy",
            "candidate_models": ["logistic_regression"],
            "n_trials": 10,
            "search_spaces": {},
            "rationale": "override mismatch",
        },
    )

    events = _read_workflow_sse(task_id, problem_type="regression", n_trials=10, cv_folds=3)

    done = events[-1]
    assert done["status"] == "DONE"
    assert done["problem_spec"]["requested_type"] == "regression"
    assert done["problem_spec"]["resolved_type"] == "regression"
    assert done["problem_spec"]["source"] == "user_override"
    assert done["suggested_type"] == "multiclass_classification"
    assert done["metric"] == "rmse"


def test_workflow_rejects_invalid_cv_before_opening_sse(tmp_path, monkeypatch):
    task_id = _prepare_workflow_task(tmp_path)
    task_dir = tmp_path / "output" / task_id
    frame = pd.DataFrame(
        {
            "age": np.arange(12, dtype=float),
            "city_code": ["北京", "上海"] * 6,
            "order_fee": ["normal"] * 8 + ["rare"] * 4,
        }
    )
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")

    response = TestClient(app).get(
        "/api/workflow/step3_train_stream",
        params={"task_id": task_id, "problem_type": "auto", "n_trials": 10, "cv_folds": 4},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == {
        "code": "insufficient_cv_samples",
        "field": "cv_folds",
        "message": "当前任务每折至少需要一个样本，可用上限为 3",
    }
    assert not (task_dir / "workflow_state.json").exists()


def test_workflow_rejects_too_few_regression_cv_rows_before_sse(tmp_path, monkeypatch):
    task_id = _prepare_workflow_task(tmp_path)
    task_dir = tmp_path / "output" / task_id
    pd.DataFrame(
        {
            "age": np.arange(5, dtype=float),
            "city_code": ["a", "b", "c", "d", "e"],
            "order_fee": [1.1, 2.4, 4.2, 7.9, 13.7],
        }
    ).to_parquet(task_dir / "raw_data.parquet", index=False)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")

    response = TestClient(app).get(
        "/api/workflow/step3_train_stream",
        params={"task_id": task_id, "problem_type": "regression", "n_trials": 10, "cv_folds": 4},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "insufficient_cv_samples"
    assert response.json()["detail"]["field"] == "cv_folds"


def test_workflow_failed_state_and_error_survive_reconnect_without_artifact(tmp_path, monkeypatch):
    """Catches transient-only failures reverting to in_progress after the stream ends."""
    task_id = _prepare_workflow_task(tmp_path)
    task_dir = tmp_path / "output" / task_id
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        AutoMLOptimizer,
        "optimize",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("unsafe /private/server/model.pkl")),
    )

    events = _read_workflow_sse(task_id, problem_type="auto", n_trials=10, cv_folds=3)

    assert [event["status"] for event in events] == ["ERROR"]
    assert "msg" not in events[0]
    assert events[0]["error"] == "模型训练失败，请检查配置后重试"
    assert "/private/" not in events[0]["error"]
    state_path = task_dir / "workflow_state.json"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["error"] == events[0]["error"]
    assert not (tmp_path / "saved" / f"model_{task_id}.pkl").exists()
    assert list(task_dir.glob(".workflow_state.*.tmp")) == []

    # A new client has no access to the stream-local queue/completion payload.
    with TestClient(app) as client:
        restarted_status = client.get(f"/api/workflow/task/{task_id}/status").json()

    assert restarted_status["status"] == "failed"
    assert restarted_status["error"] == persisted["error"]
    assert "msg" not in restarted_status


def test_workflow_rejects_overlapping_run_and_keeps_original_run_id(tmp_path, monkeypatch):
    task_id = _prepare_workflow_task(tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    entered = threading.Event()
    release = threading.Event()

    def blocking_optimize(self, **kwargs):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("stop test run")

    monkeypatch.setattr(AutoMLOptimizer, "optimize", blocking_optimize)
    first_events: list[dict] = []

    def consume_first_stream():
        first_events.extend(_read_workflow_sse(task_id, "auto", 10, 3))

    thread = threading.Thread(target=consume_first_stream)
    thread.start()
    assert entered.wait(timeout=5)
    state_path = tmp_path / "output" / task_id / "workflow_state.json"
    original_state = json.loads(state_path.read_text(encoding="utf-8"))

    second = TestClient(app).get(
        "/api/workflow/step3_train_stream",
        params={"task_id": task_id, "problem_type": "regression", "n_trials": 10, "cv_folds": 3},
    )
    state_after_overlap = json.loads(state_path.read_text(encoding="utf-8"))
    release.set()
    thread.join(timeout=5)

    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "workflow_run_active"
    assert state_after_overlap["run_id"] == original_state["run_id"]
    assert state_after_overlap["requested_type"] == "auto"
    assert not thread.is_alive()


def test_workflow_cancel_endpoint_propagates_to_optimizer_and_persists_cancelled_state(
    tmp_path, monkeypatch
):
    task_id = _prepare_workflow_task(tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    entered = threading.Event()
    observed_cancel_event: list[threading.Event] = []

    def cancellable_optimize(self, **kwargs):
        observed_cancel_event.append(self.cancel_event)
        entered.set()
        assert self.cancel_event.wait(timeout=5)
        raise OptimizationCancelled("cancelled by test")

    monkeypatch.setattr(AutoMLOptimizer, "optimize", cancellable_optimize)
    events: list[dict] = []

    def consume_stream():
        events.extend(_read_workflow_sse(task_id, "auto", 10, 3))

    thread = threading.Thread(target=consume_stream)
    thread.start()
    assert entered.wait(timeout=5)
    state_path = tmp_path / "output" / task_id / "workflow_state.json"
    run_id = json.loads(state_path.read_text(encoding="utf-8"))["run_id"]

    response = TestClient(app).post(f"/api/workflow/task/{task_id}/run/{run_id}/cancel")
    thread.join(timeout=5)

    assert response.status_code == 200
    assert observed_cancel_event[0].is_set()
    assert events == [{"status": "CANCELLED", "error": "训练已取消", "run_id": run_id}]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "cancelled"
    assert persisted["error"] == "训练已取消"
    assert not thread.is_alive()
