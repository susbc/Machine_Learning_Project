"""Workflow optimizer, task reconciliation, persistence, and inference tests."""
import json
import logging
import math
import threading
import time
from types import SimpleNamespace

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


def test_workflow_public_endpoints_do_not_expose_internal_exception_paths(tmp_path, monkeypatch):
    unsafe_error = RuntimeError("unsafe /private/server/secret.parquet")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_data_retrieval_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(unsafe_error),
    )
    retrieve = TestClient(app).post(
        "/api/workflow/step1_retrieve_data",
        json={"user_query": "查询订单", "target_col": "is_fraud"},
    )

    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "eda_summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_feature_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(unsafe_error),
    )
    features = TestClient(app).post(
        "/api/workflow/step2_generate_features",
        json={"task_id": task_id, "eda_report_text": ""},
    )

    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / f"model_{task_id}.pkl").write_bytes(b"broken")
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)
    monkeypatch.setattr(
        workflow.joblib,
        "load",
        lambda path: (_ for _ in ()).throw(unsafe_error),
    )
    prediction = TestClient(app).post(
        "/api/workflow/predict",
        json={"model_version": task_id, "features": {"age": 55}},
    )

    assert retrieve.status_code == 500
    assert retrieve.json()["detail"] == "数据提取失败，请检查查询配置后重试"
    assert features.status_code == 500
    assert features.json()["detail"] == "特征策略生成失败，请检查任务配置后重试"
    # Unreadable workflow artifacts fail closed during reconciliation rather
    # than being exposed as loadable legacy models.
    assert prediction.status_code == 404
    assert prediction.json()["detail"] == f"模型未找到：{task_id}，请先训练模型"
    assert all("/private/" not in response.text for response in (retrieve, features, prediction))


def test_workflow_step1_persists_partition_and_reports_train_only_eda(tmp_path, monkeypatch):
    """Catches Step 1 exposing full-data statistics before a holdout is assigned."""
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_data_retrieval_agent",
        lambda *args, **kwargs: SimpleNamespace(
            is_executable=True,
            sql="SELECT * FROM orders",
            thought_process="deterministic test query",
        ),
    )

    response = TestClient(app).post(
        "/api/workflow/step1_retrieve_data",
        json={"user_query": "预测欺诈", "target_col": "is_fraud"},
    )

    assert response.status_code == 200
    payload = response.json()
    task_dir = tmp_path / "output" / payload["task_id"]
    partition = json.loads((task_dir / "workflow_partition.json").read_text(encoding="utf-8"))
    persisted_eda = json.loads((task_dir / "eda_summary.json").read_text(encoding="utf-8"))
    assert partition["target"] == "is_fraud"
    assert partition["row_count"] == 1000
    assert len(partition["train_positions"]) == 750
    assert len(partition["holdout_positions"]) == 250
    assert set(partition["train_positions"]).isdisjoint(partition["holdout_positions"])
    assert payload["eda_summary"] == persisted_eda
    assert persisted_eda["rows"] == 750
    assert persisted_eda["source_rows"] == 1000


def _full_frame_eda(frame: pd.DataFrame, target_col: str) -> dict:
    """Reproduce the pre-fix full-frame EDA artifact used by legacy workflow tasks."""
    features = {}
    for column in frame.columns:
        values = frame[column]
        stats = {
            "type": str(values.dtype),
            "null_rate": float(values.isnull().mean()),
            "unique_values": int(values.nunique()),
        }
        if pd.api.types.is_numeric_dtype(values):
            stats.update({"mean": float(values.mean()), "std": float(values.std())})
        features[column] = stats
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "target": target_col,
        "user_query": "holdout isolation sentinel",
        "features": features,
    }


def test_workflow_holdout_only_mutations_cannot_change_strategy_agent_or_fitted_search(
    tmp_path, monkeypatch
):
    """Catches any holdout statistic entering strategy, Agent, evaluator, or CV selection."""
    output_dir = tmp_path / "output"
    saved_dir = tmp_path / "saved"
    monkeypatch.setattr(workflow, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)

    train_rows = 60
    base = pd.DataFrame(
        {
            "signal": np.arange(80, dtype=float),
            "category_code": np.tile([0, 1], 40),
            "is_fraud": [0] * 45 + [1] * 15 + [0] * 15 + [1] * 5,
        }
    )
    mutated = base.copy()
    mutated.loc[60:69, "signal"] = np.nan
    mutated.loc[70:79, "signal"] = np.arange(10_000, 10_010, dtype=float)
    mutated.loc[60:79, "category_code"] = np.arange(100, 120)
    mutated.loc[60:79, "is_fraud"] = [0] + [1] * 19

    task_frames = {
        "task_aaaa0001": base,
        "task_bbbb0002": mutated,
    }
    for task_id, frame in task_frames.items():
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True)
        frame.to_parquet(task_dir / "raw_data.parquet", index=False)
        partition = {
            "schema_version": 2,
            "dataset_fingerprint": workflow._workflow_dataset_fingerprint(frame, "is_fraud"),
            "target": "is_fraud",
            "row_count": 80,
            "requested_type": "auto",
            "resolved_type": "binary_classification",
            "split_semantics": "stratified",
            "stratified": True,
            "holdout_size": 0.25,
            "random_state": 42,
            "train_positions": list(range(train_rows)),
            "holdout_positions": list(range(train_rows, 80)),
        }
        partition["provenance_id"] = workflow._partition_provenance_id(partition)
        (task_dir / "workflow_partition.json").write_text(
            json.dumps(partition), encoding="utf-8"
        )
        (task_dir / "eda_summary.json").write_text(
            json.dumps(_full_frame_eda(frame, "is_fraud")), encoding="utf-8"
        )

    feature_agent_inputs: list[dict] = []

    def sentinel_feature_agent(eda_json: str, _report: str):
        summary = json.loads(eda_json)
        feature_agent_inputs.append(summary)
        action = (
            "fillna_mean"
            if summary["features"]["signal"]["null_rate"] > 0
            else "scale_standard"
        )
        return FeatureStrategy(
            actions=[FeatureAction(column="signal", action=action, reason="sentinel strategy")]
        )

    monkeypatch.setattr(workflow.orchestrator, "run_feature_agent", sentinel_feature_agent)
    client = TestClient(app)
    strategies = []
    for task_id in task_frames:
        generated = client.post(
            "/api/workflow/step2_generate_features",
            json={"task_id": task_id, "eda_report_text": ""},
        )
        assert generated.status_code == 200
        strategies.append(generated.json()["strategy"])

    automl_agent_inputs: list[dict] = []

    def sentinel_automl_agent(data_features: dict, _scenario: str) -> dict:
        automl_agent_inputs.append(dict(data_features))
        return {
            "task_type": "binary_classification",
            "optimization_metric": "accuracy",
            "candidate_models": ["logistic_regression"],
            "n_trials": 10,
            "search_spaces": {},
            "rationale": "holdout isolation sentinel",
        }

    monkeypatch.setattr(workflow, "recommend_automl_config", sentinel_automl_agent)
    optimizer_inputs: list[tuple[pd.DataFrame, np.ndarray, dict]] = []
    original_optimizer_init = AutoMLOptimizer.__init__

    def recording_optimizer_init(self, X, y, agent_config, *args, **kwargs):
        optimizer_inputs.append((X.copy(), np.asarray(y).copy(), dict(agent_config)))
        original_optimizer_init(self, X, y, agent_config, *args, **kwargs)

    monkeypatch.setattr(AutoMLOptimizer, "__init__", recording_optimizer_init)

    terminal_events = []
    for task_id in task_frames:
        events = _read_workflow_sse(task_id, "auto", 10, 3)
        terminal_events.append(events[-1])
        assert events[-1]["status"] == "DONE"

    artifacts = [joblib.load(saved_dir / f"model_{task_id}.pkl") for task_id in task_frames]
    pd.testing.assert_frame_equal(optimizer_inputs[0][0], optimizer_inputs[1][0])
    np.testing.assert_array_equal(optimizer_inputs[0][1], optimizer_inputs[1][1])
    assert optimizer_inputs[0][2] == optimizer_inputs[1][2]
    assert feature_agent_inputs[0] == feature_agent_inputs[1]
    assert automl_agent_inputs[0] == automl_agent_inputs[1]
    assert strategies[0] == strategies[1]
    assert artifacts[0]["config"] == artifacts[1]["config"]
    assert artifacts[0]["best_value"] == pytest.approx(artifacts[1]["best_value"])

    train_features = base.iloc[:train_rows].drop(columns=["is_fraud"])
    transformed = [artifact["feature_executor"].transform(train_features) for artifact in artifacts]
    pd.testing.assert_frame_equal(transformed[0], transformed[1])
    np.testing.assert_allclose(
        artifacts[0]["model"].predict_proba(transformed[0]),
        artifacts[1]["model"].predict_proba(transformed[1]),
    )
    assert terminal_events[0]["metrics"] != terminal_events[1]["metrics"]


def test_workflow_rejects_legacy_strategy_before_optimizer_can_use_it(tmp_path, monkeypatch):
    """Catches a pre-partition, full-data strategy entering model selection after upgrade."""
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "signal": np.linspace(0.0, 1.0, 40),
            "amount": np.linspace(10.0, 90.0, 40),
        }
    )
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps(_full_frame_eda(frame, "amount")), encoding="utf-8"
    )
    legacy_strategy = FeatureStrategy(
        actions=[FeatureAction(column="signal", action="scale_standard", reason="full-data")]
    )
    (task_dir / "feature_strategy.json").write_text(
        legacy_strategy.model_dump_json(), encoding="utf-8"
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    optimizer_constructed = False

    def reject_optimizer(*args, **kwargs):
        nonlocal optimizer_constructed
        optimizer_constructed = True
        raise AssertionError("legacy strategy reached optimizer")

    monkeypatch.setattr(AutoMLOptimizer, "__init__", reject_optimizer)

    response = TestClient(app).get(
        "/api/workflow/step3_train_stream",
        params={"task_id": task_id, "problem_type": "regression", "n_trials": 10, "cv_folds": 3},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "workflow_strategy_provenance_required",
        "field": "feature_strategy",
        "message": "工作流特征策略缺少可信来源，请重新运行特征生成",
    }
    assert not optimizer_constructed


@pytest.mark.parametrize("mutation", ["replacement", "reorder"])
def test_workflow_partition_rejects_same_shape_dataset_fingerprint_mismatch(
    tmp_path, mutation
):
    """Catches positional partition reuse after same-shape content replacement or row reorder."""
    task_dir = tmp_path / "task_ab12cd34"
    task_dir.mkdir()
    frame = pd.DataFrame(
        {
            "feature": np.arange(40, dtype=float),
            "amount": np.linspace(1.25, 80.75, 40),
        }
    )
    workflow._load_or_create_workflow_partition(task_dir, frame, "amount")
    if mutation == "replacement":
        changed = frame.copy()
        changed["feature"] = changed["feature"] + 10_000.0
    else:
        changed = frame.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="指纹"):
        workflow._load_or_create_workflow_partition(task_dir, changed, "amount")


def test_workflow_numeric_string_explicit_regression_is_not_blocked_by_auto_split(
    tmp_path, monkeypatch
):
    """Catches native object auto classification rejecting a valid explicit regression override."""
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "feature": np.linspace(-4.0, 4.0, 80),
            "amount": [f"{value / 10:.1f}" for value in range(80)],
        }
    )
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps({"target": "amount", "user_query": "预测数值字符串"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_feature_agent",
        lambda *_args, **_kwargs: FeatureStrategy(
            actions=[FeatureAction(column="feature", action="scale_standard", reason="test")]
        ),
    )
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
    client = TestClient(app)

    generated = client.post(
        "/api/workflow/step2_generate_features",
        json={"task_id": task_id, "eda_report_text": ""},
    )
    events = _read_workflow_sse(task_id, "regression", 10, 3) if generated.status_code == 200 else []

    assert generated.status_code == 200
    assert events[-1]["status"] == "DONE"
    assert events[-1]["problem_spec"]["requested_type"] == "regression"
    assert events[-1]["problem_spec"]["resolved_type"] == "regression"
    assert all(math.isfinite(value) for value in events[-1]["metrics"].values())


def test_workflow_explicit_multiclass_rebuilds_auto_regression_partition_with_coverage(
    tmp_path, monkeypatch
):
    """Catches an explicit classification override retaining an unstratified auto split."""
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    classes = np.repeat(np.arange(21), 4)
    frame = pd.DataFrame({"feature": np.arange(len(classes), dtype=float), "label": classes})
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps({"target": "label", "user_query": "显式多分类"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    feature_agent_inputs: list[dict] = []

    def feature_agent(eda_json: str, _report: str):
        feature_agent_inputs.append(json.loads(eda_json))
        return FeatureStrategy(
            actions=[FeatureAction(column="feature", action="scale_standard", reason="test")]
        )

    monkeypatch.setattr(workflow.orchestrator, "run_feature_agent", feature_agent)
    monkeypatch.setattr(
        workflow,
        "recommend_automl_config",
        lambda *_args, **_kwargs: {
            "task_type": "multiclass_classification",
            "optimization_metric": "accuracy",
            "candidate_models": ["logistic_regression"],
            "n_trials": 10,
            "search_spaces": {},
            "rationale": "explicit multiclass coverage",
        },
    )
    client = TestClient(app)

    generated = client.post(
        "/api/workflow/step2_generate_features",
        json={"task_id": task_id, "eda_report_text": ""},
    )
    assert generated.status_code == 200
    events = _read_workflow_sse(task_id, "multiclass_classification", 10, 3)
    partition = json.loads((task_dir / "workflow_partition.json").read_text(encoding="utf-8"))
    train_classes = set(frame.iloc[partition["train_positions"]]["label"])
    holdout_classes = set(frame.iloc[partition["holdout_positions"]]["label"])

    assert events[-1]["status"] == "DONE"
    assert partition["requested_type"] == "multiclass_classification"
    assert partition["resolved_type"] == "multiclass_classification"
    assert partition["split_semantics"] == "stratified"
    assert train_classes == set(range(21))
    assert holdout_classes == set(range(21))
    assert len(feature_agent_inputs) == 2


def test_workflow_rejects_strategy_with_mismatched_partition_provenance(tmp_path, monkeypatch):
    """Catches a valid-shaped strategy being paired with a different partition generation."""
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {"feature": np.arange(60, dtype=float), "amount": np.linspace(1.0, 10.0, 60)}
    )
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps({"target": "amount", "user_query": "provenance mismatch"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_feature_agent",
        lambda *_args, **_kwargs: FeatureStrategy(
            actions=[FeatureAction(column="feature", action="scale_standard", reason="test")]
        ),
    )
    client = TestClient(app)
    generated = client.post(
        "/api/workflow/step2_generate_features",
        json={"task_id": task_id, "eda_report_text": ""},
    )
    assert generated.status_code == 200
    persisted = json.loads((task_dir / "feature_strategy.json").read_text(encoding="utf-8"))
    (task_dir / "feature_strategy.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "provenance_id": "0" * 64,
                "strategy": persisted.get("strategy", persisted),
                "eda_report_text": "",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(
        "/api/workflow/step3_train_stream",
        params={"task_id": task_id, "problem_type": "regression", "n_trials": 10, "cv_folds": 3},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "workflow_strategy_provenance_mismatch",
        "field": "feature_strategy",
        "message": "工作流特征策略与数据分区来源不一致，请重新运行特征生成",
    }


def _persist_test_workflow_selection(
    task_dir, frame: pd.DataFrame, target_col: str, strategy: FeatureStrategy, problem_type: str
) -> None:
    partition = workflow._build_workflow_partition(frame, target_col, problem_type)
    eda = workflow._build_workflow_eda_summary(
        frame.iloc[partition["train_positions"]],
        target_col,
        partition=partition,
        source_rows=len(frame),
        user_query="预测订单金额",
    )
    workflow._persist_workflow_selection(task_dir, partition, eda, strategy, "")


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
    strategy = FeatureStrategy(
        actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
    )
    _persist_test_workflow_selection(task_dir, frame, "order_fee", strategy, "auto")
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


def _linear_regression_agent_config(_data_features: dict, _scenario: str) -> dict:
    """Keeps step3 offline: without this stub the AutoML agent calls a real LLM provider."""
    return {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["linear_regression"],
        "n_trials": 10,
        "search_spaces": {},
        "rationale": "deterministic offline sentinel",
    }


def test_workflow_regression_corrects_agent_suggestion_and_predicts_float(
    tmp_path, monkeypatch, caplog
):
    """Catches an Agent suggestion overriding the canonical target-derived task."""
    task_id = _prepare_workflow_task(tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_feature_agent",
        lambda *_args, **_kwargs: FeatureStrategy(
            actions=[FeatureAction(column="age", action="fillna_mean", reason="override")]
        ),
    )
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
    _persist_test_workflow_selection(
        task_dir,
        frame,
        "order_fee",
        FeatureStrategy(
            actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
        ),
        "auto",
    )
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow.orchestrator,
        "run_feature_agent",
        lambda *_args, **_kwargs: FeatureStrategy(
            actions=[FeatureAction(column="age", action="fillna_mean", reason="override")]
        ),
    )
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
    _persist_test_workflow_selection(
        task_dir,
        frame,
        "order_fee",
        FeatureStrategy(
            actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
        ),
        "auto",
    )
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
    small_frame = pd.read_parquet(task_dir / "raw_data.parquet")
    _persist_test_workflow_selection(
        task_dir,
        small_frame,
        "order_fee",
        FeatureStrategy(
            actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")]
        ),
        "regression",
    )
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
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
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
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
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
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
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


def test_workflow_cancel_after_artifact_replace_rolls_back_run_owned_publication(
    tmp_path, monkeypatch
):
    """Catches cancellation publishing a selectable artifact without a completed state."""
    task_id = _prepare_workflow_task(tmp_path)
    saved_dir = tmp_path / "saved"
    artifact_path = saved_dir / f"model_{task_id}.pkl"
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
    artifact_replaced = threading.Event()
    release_publication = threading.Event()
    original_replace = workflow.os.replace

    def replace_with_cancel_barrier(source, destination):
        original_replace(source, destination)
        if destination == artifact_path:
            # This is the exact publication boundary: the run's cancellation signal wins
            # before completed state is allowed to commit.
            workflow.active_workflow_runs[task_id].cancel_event.set()
            artifact_replaced.set()
            assert release_publication.wait(timeout=5)

    monkeypatch.setattr(workflow.os, "replace", replace_with_cancel_barrier)
    events: list[dict] = []

    def consume_stream():
        events.extend(_read_workflow_sse(task_id, "regression", 10, 3))

    thread = threading.Thread(target=consume_stream)
    thread.start()
    try:
        assert artifact_replaced.wait(timeout=5)
        assert artifact_path.is_file()
    finally:
        release_publication.set()
    thread.join(timeout=5)

    state = json.loads(
        (tmp_path / "output" / task_id / "workflow_state.json").read_text(encoding="utf-8")
    )
    terminal_statuses = [
        event["status"] for event in events if event["status"] in {"DONE", "CANCELLED", "ERROR"}
    ]
    assert terminal_statuses == ["CANCELLED"]
    assert state["status"] == "cancelled"
    assert not artifact_path.exists()
    assert task_id not in workflow.active_workflow_runs
    assert not thread.is_alive()


def test_workflow_cancel_after_completed_state_returns_completed_during_baseline(
    tmp_path, monkeypatch
):
    """Catches baseline work keeping a completed run cancellable and active."""
    from model_code.mlops_monitor import DataDriftMonitor

    task_id = _prepare_workflow_task(tmp_path)
    saved_dir = tmp_path / "saved"
    artifact_path = saved_dir / f"model_{task_id}.pkl"
    state_path = tmp_path / "output" / task_id / "workflow_state.json"
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)
    monkeypatch.setattr(workflow, "recommend_automl_config", _linear_regression_agent_config)
    baseline_entered = threading.Event()
    release_baseline = threading.Event()
    original_extract_baseline = DataDriftMonitor.extract_baseline

    def extract_with_completed_barrier(self, data, features):
        baseline_entered.set()
        assert release_baseline.wait(timeout=5)
        return original_extract_baseline(self, data, features)

    monkeypatch.setattr(DataDriftMonitor, "extract_baseline", extract_with_completed_barrier)
    events: list[dict] = []

    def consume_stream():
        events.extend(_read_workflow_sse(task_id, "regression", 10, 3))

    thread = threading.Thread(target=consume_stream)
    thread.start()
    try:
        assert baseline_entered.wait(timeout=5)
        completed_state = json.loads(state_path.read_text(encoding="utf-8"))
        active_during_baseline = task_id in workflow.active_workflow_runs
        cancel = TestClient(app).post(
            f"/api/workflow/task/{task_id}/run/{completed_state['run_id']}/cancel"
        )
    finally:
        release_baseline.set()
    thread.join(timeout=5)

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    terminal_statuses = [
        event["status"] for event in events if event["status"] in {"DONE", "CANCELLED", "ERROR"}
    ]
    assert completed_state["status"] == "completed"
    assert not active_during_baseline
    assert cancel.status_code == 200
    assert cancel.json() == {
        "task_id": task_id,
        "run_id": completed_state["run_id"],
        "status": "completed",
    }
    assert terminal_statuses == ["DONE"]
    assert final_state["status"] == "completed"
    assert artifact_path.is_file()
    assert joblib.load(artifact_path)["run_id"] == completed_state["run_id"]
    assert not thread.is_alive()
