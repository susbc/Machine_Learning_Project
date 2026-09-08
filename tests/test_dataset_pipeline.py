"""共享数据处理管线（backend/api/_dataset.py）的单元测试。

覆盖：先划分后拟合（无泄漏）、小数据集、高基数类别限爆、指标省略语义、
统一推理（v2/executor/legacy 三形态）、特征重要性命名。
"""
import numpy as np
import pandas as pd
import pytest

from backend.api._dataset import (
    TrainBundle,
    _normalise_dataframe,
    _read_csv_bytes,
    build_preprocessor,
    calibrate_binary_threshold,
    compute_classification_metrics,
    compute_regression_metrics,
    dataframe_quality_report,
    feature_importance_rows,
    prepare_train_pipeline,
    predict_with_artifact,
)
from core.automl.problem_spec import resolve_problem_spec


def _sample_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, n).astype(float),
            "income": rng.normal(50000, 15000, n),
            "city": rng.choice(["北京", "上海", "广州"], n),
            "label": rng.choice([0, 1], n),
        }
    )


# ---------- prepare_train_pipeline / build_preprocessor ----------


def test_split_before_fit_no_leakage():
    df = _sample_df(100)
    df.loc[df.index[:10], "income"] = np.nan
    bundle = prepare_train_pipeline(df, "label", [])

    assert isinstance(bundle, TrainBundle)
    assert set(bundle.feature_columns) == {"age", "income", "city"}

    preprocessor = bundle.preprocessor.fit(bundle.X_train)
    stats = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_
    train_medians = bundle.X_train[["age", "income"]].median().to_numpy()
    # 拟合统计量必须来自训练集（确定性断言），且与全量中位数不同（泄漏哨兵）
    assert np.allclose(stats, train_medians)
    full_medians = df[["age", "income"]].median().to_numpy()
    assert not np.allclose(stats, full_medians)


def test_transform_tolerates_unseen_category():
    bundle = prepare_train_pipeline(_sample_df(60), "label", [])
    preprocessor = bundle.preprocessor.fit(bundle.X_train)
    expected_width = len(preprocessor.get_feature_names_out())

    unseen = bundle.X_test.copy()
    unseen.iloc[0, unseen.columns.get_loc("city")] = "从未见过的城市"
    transformed = preprocessor.transform(unseen)

    assert transformed.shape[1] == expected_width


def test_small_dataset_20_rows():
    df = _sample_df(20)
    bundle = prepare_train_pipeline(df, "label", [])

    # 20 行自适应 test_size=0.3（分层取整后约 5-6 行测试集）
    assert len(bundle.X_train) + len(bundle.X_test) == 20
    assert 4 <= len(bundle.X_test) <= 7
    assert bundle.label_encoder is not None


def test_min_frequency_bounds_high_cardinality():
    n = 200
    df = pd.DataFrame(
        {
            "num": np.arange(n, dtype=float),
            "cat": [f"cat_{i % 100}" for i in range(n)],  # 100 个唯一值，每个约 2 次
            "label": [0, 1] * (n // 2),
        }
    )
    bundle = prepare_train_pipeline(df, "label", [])
    preprocessor = bundle.preprocessor.fit(bundle.X_train)

    # min_frequency=2 下单例类别被合并，宽度远小于 100 个 one-hot 列
    assert len(preprocessor.get_feature_names_out()) < 100


def test_preprocessor_keeps_high_cardinality_output_sparse_and_bounded():
    from scipy import sparse

    n = 400
    df = pd.DataFrame(
        {
            "amount": np.arange(n, dtype=float),
            "merchant": [f"merchant_{i % 150}" for i in range(n)],
            "label": [0, 1] * (n // 2),
        }
    )
    bundle = prepare_train_pipeline(df, "label", [])
    matrix = bundle.preprocessor.fit_transform(bundle.X_train)

    assert sparse.issparse(matrix)
    assert matrix.shape[1] <= 102


def test_csv_reader_supports_gb18030_and_pipe_separator():
    raw = "城市|金额|label\n北京|100|0\n上海|200|1\n".encode("gb18030")

    frame = _normalise_dataframe(_read_csv_bytes(raw))

    assert frame.columns.tolist() == ["城市", "金额", "label"]
    assert frame.loc[1, "城市"] == "上海"


def test_quality_report_exposes_actionable_data_issues():
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0],
            "value": [np.inf, np.nan, np.nan],
            "label": [0, 0, 0],
        }
    )

    quality = dataframe_quality_report(frame, "label")

    assert quality["infinite_values"] == 1
    assert quality["constant_columns"] == ["constant", "value"]
    assert quality["high_missing_columns"] == ["value"]


def test_rare_class_is_rejected_with_clear_message():
    frame = pd.DataFrame({"x": range(8), "label": [0, 0, 0, 0, 0, 0, 0, 1]})

    with pytest.raises(ValueError, match="每个类别至少需要 2 条记录"):
        prepare_train_pipeline(frame, "label", [])


def test_build_preprocessor_rejects_empty_features():
    df = pd.DataFrame({"label": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="特征"):
        prepare_train_pipeline(df, "label", [])


def test_prepare_regression_pipeline_keeps_continuous_target():
    df = pd.DataFrame({"x": range(20), "price": np.linspace(10.25, 40.75, 20)})
    spec = resolve_problem_spec(df["price"], "price", "auto")

    bundle = prepare_train_pipeline(df, "price", [], test_size=0.25, problem_spec=spec)

    assert bundle.label_encoder is None
    assert bundle.problem_spec.resolved_type.value == "regression"
    assert bundle.y_train.dtype.kind == "f"
    assert set(bundle.y_train) <= set(df["price"])


def test_prepare_regression_pipeline_rejects_non_finite_target_with_saved_spec():
    valid = pd.Series(np.linspace(1.0, 8.0, 8), name="price")
    spec = resolve_problem_spec(valid, "price", "regression")
    df = pd.DataFrame({"x": range(8), "price": [1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0]})

    with pytest.raises(ValueError, match="有限数值"):
        prepare_train_pipeline(df, "price", [], problem_spec=spec)


# ---------- compute_classification_metrics ----------


class _SingleClassModel:
    """predict_proba 正常但 y_test 只有一个类别 → roc_auc 不可计算。"""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.8, 0.2], (len(X), 1))


def test_metrics_omit_roc_auc_when_undefined():
    X_test = np.zeros((10, 3))
    y_test = pd.Series([0] * 10)

    metrics = compute_classification_metrics(_SingleClassModel(), X_test, y_test)

    assert "roc_auc" not in metrics
    assert metrics["accuracy"] == 1.0
    assert "f1" in metrics
    assert "best_threshold" in metrics
    assert "confusion_matrix" in metrics


def test_binary_threshold_is_calibrated_without_test_data():
    from sklearn.linear_model import LogisticRegression

    X = np.arange(80, dtype=float).reshape(-1, 1)
    y = np.array([0] * 40 + [1] * 40)
    calibration = calibrate_binary_threshold(LogisticRegression(), X, y, random_state=7)

    assert calibration["source"] == "training_calibration_split"
    assert 0 <= calibration["threshold"] <= 1
    assert calibration["calibration_size"] == 16


def test_compute_regression_metrics_returns_only_regression_metrics():
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0], [1], [2], [3]], [1.0, 3.0, 5.0, 7.0])

    metrics = compute_regression_metrics(model, [[4], [5]], [9.0, 11.0])

    assert metrics == {
        "rmse": pytest.approx(0.0),
        "mae": pytest.approx(0.0),
        "r2": pytest.approx(1.0),
    }


# ---------- feature_importance_rows ----------


class _ImportanceModel:
    feature_importances_ = np.array([0.5, 0.3, 0.2])


def test_feature_importance_rows_names_and_top_n():
    rows = feature_importance_rows(_ImportanceModel(), ["num__a", "cat__b_北京", "cat__b_上海"], top_n=2)

    assert rows == [
        {"feature": "num__a", "importance": pytest.approx(0.5)},
        {"feature": "cat__b_北京", "importance": pytest.approx(0.3)},
    ]


def test_feature_importance_rows_mismatch_zips_shortest():
    rows = feature_importance_rows(_ImportanceModel(), ["only_one"])

    assert rows == [{"feature": "only_one", "importance": pytest.approx(0.5)}]


def test_feature_importance_rows_no_attributes_returns_empty():
    assert feature_importance_rows(object(), ["a", "b"]) == []


def test_feature_importance_rows_supports_linear_regression_coefficients():
    from sklearn.linear_model import LinearRegression

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 4.0
    model = LinearRegression().fit(X, y)

    rows = feature_importance_rows(model, ["first", "second"])

    assert rows == [
        {"feature": "second", "importance": pytest.approx(3.0)},
        {"feature": "first", "importance": pytest.approx(2.0)},
    ]


# ---------- predict_with_artifact ----------


def _train_v2_artifact(labels=("正常", "违约")):
    """训练一个最小 v2 artifact（字符串标签）。"""
    from sklearn.linear_model import LogisticRegression

    n = 60
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
    model = LogisticRegression(max_iter=200)
    model.fit(preprocessor.transform(bundle.X_train), bundle.y_train)
    return {
        "artifact_version": 2,
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": bundle.label_encoder,
        "feature_columns": bundle.feature_columns,
    }


def test_predict_v2_string_labels_roundtrip():
    artifact = _train_v2_artifact()

    rows = predict_with_artifact(artifact, [{"age": 45, "city": "北京"}])

    assert len(rows) == 1
    row = rows[0]
    assert row["prediction"] in {"正常", "违约"}
    assert isinstance(row["encoded_prediction"], float)
    assert 0.0 <= row["probability"] <= 1.0
    # 概率必须是“预测类别”的概率，而非固定类别 1 的概率
    proba = artifact["model"].predict_proba(
        artifact["preprocessor"].transform(pd.DataFrame([{"age": 45, "city": "北京"}]))
    )[0]
    assert row["probability"] == pytest.approx(float(proba[int(row["encoded_prediction"])]))


def test_schema_v3_regression_prediction_has_no_class_fields():
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0], [1], [2]], [1.5, 3.5, 5.5])

    result = predict_with_artifact(
        {
            "artifact_schema_version": 3,
            "problem_type": "regression",
            "model": model,
            "feature_columns": ["x"],
        },
        [{"x": 3}],
    )[0]

    assert result["prediction"] == pytest.approx(7.5)
    assert result["encoded_prediction"] is None
    assert result["probability"] is None


def test_schema_v3_regression_prediction_uses_nested_persisted_problem_spec():
    from sklearn.linear_model import LinearRegression

    target = pd.Series([1.5, 3.5, 5.5], name="price")
    spec = resolve_problem_spec(target, "price", "regression")
    model = LinearRegression().fit([[0], [1], [2]], target)

    result = predict_with_artifact(
        {
            "artifact_schema_version": 3,
            "problem_spec": spec.to_dict(),
            "model": model,
            "feature_columns": ["x"],
        },
        [{"x": 3}],
    )[0]

    assert result == {
        "prediction": pytest.approx(7.5),
        "encoded_prediction": None,
        "probability": None,
        "diagnostics": {"unknown_input_keys": [], "missing_columns_imputed": []},
    }


def test_predict_v2_diagnostics_for_unknown_missing_and_typo_keys():
    artifact = _train_v2_artifact()

    rows = predict_with_artifact(artifact, [{"city": "亚特兰蒂斯", "aeg": 1}])

    row = rows[0]
    assert row["diagnostics"]["unknown_input_keys"] == ["aeg"]
    assert row["diagnostics"]["missing_columns_imputed"] == ["age"]
    # 未知类别值被 OHE ignore，缺失键被 imputer 兜底，不抛错
    assert row["prediction"] in {"正常", "违约"}


def test_predict_v2_uses_saved_decision_threshold():
    artifact = _train_v2_artifact(labels=("negative", "positive"))
    artifact["decision_threshold"] = 1.0

    row = predict_with_artifact(artifact, [{"age": 45, "city": "北京"}])[0]

    assert row["encoded_prediction"] == 0.0
    assert row["prediction"] == "negative"


def test_predict_legacy_fallback_parity():
    """legacy artifact（无 preprocessor/executor）：保持 get_dummies 对齐行为。"""
    from sklearn.linear_model import LogisticRegression

    train = pd.DataFrame({"age": [20, 30, 50, 60], "city": ["北京", "上海", "北京", "上海"]})
    encoded = pd.get_dummies(train, dummy_na=True)
    model = LogisticRegression(max_iter=200).fit(encoded, [0, 0, 1, 1])
    artifact = {"model": model, "feature_columns": list(encoded.columns)}

    rows = predict_with_artifact(artifact, [{"age": 55, "city": "上海"}])

    assert rows[0]["prediction"] in (0.0, 1.0)
    assert 0.0 <= rows[0]["probability"] <= 1.0


def test_predict_executor_form_without_double_get_dummies():
    from sklearn.linear_model import LogisticRegression
    from model_code.feature_executor import FeatureEngineeringExecutor
    from core.llm.agent_schemas import FeatureAction, FeatureStrategy

    df = pd.DataFrame({"age": [20, 30, 50, 60, 25, 35], "is_fraud": [0, 0, 1, 1, 0, 1]})
    executor = FeatureEngineeringExecutor(target_col="is_fraud")
    strategy = FeatureStrategy(actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")])
    transformed = executor.fit_transform(df, strategy)
    feature_columns = [c for c in transformed.columns if c != "is_fraud"]
    model = LogisticRegression(max_iter=200).fit(transformed[feature_columns], transformed["is_fraud"])
    artifact = {"feature_executor": executor, "model": model, "feature_columns": feature_columns}

    rows = predict_with_artifact(artifact, [{"age": 55}])

    assert rows[0]["prediction"] in (0.0, 1.0)
    assert 0.0 <= rows[0]["probability"] <= 1.0


def test_feature_executor_reuses_binning_edges_and_aligns_inference_columns():
    from model_code.feature_executor import FeatureEngineeringExecutor
    from core.llm.agent_schemas import FeatureAction, FeatureStrategy

    frame = pd.DataFrame({"value": np.arange(100), "label": [0, 1] * 50})
    executor = FeatureEngineeringExecutor(target_col="label")
    strategy = FeatureStrategy(actions=[FeatureAction(column="value", action="binning", reason="四分位", params={"bins": 4})])
    transformed = executor.fit_transform(frame, strategy)

    inference = executor.transform(pd.DataFrame({"value": [10, 90], "ignored": [1, 2]}))

    assert executor.state["feature_columns"] == ["value"]
    assert transformed["label"].tolist() == frame["label"].tolist()
    assert inference.columns.tolist() == ["value"]
    assert inference["value"].tolist() == [0, 3]


def test_feature_executor_can_disable_supervised_target_encoding():
    from model_code.feature_executor import FeatureEngineeringExecutor
    from core.llm.agent_schemas import FeatureAction, FeatureStrategy

    frame = pd.DataFrame({"merchant": ["a", "a", "b", "c"], "label": [0, 1, 1, 0]})
    executor = FeatureEngineeringExecutor(target_col="label", allow_target_encoding=False)
    strategy = FeatureStrategy(actions=[FeatureAction(column="merchant", action="target_encode", reason="高基数")])

    transformed = executor.fit_transform(frame, strategy)
    inference = executor.transform(pd.DataFrame({"merchant": ["a", "new"]}))

    assert transformed["merchant"].tolist() == [0.5, 0.5, 0.25, 0.25]
    assert inference["merchant"].tolist() == [0.5, 0.0]
