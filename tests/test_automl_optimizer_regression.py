"""Regression behavior for the shared Agent workflow optimizer."""

import numpy as np
import pandas as pd
import pytest

from core.automl.automl_optimizer import AutoMLOptimizer
from core.llm.agent_schemas import FeatureAction, FeatureStrategy
from model_code.feature_executor import FeatureEngineeringExecutor, FeatureEngineeringTransformer


def test_feature_executor_retains_fitted_state_api():
    executor = FeatureEngineeringExecutor(target_col="target")

    assert executor.is_fitted() is False


def test_optimizer_minimizes_readable_rmse():
    """Catches negated RMSE values or a maximize study for regression."""
    X = pd.DataFrame({"x": np.linspace(0, 10, 60)})
    y = pd.Series(4.0 * X["x"] + 2.0)
    config = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["linear_regression"],
        "n_trials": 1,
        "search_spaces": {"linear_regression": {}},
    }

    optimizer = AutoMLOptimizer(X, y, config, cv_splits=3, n_jobs=1)
    study, model = optimizer.optimize(n_trials=1, show_progress=False)

    assert study.direction.name == "MINIMIZE"
    assert study.best_value >= 0
    assert model.predict([[12.0]])[0] == pytest.approx(50.0, rel=0.02)


def test_optimizer_reports_task_incompatible_model_build_failure():
    """Catches bypassing ModelFactory compatibility validation."""
    X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
    y = pd.Series(np.linspace(10.0, 40.0, 30))
    config = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["logistic_regression"],
        "n_trials": 1,
        "search_spaces": {"logistic_regression": {}},
    }

    optimizer = AutoMLOptimizer(X, y, config, cv_splits=3, n_jobs=1)

    with pytest.raises(RuntimeError, match="不适用于 regression"):
        optimizer.optimize(n_trials=1, show_progress=False)


def test_optimizer_fits_stateful_feature_executor_inside_each_cv_fold(monkeypatch):
    """A validation-only statistic/category must not enter that fold's executor state."""
    fit_records: list[dict] = []
    original_fit_transform = FeatureEngineeringExecutor.fit_transform

    def audited_fit_transform(self, frame, strategy, y=None):
        transformed = original_fit_transform(self, frame, strategy, y)
        fit_records.append(
            {
                "indices": set(frame.index),
                "fill_value": float(self.state["fill_values"]["amount"]),
                "categories": set(self.state["encoders"]["segment"]["mapping"]),
            }
        )
        return transformed

    monkeypatch.setattr(FeatureEngineeringExecutor, "fit_transform", audited_fit_transform)
    rows = 60
    frame = pd.DataFrame(
        {
            "amount": [np.nan, *np.arange(1, rows, dtype=float)],
            "segment": ["validation_only", *(["common"] * (rows - 1))],
        }
    )
    target = pd.Series(3.0 * np.arange(rows, dtype=float) + 2.0)
    strategy = FeatureStrategy(
        actions=[
            FeatureAction(column="amount", action="fillna_mean", reason="fold statistic"),
            FeatureAction(column="segment", action="frequency_encode", reason="fold category"),
        ]
    )
    transformer = FeatureEngineeringTransformer(strategy=strategy, target_col="target")
    config = {
        "task_type": "regression",
        "optimization_metric": "rmse",
        "candidate_models": ["linear_regression"],
        "n_trials": 1,
        "search_spaces": {"linear_regression": {}},
    }

    optimizer = AutoMLOptimizer(
        frame,
        target,
        config,
        cv_splits=3,
        n_jobs=1,
        feature_transformer=transformer,
    )
    optimizer.optimize(n_trials=1, show_progress=False)

    fold_records = [record for record in fit_records if len(record["indices"]) == 40]
    assert len(fold_records) == 3
    sentinel_fold = next(record for record in fold_records if 0 not in record["indices"])
    assert "validation_only" not in sentinel_fold["categories"]
    expected_mean = frame.loc[list(sentinel_fold["indices"]), "amount"].mean()
    assert sentinel_fold["fill_value"] == pytest.approx(expected_mean)
    assert len(fit_records[-1]["indices"]) == rows
