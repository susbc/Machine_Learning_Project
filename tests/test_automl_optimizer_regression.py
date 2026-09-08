"""Regression behavior for the shared Agent workflow optimizer."""

import numpy as np
import pandas as pd
import pytest

from core.automl.automl_optimizer import AutoMLOptimizer


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
