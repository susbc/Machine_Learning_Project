import numpy as np
import pandas as pd
import pytest

from core.automl.problem_spec import (
    ProblemSpec,
    ProblemSpecError,
    capabilities_payload,
    normalize_problem_type,
    resolve_metric,
    resolve_problem_spec,
    validate_algorithm,
    validate_cv,
    validate_train_test_split,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([False, True] * 10, "binary_classification"),
        (["low", "mid", "high"] * 10, "multiclass_classification"),
        ([0, 1] * 10, "binary_classification"),
        (list(range(5)) * 20, "multiclass_classification"),
        (np.linspace(10.25, 90.75, 100), "regression"),
        (list(range(100)), "regression"),
    ],
)
def test_resolve_problem_spec_auto_uses_target_shape(values, expected):
    spec = resolve_problem_spec(pd.Series(values, name="target"), "target", "auto")

    assert spec.resolved_type.value == expected
    assert spec.source == "auto_inferred"
    assert spec.target.unique_count == len(pd.Series(values).unique())
    assert spec.explanation


def test_explicit_regression_override_is_canonical():
    spec = resolve_problem_spec(pd.Series([1, 2, 3, 1, 2, 3]), "score", "regression")

    assert spec.requested_type.value == "regression"
    assert spec.resolved_type.value == "regression"
    assert spec.source == "user_override"


def test_non_numeric_regression_has_stable_error_detail():
    with pytest.raises(ProblemSpecError) as caught:
        resolve_problem_spec(pd.Series(["a", "b", "c"]), "label", "regression")

    assert caught.value.as_detail() == {
        "code": "non_numeric_regression_target",
        "field": "problem_type",
        "message": "回归任务的目标列必须是有限数值",
    }


def test_auto_treats_numeric_looking_strings_as_categorical():
    spec = resolve_problem_spec(pd.Series(["1", "2", "3"] * 4), "label", "auto")

    assert spec.resolved_type.value == "multiclass_classification"
    assert spec.reason == "categorical_target"


def test_problem_spec_serialization_preserves_canonical_types_and_capabilities():
    spec = resolve_problem_spec(pd.Series([0, 1] * 5), "outcome", "binary")
    restored = ProblemSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert spec.to_dict()["capabilities"]["default_metric"] == "roc_auc"


def test_unsupported_problem_type_reports_all_canonical_values():
    with pytest.raises(ProblemSpecError) as caught:
        normalize_problem_type("clustering")

    assert caught.value.as_detail() == {
        "code": "unsupported_problem_type",
        "field": "problem_type",
        "message": "不支持的任务类型：clustering",
        "allowed": ["auto", "binary_classification", "multiclass_classification", "regression"],
    }


def test_regression_capabilities_and_metric_direction():
    spec = resolve_problem_spec(pd.Series(np.linspace(0.1, 9.9, 40)), "price", "auto")

    assert resolve_metric(spec, "auto") == "rmse"
    assert spec.metric_directions["rmse"] == "minimize"
    assert "linear_regression" in spec.algorithms
    with pytest.raises(ProblemSpecError, match="logistic_regression"):
        validate_algorithm(spec, "logistic_regression")
    assert capabilities_payload()["problem_types"]["regression"]["default_metric"] == "rmse"


def test_incompatible_metric_reports_the_metrics_for_the_resolved_type():
    spec = resolve_problem_spec(pd.Series(np.linspace(0.1, 9.9, 40)), "price", "auto")

    with pytest.raises(ProblemSpecError) as caught:
        resolve_metric(spec, "accuracy")

    assert caught.value.as_detail()["allowed"] == ["rmse", "mae", "r2"]


def test_regression_split_and_cv_require_enough_rows():
    spec = resolve_problem_spec(pd.Series([1.1, 2.2, 3.3, 4.4]), "price", "regression")

    validate_train_test_split(spec, pd.Series([1.1, 2.2, 3.3, 4.4]), 0.5)
    with pytest.raises(ProblemSpecError):
        validate_train_test_split(spec, pd.Series([1.1, 2.2, 3.3]), 0.5)
    with pytest.raises(ProblemSpecError):
        validate_cv(spec, pd.Series([1.1, 2.2, 3.3, 4.4]), 5)


def test_classification_split_rejects_a_rounded_split_that_drops_a_rare_class():
    target = pd.Series(["rare"] * 2 + ["common"] * 8)
    spec = resolve_problem_spec(target, "label", "binary")

    with pytest.raises(ProblemSpecError) as caught:
        validate_train_test_split(spec, target, 0.2)

    assert caught.value.code == "insufficient_classification_split"


def test_classification_cv_uses_the_smallest_class_count():
    target = pd.Series(["a"] * 3 + ["b"] * 5)
    spec = resolve_problem_spec(target, "label", "binary")

    with pytest.raises(ProblemSpecError) as caught:
        validate_cv(spec, target, 4)

    assert caught.value.as_detail()["message"].endswith("3")
