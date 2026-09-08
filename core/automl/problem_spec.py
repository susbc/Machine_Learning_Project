"""Canonical task-type resolution for the model-training entry points."""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


class ProblemType(str, Enum):
    AUTO = "auto"
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"


ALGORITHMS = {
    ProblemType.BINARY: ("lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"),
    ProblemType.MULTICLASS: ("lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"),
    ProblemType.REGRESSION: ("lightgbm", "xgboost", "catboost", "random_forest", "linear_regression"),
}
METRICS = {
    ProblemType.BINARY: ("roc_auc", "accuracy", "f1"),
    ProblemType.MULTICLASS: ("accuracy", "f1", "roc_auc"),
    ProblemType.REGRESSION: ("rmse", "mae", "r2"),
}
DEFAULT_METRIC = {
    ProblemType.BINARY: "roc_auc",
    ProblemType.MULTICLASS: "accuracy",
    ProblemType.REGRESSION: "rmse",
}
METRIC_DIRECTION = {
    "roc_auc": "maximize",
    "accuracy": "maximize",
    "f1": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "r2": "maximize",
}
ALIASES = {"binary": ProblemType.BINARY, "multiclass": ProblemType.MULTICLASS}
EXPLANATIONS = {
    "categorical_target": "目标列为类别数据，按唯一值数量识别为分类任务",
    "two_numeric_values": "数值目标只有两个不同值，识别为二分类任务",
    "low_cardinality_integer_target": "整数目标的唯一值数量低于分类阈值，识别为多分类任务",
    "numeric_continuous_target": "数值目标具有连续取值，识别为回归任务",
}


class ProblemSpecError(ValueError):
    """A stable, field-oriented validation error for task resolution."""

    def __init__(
        self,
        code: str,
        field: str,
        message: str,
        allowed: Sequence[str] | None = None,
        received: Any | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message
        self.allowed = allowed
        self.received = received

    def as_detail(self) -> dict[str, Any]:
        detail = {"code": self.code, "field": self.field, "message": self.message}
        if self.received is not None:
            detail["received"] = self.received
        if self.allowed is not None:
            detail["allowed"] = list(self.allowed)
        return detail


@dataclass(frozen=True)
class TargetProfile:
    column: str
    dtype: str
    valid_rows: int
    unique_count: int
    unique_ratio: float
    integer_like: bool


@dataclass(frozen=True)
class ProblemSpec:
    requested_type: ProblemType
    resolved_type: ProblemType
    source: str
    reason: str
    explanation: str
    target: TargetProfile

    @property
    def algorithms(self) -> tuple[str, ...]:
        return ALGORITHMS[self.resolved_type]

    @property
    def metrics(self) -> tuple[str, ...]:
        return METRICS[self.resolved_type]

    @property
    def metric_directions(self) -> dict[str, str]:
        return {metric: METRIC_DIRECTION[metric] for metric in self.metrics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "requested_type": self.requested_type.value,
            "resolved_type": self.resolved_type.value,
            "source": self.source,
            "reason": self.reason,
            "explanation": self.explanation,
            "target": asdict(self.target),
            "capabilities": {
                "algorithms": list(self.algorithms),
                "metrics": list(self.metrics),
                "default_metric": DEFAULT_METRIC[self.resolved_type],
                "metric_directions": self.metric_directions,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemSpec":
        return cls(
            requested_type=normalize_problem_type(payload["requested_type"]),
            resolved_type=normalize_problem_type(payload["resolved_type"]),
            source=payload["source"],
            reason=payload["reason"],
            explanation=payload["explanation"],
            target=TargetProfile(**payload["target"]),
        )


def normalize_problem_type(value: Any | None) -> ProblemType:
    allowed = [item.value for item in ProblemType]
    if value is None:
        normalized = "auto"
    elif not isinstance(value, str):
        raise ProblemSpecError(
            "unsupported_problem_type",
            "problem_type",
            f"不支持的任务类型：{value}",
            allowed,
            received=value,
        )
    else:
        normalized = value.strip().lower()
    if normalized in ALIASES:
        return ALIASES[normalized]
    try:
        return ProblemType(normalized)
    except ValueError as exc:
        raise ProblemSpecError(
            "unsupported_problem_type",
            "problem_type",
            f"不支持的任务类型：{value}",
            allowed,
            received=value,
        ) from exc


def _validate_target_for_type(clean: pd.Series, resolved: ProblemType, convertible_numeric: bool) -> None:
    unique_count = int(clean.nunique(dropna=True))
    if resolved == ProblemType.BINARY and unique_count != 2:
        raise ProblemSpecError("invalid_binary_target", "problem_type", "二分类目标必须恰好有两个不同值")
    if resolved == ProblemType.MULTICLASS and unique_count < 3:
        raise ProblemSpecError("invalid_multiclass_target", "problem_type", "多分类目标至少需要三个不同值")
    if resolved == ProblemType.REGRESSION and not convertible_numeric:
        raise ProblemSpecError("non_numeric_regression_target", "problem_type", "回归任务的目标列必须是有限数值")


def _build_problem_spec(
    requested: ProblemType,
    resolved: ProblemType,
    target_col: str,
    clean: pd.Series,
    integer_like: bool,
    reason: str,
) -> ProblemSpec:
    source = "auto_inferred" if requested == ProblemType.AUTO else "user_override"
    explanation = EXPLANATIONS.get(reason, f"用户显式选择了{resolved.value}")
    profile = TargetProfile(
        column=target_col,
        dtype=str(clean.dtype),
        valid_rows=len(clean),
        unique_count=int(clean.nunique(dropna=True)),
        unique_ratio=float(clean.nunique(dropna=True) / len(clean)),
        integer_like=integer_like,
    )
    return ProblemSpec(requested, resolved, source, reason, explanation, profile)


def resolve_problem_spec(target: pd.Series, target_col: str, requested_type: str | None = "auto") -> ProblemSpec:
    """Infer a task type from a target series or validate an explicit override."""
    requested = normalize_problem_type(requested_type)
    clean = target.dropna()
    if clean.nunique(dropna=True) < 2:
        raise ProblemSpecError("single_value_target", "target_col", "目标变量至少需要两个不同值")

    numeric = pd.to_numeric(clean, errors="coerce")
    is_numeric = pd.api.types.is_numeric_dtype(clean)
    convertible_numeric = bool(numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all())
    integer_like = bool(convertible_numeric and np.allclose(numeric, np.round(numeric)))
    unique_count = int(clean.nunique(dropna=True))
    threshold = min(20, math.ceil(math.sqrt(len(clean))))

    if requested == ProblemType.AUTO:
        if not is_numeric:
            resolved = ProblemType.BINARY if unique_count == 2 else ProblemType.MULTICLASS
            reason = "categorical_target"
        elif not convertible_numeric:
            raise ProblemSpecError("non_finite_numeric_target", "target_col", "数值目标列包含非有限值")
        elif unique_count == 2:
            resolved, reason = ProblemType.BINARY, "two_numeric_values"
        elif integer_like and unique_count <= threshold:
            resolved, reason = ProblemType.MULTICLASS, "low_cardinality_integer_target"
        else:
            resolved, reason = ProblemType.REGRESSION, "numeric_continuous_target"
    else:
        resolved, reason = requested, f"user_selected_{requested.value}"
        _validate_target_for_type(clean, resolved, convertible_numeric)

    return _build_problem_spec(requested, resolved, target_col, clean, integer_like, reason)


def capabilities_payload() -> dict[str, Any]:
    """Return the task-type capabilities consumed by the API and frontend."""
    return {
        "problem_types": {
            problem_type.value: {
                "algorithms": list(ALGORITHMS[problem_type]),
                "metrics": list(METRICS[problem_type]),
                "default_metric": DEFAULT_METRIC[problem_type],
                "metric_directions": {
                    metric: METRIC_DIRECTION[metric] for metric in METRICS[problem_type]
                },
            }
            for problem_type in (ProblemType.BINARY, ProblemType.MULTICLASS, ProblemType.REGRESSION)
        }
    }


def resolve_metric(spec: ProblemSpec, requested_metric: str | None) -> str:
    metric = (requested_metric or "auto").strip().lower()
    if metric == "auto":
        return DEFAULT_METRIC[spec.resolved_type]
    if metric not in spec.metrics:
        raise ProblemSpecError(
            "incompatible_metric",
            "optimization_metric",
            f"指标 {metric} 不适用于 {spec.resolved_type.value}",
            spec.metrics,
        )
    return metric


def validate_algorithm(spec: ProblemSpec, algorithm: str) -> None:
    if algorithm not in spec.algorithms:
        raise ProblemSpecError(
            "incompatible_algorithm",
            "algorithm",
            f"算法 {algorithm} 不适用于 {spec.resolved_type.value}",
            spec.algorithms,
            received=algorithm,
        )


def validate_train_test_split(spec: ProblemSpec, target: pd.Series, test_size: float) -> None:
    clean = target.dropna()
    n_test = math.ceil(len(clean) * test_size)
    n_train = len(clean) - n_test
    if spec.resolved_type == ProblemType.REGRESSION:
        if len(clean) < 4 or min(n_train, n_test) < 2:
            raise ProblemSpecError(
                "insufficient_regression_split",
                "test_size",
                "回归训练集和测试集都至少需要两条记录",
            )
        return

    counts = clean.value_counts()
    if int(counts.min()) < 2 or min(n_train, n_test) < len(counts):
        raise ProblemSpecError(
            "insufficient_classification_split",
            "test_size",
            "训练集和测试集必须覆盖全部类别",
        )

    # StratifiedShuffleSplit rounds class allocations. Verify the actual rounded
    # allocation instead of only checking the aggregate split sizes.
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_indices, test_indices = next(splitter.split(np.zeros(len(clean)), clean.to_numpy()))
    if clean.iloc[train_indices].nunique() != len(counts) or clean.iloc[test_indices].nunique() != len(counts):
        raise ProblemSpecError(
            "insufficient_classification_split",
            "test_size",
            "训练集和测试集必须覆盖全部类别",
        )


def validate_cv(spec: ProblemSpec, target: pd.Series, cv_folds: int) -> None:
    clean = target.dropna()
    available = len(clean) if spec.resolved_type == ProblemType.REGRESSION else int(clean.value_counts().min())
    if available < cv_folds:
        raise ProblemSpecError(
            "insufficient_cv_samples",
            "cv_folds",
            f"当前任务每折至少需要一个样本，可用上限为 {available}",
        )
