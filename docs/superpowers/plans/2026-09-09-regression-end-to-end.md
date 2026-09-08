# 回归任务端到端支持实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 AutoML、手动训练、Agent 工作流、模型注册和预测实验室中提供可自动识别、可显式覆盖、可训练并可预测的完整回归任务链路。

**Architecture:** 新增唯一的 `ProblemSpec` 任务规范层，由数据管线、模型工厂和三个训练入口共同消费；后端负责推断、能力与校验，前端只消费规范和能力响应。分类路径保持兼容，新产物以 schema v3 保存任务类型，回归预测返回连续浮点值并将分类专属字段置空。

**Tech Stack:** Python 3.11/3.12、FastAPI、pandas、scikit-learn、LightGBM、XGBoost、CatBoost、Optuna、joblib、Vue 3、TypeScript、Element Plus、Vitest、pytest。

**Spec:** `docs/superpowers/specs/2026-09-09-regression-end-to-end-design.md`

## Global Constraints

- 对外规范值固定为 `auto`、`binary_classification`、`multiclass_classification`、`regression`；输入兼容 `binary`、`multiclass`。
- 数值目标自动推断阈值固定为 `min(20, ceil(sqrt(valid_rows)))`，只有整数型低基数数值目标判为多分类。
- 回归算法固定为 LightGBM、XGBoost、CatBoost、随机森林和线性回归，不引入新依赖。
- 回归指标固定为 RMSE、MAE、R²；默认 RMSE，RMSE/MAE 最小化，R² 最大化。
- 新产物写入 `artifact_schema_version: 3`；旧 API 路径、旧分类请求和旧产物保持兼容。
- 回归预测的 `prediction` 为 `float`，`encoded_prediction` 和 `probability` 为 `null`。
- 可预见的不兼容输入在入队前返回 HTTP 422 和稳定字段级错误对象。
- 所有实现遵循 RED → GREEN → REFACTOR；不得先写生产代码后补测试。
- 不修改 Docker、CI、MLflow、监控、分布式任务或前端 E2E；这些属于后续里程碑。

## File Ownership and Execution Order

| Lane | 独占文件 | 可开始条件 |
| --- | --- | --- |
| Foundation | `core/automl/problem_spec.py`、`tests/test_problem_spec.py` | 立即 |
| Data/model | `backend/api/_dataset.py`、`model_code/model_factory.py`、对应测试 | Task 1 完成 |
| Manual API | `backend/api/training.py`、`tests/test_training_api.py` | Task 2 完成 |
| AutoML API | `backend/api/automl.py`、`tests/test_automl_api.py` | Task 2 完成 |
| Workflow | `core/automl/automl_optimizer.py`、`core/automl/automl_agent.py`、`core/llm/agent_schemas.py`、`backend/api/workflow.py`、工作流测试 | Task 2 完成 |
| Registry | `backend/api/models.py`、`tests/test_model_registry_api.py` | Task 2 完成 |
| Frontend contract | `frontend/src/api/problem.ts`、`frontend/src/api/*.ts` 测试，以及 nullable 字段所需的最小模板防护 | Task 1 完成 |
| Frontend views | 每个智能体独占自己负责的 `.vue` 与同名测试 | Task 7 及对应后端任务完成 |
| Acceptance | `tests/test_regression_end_to_end.py`、`README.md` | Tasks 3–10 完成 |

Task 1 和 Task 2 是串行基础。Task 2 与 Task 7 可在 Task 1 后并行；Tasks 3–6 可在
Task 2 后并行；Tasks 8–10 可在 Task 7 和各自依赖的后端任务后并行。每个任务合入前
进行规格符合性复审和代码质量复审，复审者不得是该任务的实现者。

---

### Task 1: 统一任务规范与能力契约

**Files:**
- Create: `core/automl/problem_spec.py`
- Modify: `core/automl/__init__.py`
- Create: `tests/test_problem_spec.py`

**Interfaces:**
- Consumes: `pandas.Series` 目标列和用户请求字符串。
- Produces: `ProblemType`、`ProblemSpecError`、`ProblemSpec`、`resolve_problem_spec()`、`capabilities_payload()`、`resolve_metric()`、`validate_algorithm()`、`validate_train_test_split()`、`validate_cv()`。
- `ProblemSpec.to_dict()` 和 `ProblemSpec.from_dict()` 是后续持久化任务使用的唯一序列化接口。

- [ ] **Step 1: 写任务推断和显式覆盖的失败测试**

```python
import numpy as np
import pandas as pd
import pytest

from core.automl.problem_spec import ProblemSpecError, resolve_problem_spec


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
def test_resolve_problem_spec_auto(values, expected):
    spec = resolve_problem_spec(pd.Series(values, name="target"), "target", "auto")
    assert spec.resolved_type.value == expected
    assert spec.source == "auto_inferred"
    assert spec.target.unique_count == len(pd.unique(values))
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
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_problem_spec.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'core.automl.problem_spec'`.

- [ ] **Step 3: 实现规范对象、推断和能力表**

```python
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, Sequence

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
METRIC_DIRECTION = {"roc_auc": "maximize", "accuracy": "maximize", "f1": "maximize", "rmse": "minimize", "mae": "minimize", "r2": "maximize"}
ALIASES = {"binary": ProblemType.BINARY, "multiclass": ProblemType.MULTICLASS}
EXPLANATIONS = {
    "categorical_target": "目标列为类别数据，按唯一值数量识别为分类任务",
    "two_numeric_values": "数值目标只有两个不同值，识别为二分类任务",
    "low_cardinality_integer_target": "整数目标的唯一值数量低于分类阈值，识别为多分类任务",
    "numeric_continuous_target": "数值目标具有连续取值，识别为回归任务",
}


class ProblemSpecError(ValueError):
    def __init__(self, code: str, field: str, message: str, allowed: Sequence[str] | None = None):
        super().__init__(message)
        self.code, self.field, self.message, self.allowed = code, field, message, allowed

    def as_detail(self) -> Dict[str, Any]:
        detail = {"code": self.code, "field": self.field, "message": self.message}
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
    def metric_directions(self) -> Dict[str, str]:
        return {metric: METRIC_DIRECTION[metric] for metric in self.metrics}

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, payload: Dict[str, Any]) -> "ProblemSpec":
        return cls(
            requested_type=normalize_problem_type(payload["requested_type"]),
            resolved_type=normalize_problem_type(payload["resolved_type"]),
            source=payload["source"],
            reason=payload["reason"],
            explanation=payload["explanation"],
            target=TargetProfile(**payload["target"]),
        )


def normalize_problem_type(value: str | None) -> ProblemType:
    normalized = (value or "auto").strip().lower()
    if normalized in ALIASES:
        return ALIASES[normalized]
    try:
        return ProblemType(normalized)
    except ValueError as exc:
        allowed = [item.value for item in ProblemType]
        raise ProblemSpecError("unsupported_problem_type", "problem_type", f"不支持的任务类型：{value}", allowed) from exc


def validate_target_for_type(clean: pd.Series, resolved: ProblemType, convertible_numeric: bool) -> None:
    unique_count = int(clean.nunique(dropna=True))
    if resolved == ProblemType.BINARY and unique_count != 2:
        raise ProblemSpecError("invalid_binary_target", "problem_type", "二分类目标必须恰好有两个不同值")
    if resolved == ProblemType.MULTICLASS and unique_count < 3:
        raise ProblemSpecError("invalid_multiclass_target", "problem_type", "多分类目标至少需要三个不同值")
    if resolved == ProblemType.REGRESSION and not convertible_numeric:
        raise ProblemSpecError("non_numeric_regression_target", "problem_type", "回归任务的目标列必须是有限数值")


def build_problem_spec(requested: ProblemType, resolved: ProblemType, target_col: str, clean: pd.Series, integer_like: bool, reason: str) -> ProblemSpec:
    source = "auto_inferred" if requested == ProblemType.AUTO else "user_override"
    explanation = EXPLANATIONS.get(reason, f"用户显式选择了{resolved.value}")
    profile = TargetProfile(target_col, str(clean.dtype), len(clean), int(clean.nunique()), float(clean.nunique() / len(clean)), integer_like)
    return ProblemSpec(requested, resolved, source, reason, explanation, profile)


def resolve_problem_spec(target: pd.Series, target_col: str, requested_type: str | None = "auto") -> ProblemSpec:
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
        validate_target_for_type(clean, resolved, convertible_numeric)
    return build_problem_spec(requested, resolved, target_col, clean, integer_like, reason)
```

将以上对象和函数连同 `resolve_problem_spec()` 写入同一文件，并从
`core/automl/__init__.py` 导出公开接口。

- [ ] **Step 4: 补充能力、划分和 CV 失败测试并实现**

```python
from core.automl.problem_spec import (
    capabilities_payload,
    resolve_metric,
    validate_algorithm,
    validate_cv,
    validate_train_test_split,
)


def test_regression_capabilities_and_metric_direction():
    spec = resolve_problem_spec(pd.Series(np.linspace(0.1, 9.9, 40)), "price", "auto")
    assert resolve_metric(spec, "auto") == "rmse"
    assert spec.metric_directions["rmse"] == "minimize"
    assert "linear_regression" in spec.algorithms
    with pytest.raises(ProblemSpecError, match="logistic_regression"):
        validate_algorithm(spec, "logistic_regression")
    assert capabilities_payload()["problem_types"]["regression"]["default_metric"] == "rmse"


def test_regression_split_and_cv_require_enough_rows():
    spec = resolve_problem_spec(pd.Series([1.1, 2.2, 3.3, 4.4]), "price", "regression")
    validate_train_test_split(spec, pd.Series([1.1, 2.2, 3.3, 4.4]), 0.5)
    with pytest.raises(ProblemSpecError):
        validate_train_test_split(spec, pd.Series([1.1, 2.2, 3.3]), 0.5)
    with pytest.raises(ProblemSpecError):
        validate_cv(spec, pd.Series([1.1, 2.2, 3.3, 4.4]), 5)
```

实现以下能力与校验函数：

```python
def capabilities_payload() -> Dict[str, Any]:
    return {
        "problem_types": {
            problem_type.value: {
                "algorithms": list(ALGORITHMS[problem_type]),
                "metrics": list(METRICS[problem_type]),
                "default_metric": DEFAULT_METRIC[problem_type],
                "metric_directions": {metric: METRIC_DIRECTION[metric] for metric in METRICS[problem_type]},
            }
            for problem_type in (ProblemType.BINARY, ProblemType.MULTICLASS, ProblemType.REGRESSION)
        }
    }


def resolve_metric(spec: ProblemSpec, requested_metric: str | None) -> str:
    metric = (requested_metric or "auto").strip().lower()
    if metric == "auto":
        return DEFAULT_METRIC[spec.resolved_type]
    if metric not in spec.metrics:
        raise ProblemSpecError("incompatible_metric", "optimization_metric", f"指标 {metric} 不适用于 {spec.resolved_type.value}", spec.metrics)
    return metric


def validate_algorithm(spec: ProblemSpec, algorithm: str) -> None:
    if algorithm not in spec.algorithms:
        raise ProblemSpecError("incompatible_algorithm", "algorithm", f"算法 {algorithm} 不适用于 {spec.resolved_type.value}", spec.algorithms)


def validate_train_test_split(spec: ProblemSpec, target: pd.Series, test_size: float) -> None:
    clean = target.dropna()
    n_test = math.ceil(len(clean) * test_size)
    n_train = len(clean) - n_test
    if spec.resolved_type == ProblemType.REGRESSION:
        if len(clean) < 4 or min(n_train, n_test) < 2:
            raise ProblemSpecError("insufficient_regression_split", "test_size", "回归训练集和测试集都至少需要两条记录")
        return
    counts = clean.value_counts()
    if int(counts.min()) < 2 or min(n_train, n_test) < len(counts):
        raise ProblemSpecError("insufficient_classification_split", "test_size", "训练集和测试集必须覆盖全部类别")


def validate_cv(spec: ProblemSpec, target: pd.Series, cv_folds: int) -> None:
    clean = target.dropna()
    available = len(clean) if spec.resolved_type == ProblemType.REGRESSION else int(clean.value_counts().min())
    if available < cv_folds:
        raise ProblemSpecError("insufficient_cv_samples", "cv_folds", f"当前任务每折至少需要一个样本，可用上限为 {available}")
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_problem_spec.py -q`

Expected: PASS.

- [ ] **Step 5: 验证新模块覆盖率并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_problem_spec.py --cov=core.automl.problem_spec --cov-report=term-missing --cov-fail-under=90 -q`

Expected: PASS and `core/automl/problem_spec.py` line coverage >= 90%.

```bash
git add core/automl/problem_spec.py core/automl/__init__.py tests/test_problem_spec.py
git commit -m "feat: add shared ML problem specification"
```

### Task 2: 回归数据管线、模型工厂、指标与推理

**Files:**
- Modify: `model_code/model_factory.py:20-249`
- Modify: `backend/api/_dataset.py:120-459`
- Modify: `tests/test_model_factory.py`
- Modify: `tests/test_dataset_pipeline.py`

**Interfaces:**
- Consumes: `ProblemSpec` and canonical `resolved_type` from Task 1.
- Produces: `TrainBundle.problem_spec`、`prepare_train_pipeline(df, target_col, key_cols, test_size, random_state, problem_spec)`、`compute_regression_metrics()` and schema-v3 behavior in `predict_with_artifact()`.
- `ModelFactory.build(algorithm, task_type, params)` continues accepting legacy `binary`/`multiclass` while adding canonical task names and `regression`.

- [ ] **Step 1: 写五种回归模型和不兼容算法的失败测试**

```python
@pytest.mark.parametrize(
    ("algorithm", "params"),
    [
        ("lightgbm", {"n_estimators": 5}),
        ("xgboost", {"n_estimators": 5, "max_depth": 2}),
        ("catboost", {"iterations": 5, "depth": 2}),
        ("random_forest", {"n_estimators": 5, "max_depth": 2}),
        ("linear_regression", {}),
    ],
)
def test_factory_builds_regressors(algorithm, params):
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = X[:, 0] * 2.5 - X[:, 1]
    model = ModelFactory.build(algorithm, task_type="regression", params=params)
    model.fit(X, y)
    prediction = model.predict(X[:2])
    assert prediction.shape == (2,)
    assert np.issubdtype(prediction.dtype, np.floating)


def test_factory_rejects_task_incompatible_linear_models():
    with pytest.raises(ValueError, match="logistic_regression"):
        ModelFactory.build("logistic_regression", task_type="regression")
    with pytest.raises(ValueError, match="linear_regression"):
        ModelFactory.build("linear_regression", task_type="binary")
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_model_factory.py -q`

Expected: FAIL because regression builders and `linear_regression` do not exist.

- [ ] **Step 2: 最小化扩展模型工厂**

在 `DISPLAY_NAMES` 和支持算法中增加 `linear_regression`，让 `build()` 先规范化任务
类型并验证 Task 1 的能力表。各 builder 按类型选择 estimator：

```python
if resolved == "regression":
    return lgb.LGBMRegressor(**defaults)
return lgb.LGBMClassifier(**defaults)

if resolved == "regression":
    return xgb.XGBRegressor(**defaults)
return xgb.XGBClassifier(**defaults)

if resolved == "regression":
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**defaults)

if resolved == "regression":
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**defaults)

if algorithm == "linear_regression":
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**params)
```

回归默认参数不得包含 `class_weight`、`num_class`、`classes_count` 或分类 objective；
XGBoost 使用 `objective="reg:squarederror"` 和 `eval_metric="rmse"`。

- [ ] **Step 3: 写回归划分、指标和 schema-v3 推理失败测试**

```python
def test_prepare_regression_pipeline_keeps_continuous_target():
    df = pd.DataFrame({"x": range(20), "price": np.linspace(10.25, 40.75, 20)})
    spec = resolve_problem_spec(df["price"], "price", "auto")
    bundle = prepare_train_pipeline(df, "price", [], test_size=0.25, problem_spec=spec)
    assert bundle.label_encoder is None
    assert bundle.problem_spec.resolved_type.value == "regression"
    assert bundle.y_train.dtype.kind == "f"
    assert set(bundle.y_train) <= set(df["price"])


def test_compute_regression_metrics_returns_only_regression_metrics():
    model = LinearRegression().fit([[0], [1], [2], [3]], [1.0, 3.0, 5.0, 7.0])
    metrics = compute_regression_metrics(model, [[4], [5]], [9.0, 11.0])
    assert metrics == {"rmse": pytest.approx(0.0), "mae": pytest.approx(0.0), "r2": pytest.approx(1.0)}


def test_schema_v3_regression_prediction_has_no_class_fields():
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
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_dataset_pipeline.py -q`

Expected: FAIL because the shared pipeline always label-encodes and regression metrics are missing.

- [ ] **Step 4: 实现任务感知的数据管线与推理**

将签名改为：

```python
def prepare_train_pipeline(
    df: pd.DataFrame,
    target_col: str,
    key_cols: List[str],
    *,
    test_size: Optional[float] = None,
    random_state: int = 42,
    problem_spec: Optional[ProblemSpec] = None,
) -> TrainBundle:
```

未传 `problem_spec` 时调用 `resolve_problem_spec(df[target_col], target_col, "auto")`。
分类分支保持 LabelEncoder 和 stratify；回归分支执行：

```python
y = pd.to_numeric(clean_df[target_col], errors="raise").astype(float)
validate_train_test_split(problem_spec, y, test_size)
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, y, test_size=test_size, random_state=random_state, stratify=None
)
label_encoder = None
```

增加 `compute_regression_metrics()`，使用 `root_mean_squared_error`、
`mean_absolute_error` 和 `r2_score`；若当前 sklearn 无 `root_mean_squared_error`，使用
`math.sqrt(mean_squared_error(y_test, y_pred))` 保持兼容。

`predict_with_artifact()` 先从 `problem_type` 或 `problem_spec.resolved_type` 判断 schema-v3
回归；回归分支不得 `int()` 强转预测，直接返回 `float(raw_prediction)` 和两个 `None`。
没有 schema-v3 元数据的旧产物继续走原分类/legacy 分支。

- [ ] **Step 5: 运行数据与模型回归测试并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_model_factory.py tests/test_dataset_pipeline.py -q`

Expected: PASS, including all pre-existing classification tests.

```bash
git add model_code/model_factory.py backend/api/_dataset.py tests/test_model_factory.py tests/test_dataset_pipeline.py
git commit -m "feat: add regression models and data pipeline"
```

### Task 3: 手动训练 API 与能力端点

**Files:**
- Modify: `backend/api/training.py:50-550`
- Modify: `tests/test_training_api.py`

**Interfaces:**
- Consumes: Task 1 `ProblemSpec` APIs and Task 2 task-aware pipeline/factory/metrics.
- Produces: `GET /api/training/capabilities`; dataset/job/result snapshots containing `problem_spec`; manual regression artifact schema v3.

- [ ] **Step 1: 写能力端点、上传推断和入队前校验失败测试**

```python
def _upload_regression_dataset(client: TestClient, rows: int = 60) -> str:
    x = np.linspace(-5.0, 5.0, rows)
    frame = pd.DataFrame({"x": x, "price": 3.0 * x + 7.0})
    response = client.post(
        "/api/training/datasets",
        files={"file": ("prices.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "auto"},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_training_capabilities_expose_regression():
    client = TestClient(app)
    response = client.get("/api/training/capabilities")
    assert response.status_code == 200
    regression = response.json()["problem_types"]["regression"]
    assert regression["default_metric"] == "rmse"
    assert "linear_regression" in regression["algorithms"]


def test_training_upload_auto_detects_regression():
    client = TestClient(app)
    df = pd.DataFrame({"x": range(40), "price": np.linspace(10.1, 80.9, 40)})
    response = client.post(
        "/api/training/datasets",
        files={"file": ("prices.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "auto"},
    )
    assert response.status_code == 201
    assert response.json()["problem_spec"]["resolved_type"] == "regression"


def test_training_rejects_incompatible_algorithm_before_enqueue():
    client = TestClient(app)
    regression_dataset_id = _upload_regression_dataset(client)
    response = client.post(
        "/api/training/jobs",
        json={"dataset_id": regression_dataset_id, "problem_type": "regression", "algorithm": "logistic_regression", "params": {}},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "algorithm"
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_training_api.py -q`

Expected: FAIL because new request fields and capability endpoint are missing.

- [ ] **Step 2: 扩展数据集/任务持久化和请求模型**

给 `TrainingDataset`、`TrainingJob` 增加 `problem_spec: Dict[str, Any]` 并在
`_persist_*`/`_restore_*` 中保存；恢复旧记录时若字段缺失，从数据集目标重新推断，
旧任务快照没有数据时降级为旧分类语义。

```python
class TrainingJobRequest(BaseModel):
    dataset_id: str
    algorithm: str
    problem_type: str = "auto"
    params: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0, le=10000)


@router.get("/capabilities")
async def get_training_capabilities():
    return capabilities_payload()
```

上传接收 `problem_type: str = Form("auto")`，捕获 `ProblemSpecError` 后返回
`HTTPException(status_code=422, detail=exc.as_detail())`。

- [ ] **Step 3: 先写手动回归训练结果测试，再实现训练分支**

```python
def test_manual_linear_regression_completes_with_regression_metrics():
    client = TestClient(app)
    regression_dataset_id = _upload_regression_dataset(client)
    created = client.post(
        "/api/training/jobs",
        json={"dataset_id": regression_dataset_id, "problem_type": "regression", "algorithm": "linear_regression", "params": {}, "test_size": 0.25},
    )
    job = _wait_for_job(client, created.json()["job_id"])
    assert job["status"] == "completed"
    result = client.get(f"/api/training/jobs/{job['job_id']}/result").json()
    assert set(result["metrics"]) == {"rmse", "mae", "r2"}
    assert result["confusion_matrix"] is None
    assert result["threshold_calibration"] is None
    assert result["problem_spec"]["resolved_type"] == "regression"
```

在创建任务时重新解析用户覆盖，调用 `validate_algorithm()`、
`validate_train_test_split()`，再入队。`_run_training_job()` 从 job 还原 spec，传给
`prepare_train_pipeline()` 和 `ModelFactory.build()`；回归调用
`compute_regression_metrics()`，分类保留阈值校准与分类指标。

schema-v3 artifact 必须包含：

```python
{
    "artifact_schema_version": 3,
    "problem_type": spec.resolved_type.value,
    "problem_spec": spec.to_dict(),
    "metric_directions": spec.metric_directions,
    "model": model,
    "preprocessor": preprocessor,
    "label_encoder": bundle.label_encoder,
    "feature_columns": bundle.feature_columns,
    "metrics": metrics,
}
```

- [ ] **Step 4: 运行手动训练和分类兼容测试并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_training_api.py tests/test_dataset_pipeline.py tests/test_model_factory.py -q`

Expected: PASS.

```bash
git add backend/api/training.py tests/test_training_api.py
git commit -m "feat: support regression in manual training API"
```

### Task 4: AutoML 回归搜索、指标方向与恢复

**Files:**
- Modify: `backend/api/automl.py:40-594`
- Modify: `tests/test_automl_api.py`

**Interfaces:**
- Consumes: Task 1 capability/metric APIs and Task 2 factory/pipeline/metrics.
- Produces: AutoML `problem_type`/`optimization_metric=auto` request support, task-aware CV and Optuna direction, schema-v3 result/artifact.

- [ ] **Step 1: 写 AutoML 回归创建和非法组合失败测试**

```python
def _regression_csv(rows: int = 80) -> tuple[str, io.BytesIO, str]:
    x = np.linspace(-10.0, 10.0, rows)
    frame = pd.DataFrame({"x": x, "price": 4.5 * x + 12.0})
    return "regression.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv"


def _create_regression_automl_job(client: TestClient, metric: str = "rmse") -> str:
    file_name, file_body, content_type = _regression_csv()
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "price", "problem_type": "auto", "optimization_metric": metric, "candidate_models": '["linear_regression"]', "n_trials": 1, "cv_folds": 3},
    )
    assert response.status_code == 201
    return response.json()["job_id"]


def test_automl_regression_auto_metric_and_problem_spec():
    client = TestClient(app)
    file_name, file_body, content_type = _regression_csv(rows=80)
    response = client.post(
        "/api/automl/jobs",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "price", "problem_type": "auto", "optimization_metric": "auto", "candidate_models": '["linear_regression"]', "n_trials": 1, "cv_folds": 3},
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
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_automl_api.py -q`

Expected: FAIL because regression metrics and request type are unsupported.

- [ ] **Step 2: 扩展任务契约、推荐和入队前验证**

给 `AutoMLJob` 增加 `problem_spec: Dict[str, Any]` 并在持久化/恢复/快照中保存。
`create_automl_job()` 增加 `problem_type: str = Form("auto")`，把指标默认改为 `auto`。

创建流程固定为：解析目标 → `resolve_problem_spec()` → `resolve_metric()` → 解析候选模型
→ 对每个显式候选调用 `validate_algorithm()` → 构建一次未拟合 bundle 验证 split/CV →
持久化并入队。候选未提供时使用 `spec.algorithms`，`_recommend_models()` 只能在能力集
内排序，不能给回归推荐逻辑回归。

- [ ] **Step 3: 写 Optuna 原始指标与回归结果失败测试**

```python
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
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_automl_api.py -q`

Expected: FAIL because `_run_job()` always uses StratifiedKFold and classification metrics.

- [ ] **Step 4: 实现任务感知的搜索和产物**

```python
def _scoring(metric: str, resolved_type: str) -> str:
    mapping = {"accuracy": "accuracy", "f1": "f1_weighted", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    if metric == "roc_auc":
        return "roc_auc" if resolved_type == "binary_classification" else "roc_auc_ovr_weighted"
    return mapping[metric]


def _readable_cv_value(metric: str, scores: np.ndarray) -> float:
    mean = float(np.mean(scores))
    return -mean if metric in {"rmse", "mae"} else mean
```

分类使用 `StratifiedKFold`，回归使用 `KFold(n_splits=job.cv_folds, shuffle=True,
random_state=42)`。Study direction 读取 `spec.metric_directions[job.optimization_metric]`；
objective 返回 `_readable_cv_value()`。`error_score` 改为 `"raise"`，不能用 0 掩盖失败。

最终训练按任务选择 `compute_classification_metrics()` 或
`compute_regression_metrics()`；只有二分类执行阈值校准。结果和 artifact 写入 Task 3
定义的 schema-v3 元数据，持久化恢复后 `problem_spec` 不丢失。

- [ ] **Step 5: 运行 AutoML、恢复与分类回归测试并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_automl_api.py tests/test_runtime_state.py tests/test_model_factory.py -q`

Expected: PASS.

```bash
git add backend/api/automl.py tests/test_automl_api.py
git commit -m "feat: support regression in AutoML jobs"
```

### Task 5: Agent 工作流与共享 AutoML 优化器

**Files:**
- Modify: `core/automl/automl_optimizer.py:20-432`
- Modify: `core/automl/automl_agent.py:19-437`
- Modify: `core/llm/agent_schemas.py:151-189`
- Modify: `backend/api/workflow.py:95-607`
- Modify: `tests/test_workflow_pipeline.py`
- Create: `tests/test_automl_optimizer_regression.py`

**Interfaces:**
- Consumes: Task 1 `ProblemSpec`, Task 2 `ModelFactory` and schema-v3 inference.
- Produces: task-aware `AutoMLOptimizer`, workflow query parameter `problem_type`, workflow status/SSE/artifact task metadata.

- [ ] **Step 1: 写优化器回归方向和模型构建失败测试**

```python
def test_optimizer_minimizes_readable_rmse():
    X = pd.DataFrame({"x": np.linspace(0, 10, 60)})
    y = pd.Series(4.0 * X["x"] + 2.0)
    config = {"task_type": "regression", "optimization_metric": "rmse", "candidate_models": ["linear_regression"], "n_trials": 1, "search_spaces": {"linear_regression": {}}}
    optimizer = AutoMLOptimizer(X, y, config, cv_splits=3, n_jobs=1)
    study, model = optimizer.optimize(n_trials=1, show_progress=False)
    assert study.direction.name == "MINIMIZE"
    assert study.best_value >= 0
    assert model.predict([[12.0]])[0] == pytest.approx(50.0, rel=0.02)
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_automl_optimizer_regression.py -q`

Expected: FAIL because the optimizer builds classifiers and always maximizes.

- [ ] **Step 2: 让优化器委托统一模型工厂并规范化分数**

保留搜索空间解析，删除五套重复 estimator builder 的调用路径；`_build_model()` 在得到
params 后调用：

```python
safe_params = dict(params)
if model_name in {"lightgbm", "xgboost", "random_forest"}:
    safe_params["n_jobs"] = 1
if model_name == "catboost":
    safe_params.update({"thread_count": 1, "verbose": False})
return ModelFactory.build(model_name, task_type=self.task_type, params=safe_params)
```

CV 选择和 scorer 与 Task 4 相同；误差 scorer 的负数在 objective 返回前转为正的
RMSE/MAE，Study direction 读取 `METRIC_DIRECTION[self.metric]`。为
`linear_regression` 接受空搜索空间。保留现有 callback 透传行为。

- [ ] **Step 3: 写工作流建议修正、回归保存和预测失败测试**

```python
import json


def prepare_workflow_task(tmp_path) -> str:
    task_id = "task_ab12cd34"
    task_dir = tmp_path / "output" / task_id
    task_dir.mkdir(parents=True)
    x = np.linspace(-8.0, 8.0, 80)
    frame = pd.DataFrame({"age": np.linspace(20.0, 60.0, 80), "city_code": ["北京", "上海"] * 40, "order_fee": 5.0 * x + 100.0})
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps({"target": "order_fee", "user_query": "预测订单金额"}, ensure_ascii=False),
        encoding="utf-8",
    )
    strategy = FeatureStrategy(actions=[FeatureAction(column="age", action="fillna_mean", reason="测试填充")])
    (task_dir / "feature_strategy.json").write_text(strategy.model_dump_json(), encoding="utf-8")
    return task_id


def read_workflow_sse(task_id: str, problem_type: str, n_trials: int, cv_folds: int) -> list[dict]:
    events = []
    with TestClient(app) as client:
        with client.stream(
            "GET",
            "/api/workflow/step3_train_stream",
            params={"task_id": task_id, "problem_type": problem_type, "n_trials": n_trials, "cv_folds": cv_folds},
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line.removeprefix("data:").strip()))
    return events


def test_workflow_regression_corrects_agent_suggestion_and_predicts_float(tmp_path, monkeypatch):
    task_id = prepare_workflow_task(tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(
        workflow,
        "recommend_automl_config",
        lambda data_features, scenario: {
            "task_type": "binary_classification",
            "optimization_metric": "roc_auc",
            "candidate_models": ["logistic_regression"],
            "search_spaces": {},
            "rationale": "test mismatch",
        },
    )
    events = read_workflow_sse(task_id, problem_type="auto", n_trials=10, cv_folds=3)
    assert events[-1]["status"] == "DONE"
    assert events[-1]["problem_spec"]["resolved_type"] == "regression"
    assert events[-1]["suggested_type"] == "binary_classification"
    artifact = joblib.load(tmp_path / "saved" / f"model_{task_id}.pkl")
    assert artifact["problem_type"] == "regression"
    prediction = TestClient(app).post("/api/workflow/predict", json={"model_version": task_id, "features": {"age": 40, "city_code": "北京"}}).json()
    assert isinstance(prediction["prediction"], float)
    assert prediction["probability"] is None
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_workflow_pipeline.py tests/test_automl_optimizer_regression.py -q`

Expected: FAIL because workflow label-encodes every target and hard-codes classification.

- [ ] **Step 4: 实现 Agent 建议与最终任务规范的协调**

`extract_data_features()` 调用 `resolve_problem_spec(df[target_col], target_col, "auto")`，不再保留独立
`_infer_task_type()` 规则。`AutoMLConfig.candidate_models` 增加 `linear_regression`。
规则降级配置依据 `data_features["task_type"]` 选择默认指标和兼容模型，核心分支固定为：

```python
if data_features["task_type"] == "regression":
    metric = "rmse"
    candidates = ["random_forest", "linear_regression"]
else:
    metric = "roc_auc" if data_features["task_type"] == "binary_classification" else "accuracy"
    candidates = ["lightgbm", "catboost", "random_forest"]
```

`generate_config()` 必须把当前 `data_features` 传给规则降级函数；LLM 调用失败、JSON
解析失败和缺字段三条路径都使用同一任务感知的降级配置。

在 workflow 中定义可 monkeypatch 的纯函数：

```python
def recommend_automl_config(data_features: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    try:
        return orchestrator.run_automl_agent(json.dumps(data_features, ensure_ascii=False), scenario).model_dump()
    except Exception:
        return AutoMLAgent().generate_config(data_features, scenario)
```

`train_stream()` 增加 `problem_type: str = Query("auto")`。特征工程后对真实目标解析
最终 spec；Agent 类型只写入 `suggested_type`，不能覆盖自动推断。候选模型和指标与
spec 协调，不兼容项记录在 `agent_corrections`；空候选时回归使用
`linear_regression`、分类使用 `random_forest` 作为确定性基线。分类才
LabelEncode；回归目标保留 float。

SSE `DONE`、任务状态和 schema-v3 artifact 返回/保存
`problem_spec`、`suggested_type`、`agent_corrections`、最终指标名和值。状态中的通用字段
改为 `best_metric` 和 `best_value`，同时仅对 legacy 分类产物保留 `best_auc`。

- [ ] **Step 5: 运行工作流、优化器和 schema 测试并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_workflow_pipeline.py tests/test_automl_optimizer_regression.py tests/test_llm_tuning.py -q`

Expected: PASS, including callback and旧分类预测测试。

```bash
git add core/automl/automl_optimizer.py core/automl/automl_agent.py core/llm/agent_schemas.py backend/api/workflow.py tests/test_workflow_pipeline.py tests/test_automl_optimizer_regression.py
git commit -m "feat: enable regression in agent workflow"
```

### Task 6: 模型注册与批量预测语义

**Files:**
- Modify: `backend/api/models.py:42-183`
- Modify: `tests/test_model_registry_api.py`

**Interfaces:**
- Consumes: schema-v3 artifacts and Task 2 `predict_with_artifact()`.
- Produces: registry records with `problem_type`/`problem_spec`/`legacy`; regression-safe batch summary.

- [ ] **Step 1: 写 schema-v3 注册记录和回归批量预测失败测试**

```python
from sklearn.linear_model import LinearRegression

from core.automl.problem_spec import resolve_problem_spec


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
    listed = TestClient(app).get("/api/models").json()["models"][0]
    assert listed["problem_type"] == "regression"
    assert listed["legacy"] is False
    batch = TestClient(app).post(
        "/api/models/manual%3Amanual_training_reg/batch-predict",
        json={"rows": [{"x": 3}, {"x": 4}]},
    ).json()
    assert batch["summary"] == {"count": 2, "positive_rate": None, "average_probability": None}
    assert batch["rows"][0]["encoded_prediction"] is None
```

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_model_registry_api.py -q`

Expected: FAIL because registry records only expose metrics and summary assumes numeric encodings.

- [ ] **Step 2: 一次加载产物并提取注册元数据**

把 `_try_load_metrics()` 替换为返回 metrics、problem metadata 的读取函数；`_record()`
输出：

```python
{
    "model_id": model_id,
    "source": source,
    "name": path.stem,
    "artifact_path": relative_path,
    "metrics": metrics,
    "problem_type": problem_type,
    "problem_spec": problem_spec,
    "legacy": artifact.get("artifact_schema_version", 0) < 3,
    "updated_at": path.stat().st_mtime,
}
```

schema-v3 回归的 `_summarize_predictions()` 直接返回两个分类汇总空值；分类与 legacy
继续按类别数计算。不得对 `encoded_prediction=None` 执行 `float()`。

- [ ] **Step 3: 运行 registry、旧产物与训练到预测测试并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_model_registry_api.py tests/test_predict.py -q`

Expected: PASS, including v2 string-label and multiclass positive-rate tests.

```bash
git add backend/api/models.py tests/test_model_registry_api.py
git commit -m "feat: expose regression model metadata and predictions"
```

### Task 7: 前端共享任务类型与 API 契约

**Files:**
- Create: `frontend/src/api/problem.ts`
- Modify: `frontend/src/api/training.ts`
- Modify: `frontend/src/api/automl.ts`
- Modify: `frontend/src/api/workflow.ts`
- Modify: `frontend/src/api/models.ts`
- Modify: `frontend/src/api/training.test.ts`
- Modify: `frontend/src/api/automl.test.ts`
- Modify: `frontend/src/api/workflow.test.ts`
- Modify: `frontend/src/api/models.test.ts`
- Modify: `frontend/src/modules/agent-workflow/views/AgentWorkflowView.vue`
- Modify: `frontend/src/modules/prediction-lab/views/PredictionLabView.vue`

**Interfaces:**
- Consumes: Task 1/Tasks 3–6 JSON field names.
- Produces: strict TypeScript `ProblemType`, `ProblemSpec`, capabilities, nullable prediction and task-aware API calls used by all views.

- [ ] **Step 1: 写表单和可空响应的失败测试**

```typescript
it('serializes problem type and automatic metric', () => {
  const form = buildAutoMLJobForm({ file, targetCol: 'price', problemType: 'regression', nTrials: 3, cvFolds: 3, optimizationMetric: 'auto' })
  expect(form.get('problem_type')).toBe('regression')
  expect(form.get('optimization_metric')).toBe('auto')
})

it('serializes manual dataset and job problem types', () => {
  const form = buildTrainingDatasetForm({ file, targetCol: 'price', problemType: 'auto' })
  expect(form.get('problem_type')).toBe('auto')
})

it('adds problem type to workflow training stream URL', () => {
  expect(api.trainStreamUrl('task_1234abcd', 3, 3, 'regression')).toContain('problem_type=regression')
})
```

Run: `cd frontend && npm test -- --run src/api/training.test.ts src/api/automl.test.ts src/api/workflow.test.ts src/api/models.test.ts`

Expected: FAIL because the shared types and request fields are missing.

- [ ] **Step 2: 创建严格共享类型并扩展 API**

```typescript
export type ProblemType = 'auto' | 'binary_classification' | 'multiclass_classification' | 'regression'
export type ResolvedProblemType = Exclude<ProblemType, 'auto'>
export type MetricDirection = 'minimize' | 'maximize'

export interface ProblemCapabilities {
  algorithms: string[]
  metrics: string[]
  default_metric: string
  metric_directions: Record<string, MetricDirection>
}

export interface ProblemSpec {
  schema_version: 1
  requested_type: ProblemType
  resolved_type: ResolvedProblemType
  source: 'auto_inferred' | 'user_override'
  reason: string
  explanation: string
  target: { column: string; dtype: string; valid_rows: number; unique_count: number; unique_ratio: number; integer_like: boolean }
  capabilities: ProblemCapabilities
}
```

`TrainingAlgorithm` 和 `AutoMLModel` 增加 `linear_regression`；`AutoMLMetric` 增加
`auto | rmse | mae | r2`。上传和建任务输入增加 `problemType` 并序列化为
`problem_type`。`trainingApi.getCapabilities()` 调用 `/api/training/capabilities`。

Workflow status/event 增加 `problem_spec`、`suggested_type`、`best_metric`、`best_value`；
`streamTraining()` 和 URL builder 增加 problem type。Registry record 增加 metadata；
所有 prediction 的 `encoded_prediction`、`probability` 改为 `number | null`。

同时只做两处编译兼容防护：Agent Workflow 和 Prediction Lab 模板把直接
`probability.toFixed(4)` 改为 `probability == null ? '-' : probability.toFixed(4)`。
是否隐藏回归专属卡片由 Task 10 负责。

- [ ] **Step 3: 运行 API 单测和类型检查并提交**

Run: `cd frontend && npm test -- --run src/api/training.test.ts src/api/automl.test.ts src/api/workflow.test.ts src/api/models.test.ts`

Expected: PASS.

Run: `cd frontend && npm run typecheck`

Expected: PASS；nullable 字段的最小模板防护保证契约提交不破坏主分支类型检查。

```bash
git add frontend/src/api/problem.ts frontend/src/api/training.ts frontend/src/api/automl.ts frontend/src/api/workflow.ts frontend/src/api/models.ts frontend/src/api/training.test.ts frontend/src/api/automl.test.ts frontend/src/api/workflow.test.ts frontend/src/api/models.test.ts frontend/src/modules/agent-workflow/views/AgentWorkflowView.vue frontend/src/modules/prediction-lab/views/PredictionLabView.vue
git commit -m "feat: add frontend regression API contracts"
```

### Task 8: 手动训练前端交互

**Files:**
- Modify: `frontend/src/modules/manual-training/views/ManualTrainingView.vue`
- Modify: `frontend/src/modules/manual-training/ManualTrainingView.test.ts`

**Interfaces:**
- Consumes: `trainingApi.getCapabilities()` and Task 7 types.
- Produces: problem selector, compatible algorithm reset, ProblemSpec card and task-aware metric view.

- [ ] **Step 1: 写回归选择与展示失败测试**

把现有 hoisted mock 增加 `getCapabilities: vi.fn()`、`uploadDataset: vi.fn()`，并让
`vi.mock('@/api/training')` 的同名方法引用这两个 mock。

```typescript
const regressionSpec = {
  schema_version: 1,
  requested_type: 'regression',
  resolved_type: 'regression',
  source: 'user_override',
  reason: 'user_selected_regression',
  explanation: '用户显式选择了回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: { algorithms: ['linear_regression', 'random_forest'], metrics: ['rmse', 'mae', 'r2'], default_metric: 'rmse', metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' } },
} as const
const capabilities = {
  problem_types: {
    binary_classification: { algorithms: ['logistic_regression', 'random_forest'], metrics: ['roc_auc', 'accuracy', 'f1'], default_metric: 'roc_auc', metric_directions: { roc_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' } },
    multiclass_classification: { algorithms: ['logistic_regression', 'random_forest'], metrics: ['accuracy', 'f1', 'roc_auc'], default_metric: 'accuracy', metric_directions: { accuracy: 'maximize', f1: 'maximize', roc_auc: 'maximize' } },
    regression: regressionSpec.capabilities,
  },
}
const regressionDataset = { dataset_id: 'dataset_reg', filename: 'reg.csv', rows: 80, columns: 2, target_col: 'price', created_at: '2026-09-09T00:00:00', preview: [], problem_spec: regressionSpec }
const regressionResult = { job_id: 'job_reg', dataset_id: 'dataset_reg', algorithm: 'linear_regression', params: {}, metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 }, confusion_matrix: null, threshold_calibration: null, feature_importance: [], train_size: 60, test_size: 20, feature_columns: ['x'], problem_spec: regressionSpec }


it('switches to compatible regression algorithm and metrics', async () => {
  trainingMocks.getCapabilities.mockResolvedValue(capabilities)
  trainingMocks.uploadDataset.mockResolvedValue(regressionDataset)
  const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
  const vm = wrapper.vm as unknown as { problemType: string; algorithm: string; applyProblemType: () => void; result: Record<string, unknown> | null }
  vm.problemType = 'regression'
  vm.applyProblemType()
  expect(vm.algorithm).not.toBe('logistic_regression')
  vm.result = regressionResult
  await wrapper.vm.$nextTick()
  expect(wrapper.text()).toContain('RMSE')
  expect(wrapper.text()).toContain('R²')
  expect(wrapper.text()).not.toContain('最佳阈值')
})
```

Run: `cd frontend && npm test -- --run src/modules/manual-training/ManualTrainingView.test.ts`

Expected: FAIL because task selection and regression result rendering do not exist.

- [ ] **Step 2: 实现任务选择、能力过滤和结果视图**

新增 `problemType = ref<ProblemType>('auto')`、能力加载和 `activeProblemSpec`。上传发送
problem type，创建任务再次发送当前选择。`compatibleAlgorithms` 从显式类型能力或上传
响应 spec 计算；当前算法失效时选择能力数组第一个值并调用参数 preset。

回归增加 `linear_regression: {}` preset。结果模板通过
`result.problem_spec.resolved_type === 'regression'` 切换：回归展示 RMSE/MAE/R²，分类
保持 ROC-AUC/Accuracy/F1/阈值。ProblemSpec 卡展示 explanation、source、unique_count。

- [ ] **Step 3: 运行组件测试、类型检查并提交**

Run: `cd frontend && npm test -- --run src/modules/manual-training/ManualTrainingView.test.ts src/modules/interaction-regression.test.ts`

Expected: PASS.

```bash
git add frontend/src/modules/manual-training/views/ManualTrainingView.vue frontend/src/modules/manual-training/ManualTrainingView.test.ts
git commit -m "feat: add regression controls to manual training UI"
```

### Task 9: AutoML 前端交互与指标方向

**Files:**
- Modify: `frontend/src/modules/automl/views/AutoMLView.vue`
- Create: `frontend/src/modules/automl/AutoMLView.test.ts`

**Interfaces:**
- Consumes: Task 7 problem types and AutoML response.
- Produces: auto/explicit problem selector, compatible model/metric controls, task-aware result cards and optimization chart labels.

- [ ] **Step 1: 写提交契约和回归结果失败测试**

```typescript
import { mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import type { AutoMLResult } from '@/api/automl'
import type { ProblemType } from '@/api/problem'
import AutoMLView from './views/AutoMLView.vue'


const automlMocks = vi.hoisted(() => ({
  createJob: vi.fn(),
  getJob: vi.fn(),
  getResult: vi.fn(),
  cancelJob: vi.fn(),
  saveResult: vi.fn(),
  streamJob: vi.fn(),
}))
const trainingApiMocks = vi.hoisted(() => ({ getCapabilities: vi.fn() }))
vi.mock('@/api/automl', () => ({ automlApi: automlMocks }))
vi.mock('@/api/training', () => ({ trainingApi: trainingApiMocks }))

const regressionSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'regression',
  source: 'auto_inferred',
  reason: 'numeric_continuous_target',
  explanation: '数值目标具有连续取值，识别为回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: { algorithms: ['linear_regression'], metrics: ['rmse', 'mae', 'r2'], default_metric: 'rmse', metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' } },
} as const
const capabilities = { problem_types: { binary_classification: { algorithms: ['logistic_regression'], metrics: ['roc_auc'], default_metric: 'roc_auc', metric_directions: { roc_auc: 'maximize' } }, multiclass_classification: { algorithms: ['random_forest'], metrics: ['accuracy'], default_metric: 'accuracy', metric_directions: { accuracy: 'maximize' } }, regression: regressionSpec.capabilities } }
const regressionResult = { job_id: 'job_reg', status: 'completed', best_model: 'linear_regression', best_params: { model_type: 'linear_regression' }, best_value: 0.1, metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 }, trial_history: [], decision_summary: {}, feature_columns: ['x'], threshold_calibration: null, problem_spec: regressionSpec }


it('submits automatic problem and metric defaults', async () => {
  trainingApiMocks.getCapabilities.mockResolvedValue(capabilities)
  automlMocks.createJob.mockResolvedValue({ job_id: 'job_reg', status: 'queued', progress: 0, current_trial: 0, n_trials: 3, cv_folds: 3, optimization_metric: 'rmse', target_col: 'price', decision_summary: {}, problem_spec: regressionSpec })
  automlMocks.streamJob.mockImplementation(async (_jobId, onEvent) => onEvent({ event: 'done', status: 'completed' }))
  automlMocks.getResult.mockResolvedValue(regressionResult)
  const wrapper = mount(AutoMLView, { global: { plugins: [ElementPlus] } })
  const vm = wrapper.vm as unknown as { datasetFile: File; problemType: ProblemType; startJob: () => Promise<void> }
  vm.datasetFile = new File(['x,price\n1,2.5'], 'reg.csv', { type: 'text/csv' })
  vm.problemType = 'auto'
  await vm.startJob()
  expect(automlMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({ problemType: 'auto', optimizationMetric: 'auto' }))
})

it('shows regression metrics and inference explanation', async () => {
  trainingApiMocks.getCapabilities.mockResolvedValue(capabilities)
  const wrapper = mount(AutoMLView, { global: { plugins: [ElementPlus] } })
  const vm = wrapper.vm as unknown as { result: AutoMLResult | null }
  vm.result = regressionResult
  await wrapper.vm.$nextTick()
  expect(wrapper.text()).toContain('连续数值目标')
  expect(wrapper.text()).toContain('RMSE')
  expect(wrapper.text()).toContain('MAE')
  expect(wrapper.text()).not.toContain('ROC-AUC：')
})
```

Run: `cd frontend && npm test -- --run src/modules/automl/AutoMLView.test.ts`

Expected: FAIL because problem selection and dynamic result cards are missing.

- [ ] **Step 2: 实现自动/显式配置与动态结果**

自动模式固定发送 `optimizationMetric: 'auto'` 且省略候选模型；显式模式从能力响应
选择候选和指标，类型切换时移除不兼容项。任务创建后以响应 `problem_spec` 为准。

结果卡、试验表和 ECharts y 轴标题使用最终指标。RMSE/MAE 保留原始正数，不做取负；
tooltip 明确“越小越好”，R²/分类指标显示“越大越好”。回归结果展示三项指标和
ProblemSpec，不渲染分类指标。

- [ ] **Step 3: 运行 AutoML 组件与交互测试并提交**

Run: `cd frontend && npm test -- --run src/modules/automl/AutoMLView.test.ts src/modules/interaction-regression.test.ts`

Expected: PASS.

```bash
git add frontend/src/modules/automl/views/AutoMLView.vue frontend/src/modules/automl/AutoMLView.test.ts
git commit -m "feat: add regression controls to AutoML UI"
```

### Task 10: 工作流、模型注册与预测实验室前端

**Files:**
- Modify: `frontend/src/modules/agent-workflow/views/AgentWorkflowView.vue`
- Create: `frontend/src/modules/agent-workflow/AgentWorkflowView.test.ts`
- Modify: `frontend/src/modules/model-registry/views/ModelRegistryView.vue`
- Modify: `frontend/src/modules/model-registry/ModelRegistryView.test.ts`
- Modify: `frontend/src/modules/prediction-lab/views/PredictionLabView.vue`
- Modify: `frontend/src/modules/prediction-lab/PredictionLabView.test.ts`

**Interfaces:**
- Consumes: Task 7 workflow/registry/prediction types and Tasks 5–6 responses.
- Produces: suggested-vs-final task display, dynamic primary metrics, regression-safe prediction rendering and legacy marker.

- [ ] **Step 1: 写三个页面的回归失败测试**

在新建的 `AgentWorkflowView.test.ts` 中写完整 mock：

```typescript
import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import AgentWorkflowView from './views/AgentWorkflowView.vue'

const workflowMocks = vi.hoisted(() => ({
  getTaskStatus: vi.fn(),
  retrieveData: vi.fn(),
  generateFeatures: vi.fn(),
  streamTraining: vi.fn(),
  getImportance: vi.fn(async () => []),
  listModels: vi.fn(async () => ({ models: [] })),
  predict: vi.fn(),
  trainStreamUrl: vi.fn(() => '/api/workflow/step3_train_stream'),
}))
vi.mock('@/api/workflow', () => ({ workflowApi: workflowMocks }))

const regressionWorkflowStatus = {
  task_id: 'task_ab12cd34',
  status: 'completed',
  best_metric: 'rmse',
  best_value: 0.12,
  suggested_type: 'binary_classification',
  agent_corrections: ['移除不兼容模型 logistic_regression', '指标由 roc_auc 修正为 rmse'],
  problem_spec: {
    schema_version: 1,
    requested_type: 'auto',
    resolved_type: 'regression',
    source: 'auto_inferred',
    reason: 'numeric_continuous_target',
    explanation: '数值目标具有连续取值，识别为回归任务',
    target: { column: 'order_fee', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
    capabilities: { algorithms: ['linear_regression'], metrics: ['rmse', 'mae', 'r2'], default_metric: 'rmse', metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' } },
  },
}


it('shows corrected workflow problem type without probability', async () => {
  workflowMocks.getTaskStatus.mockResolvedValue(regressionWorkflowStatus)
  const wrapper = mount(AgentWorkflowView, { global: { plugins: [ElementPlus] } })
  const vm = wrapper.vm as unknown as { taskId: string; refreshStatus: () => Promise<void>; predictionResult: Record<string, unknown> | null }
  vm.taskId = 'task_ab12cd34'
  await vm.refreshStatus()
  vm.predictionResult = { status: 'success', prediction: 123.45, probability: null, encoded_prediction: null }
  await flushPromises()
  expect(wrapper.text()).toContain('建议：二分类')
  expect(wrapper.text()).toContain('最终：回归')
  expect(wrapper.text()).toContain('RMSE')
  expect(wrapper.text()).not.toContain('预测概率')
})
```

在 `ModelRegistryView.test.ts` 的模型列表 mock 中加入一个 schema-v3 回归记录和一个
legacy 分类记录，再加入：

```typescript
it('uses r2 as a regression registry metric and marks legacy records', async () => {
  const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
  await flushPromises()
  expect(wrapper.text()).toContain('0.9900')
  expect(wrapper.text()).toContain('Legacy')
})
```

回归记录固定为：

```typescript
{
  model_id: 'manual:reg',
  source: 'manual_training',
  name: 'reg',
  artifact_path: 'saved/manual_training/reg.joblib',
  metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 },
  problem_type: 'regression',
  problem_spec: {
    schema_version: 1,
    requested_type: 'auto',
    resolved_type: 'regression',
    source: 'auto_inferred',
    reason: 'numeric_continuous_target',
    explanation: '数值目标具有连续取值，识别为回归任务',
    target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
    capabilities: { algorithms: ['linear_regression'], metrics: ['rmse', 'mae', 'r2'], default_metric: 'rmse', metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' } },
  },
  legacy: false,
  updated_at: 2,
}
```

legacy 记录固定为
`{ model_id: 'legacy:old', source: 'legacy', name: 'old', artifact_path: 'model_code/saved/old.joblib', metrics: { accuracy: 0.8 }, problem_type: 'binary_classification', problem_spec: null, legacy: true, updated_at: 1 }`。

在 `PredictionLabView.test.ts` 把模型列表首项改为上述 `manual:reg` 回归记录，并加入：

```typescript
it('renders regression prediction without probability cards', async () => {
  modelMocks.predict.mockResolvedValue({ model_id: 'manual:reg', source: 'manual_training', input_features: { x: 3 }, prediction: 7.5, encoded_prediction: null, probability: null })
  const wrapper = mount(PredictionLabView, { global: { plugins: [ElementPlus] } })
  await flushPromises()
  const vm = wrapper.vm as unknown as { runPrediction: () => Promise<void> }
  await vm.runPrediction()
  await flushPromises()
  expect(wrapper.text()).toContain('7.5')
  expect(wrapper.text()).not.toContain('预测概率')
})
```

Run: `cd frontend && npm test -- --run src/modules/agent-workflow/AgentWorkflowView.test.ts src/modules/model-registry/ModelRegistryView.test.ts src/modules/prediction-lab/PredictionLabView.test.ts`

Expected: FAIL because pages assume AUC/probability and ignore task metadata.

- [ ] **Step 2: 实现工作流任务类型与结果显示**

Agent 页面增加 task selector，并把值传入 `streamTraining()`。状态卡使用
`best_metric`/`best_value`；有 `suggested_type` 时同时展示建议和最终类型，存在
`agent_corrections` 时显示修正原因。预测 probability 为 null 时隐藏概率字段。

- [ ] **Step 3: 实现 registry 和 prediction 的任务感知显示**

Registry `metricText(record)`：回归按优化指标、R²、RMSE、MAE 顺序，分类保持
ROC-AUC/val_auc/Accuracy；legacy 显示徽标。Prediction Lab 从当前选中 record 的
`problem_type` 和响应空值控制卡片/列：回归隐藏正类率、平均概率和行概率，分类保持
现状。模板中不得对 nullable probability 直接调用 `.toFixed()`。

- [ ] **Step 4: 运行三个页面、类型检查并提交**

Run: `cd frontend && npm test -- --run src/modules/agent-workflow/AgentWorkflowView.test.ts src/modules/model-registry/ModelRegistryView.test.ts src/modules/prediction-lab/PredictionLabView.test.ts src/modules/interaction-regression.test.ts`

Expected: PASS.

Run: `cd frontend && npm run typecheck`

Expected: PASS.

```bash
git add frontend/src/modules/agent-workflow frontend/src/modules/model-registry frontend/src/modules/prediction-lab
git commit -m "feat: render regression workflow and predictions"
```

### Task 11: 端到端验收用例与用户文档

**Files:**
- Create: `tests/test_regression_end_to_end.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all backend and artifact interfaces from Tasks 1–6.
- Produces: deterministic acceptance test covering train → save → registry → predict and documented regression workflow.

- [ ] **Step 1: 写线性数据端到端验收测试**

```python
import io
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.api.models as model_registry
import backend.api.training as manual_training
from backend.main import app


def wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 15.0
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/training/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] == "completed":
            return last
        if last["status"] in {"failed", "cancelled"}:
            raise AssertionError(f"Regression training ended as {last['status']}: {last.get('error')}")
        time.sleep(0.1)
    raise AssertionError(f"Regression training timed out: {last}")


def test_manual_regression_train_save_registry_predict(tmp_path, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(manual_training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manual_training, "ARTIFACT_DIR", tmp_path / "saved" / "manual_training")
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    x = np.linspace(-20.0, 20.0, 120)
    frame = pd.DataFrame({"x": x, "segment": np.where(x > 0, "positive", "negative"), "price": 3.25 * x + 17.0})
    upload = client.post(
        "/api/training/datasets",
        files={"file": ("linear.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
        data={"target_col": "price", "problem_type": "auto"},
    )
    assert upload.json()["problem_spec"]["resolved_type"] == "regression"
    created = client.post(
        "/api/training/jobs",
        json={"dataset_id": upload.json()["dataset_id"], "algorithm": "linear_regression", "problem_type": "auto", "params": {}, "test_size": 0.25, "random_state": 42},
    )
    completed = wait_for_job(client, created.json()["job_id"])
    result = client.get(f"/api/training/jobs/{completed['job_id']}/result").json()
    assert result["metrics"]["r2"] >= 0.95
    assert set(result["metrics"]) == {"rmse", "mae", "r2"}
    saved = client.post(f"/api/training/jobs/{completed['job_id']}/save").json()
    model_id = f"manual:manual_training_{completed['job_id']}"
    registered = client.get(f"/api/models/{quote(model_id, safe='')}").json()
    assert registered["problem_type"] == "regression"
    prediction = client.post(f"/api/models/{quote(model_id, safe='')}/predict", json={"features": {"x": 8.0, "segment": "positive"}}).json()
    assert prediction["prediction"] == pytest.approx(43.0, abs=0.5)
    assert prediction["encoded_prediction"] is None
    assert prediction["probability"] is None
    assert (tmp_path / saved["artifact_path"]).is_file()
```

测试 helper 必须轮询明确终态并设置 15 秒截止时间；失败时输出 job error，不使用固定
sleep 作为唯一同步手段。产物写入 pytest 临时目录，由 pytest 安全清理。

- [ ] **Step 2: 运行端到端用例并确认通过**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_regression_end_to_end.py -q`

Expected: PASS with R² >= 0.95 and continuous registry prediction.

- [ ] **Step 3: 更新 README 的任务类型与回归示例**

在 API/功能说明中增加：自动识别与覆盖规则、五个回归算法、三项指标、AutoML
`optimization_metric=auto`、回归预测字段语义，以及一条 curl/界面操作示例。明确
时间序列和多输出回归不在当前支持范围。

- [ ] **Step 4: 运行文档相关 smoke test 并提交**

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_frontend_static.py tests/test_regression_end_to_end.py -q`

Expected: PASS.

```bash
git add tests/test_regression_end_to_end.py README.md
git commit -m "test: add regression end-to-end acceptance coverage"
```

### Task 12: 全量验证、独立复审与里程碑验收

**Files:**
- Modify only if a verification or review finding requires a scoped fix.
- Review: all files changed since commit `5a8e992`.

**Interfaces:**
- Consumes: completed Tasks 1–11.
- Produces: fresh verification evidence and a review report with zero unresolved high/medium findings.

- [ ] **Step 1: 运行后端定向回归套件**

Run:

```bash
.venv/bin/python -m pytest -o addopts='' \
  tests/test_problem_spec.py \
  tests/test_model_factory.py \
  tests/test_dataset_pipeline.py \
  tests/test_training_api.py \
  tests/test_automl_api.py \
  tests/test_automl_optimizer_regression.py \
  tests/test_workflow_pipeline.py \
  tests/test_model_registry_api.py \
  tests/test_regression_end_to_end.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 2: 运行 Python 全量测试与覆盖门禁**

Run: `.venv/bin/python -m pytest tests/ --cov=model_code --cov=web_app --cov-report=term-missing --cov-fail-under=20`

Expected: all main-suite tests PASS and total coverage >= 20%.

Run: `.venv/bin/python -m pytest -o addopts='' tests/test_problem_spec.py --cov=core.automl.problem_spec --cov-report=term-missing --cov-fail-under=90 -q`

Expected: task-spec module coverage >= 90%.

- [ ] **Step 3: 运行前端全量门禁**

Run:

```bash
cd frontend
npm test
npm run typecheck
npm run build
npm audit --omit=dev --audit-level=high
```

Expected: all Vitest tests PASS, typecheck PASS, production build PASS, 0 high/critical production dependency vulnerabilities.

- [ ] **Step 4: 运行 Python 安全门禁**

Run:

```bash
.venv/bin/python -m bandit -r backend core model_code -q -lll
.venv/bin/python -m pip_audit -r requirements.txt
```

Expected: 0 medium/high Bandit findings and 0 known dependency vulnerabilities. If the installed CLI module is named `pip-audit`, run `.venv/bin/pip-audit -r requirements.txt` with the same expected result.

- [ ] **Step 5: 派发独立规格复审和代码质量复审**

规格复审输入：设计规格、实施计划和 `git diff 5a8e992..HEAD`；逐条核对 Tasks 1–11
与验收标准。代码质量复审重点检查数据泄漏、错误指标方向、分类兼容性、持久化恢复、
nullable 前端字段和路径安全。任何高/中优先级发现必须增加能复现问题的失败测试，再按
RED → GREEN 修复并重跑相关门禁。

- [ ] **Step 6: 检查工作区和提交最终验收修复**

Run: `git status --short --branch && git log --oneline 5a8e992..HEAD`

Expected: 工作区干净；日志包含每个独立任务的原子提交。

若复审产生修复，确认 `git diff --name-only` 只包含本里程碑文件后提交所有已跟踪修复：

```bash
git add -u backend core model_code frontend/src tests README.md
git commit -m "fix: resolve regression milestone review findings"
```

最终重新执行 Steps 1–4；只有所有命令的新鲜输出均通过且复审无未解决高/中问题，才
宣布里程碑 1 验收完成并进入“Vue + FastAPI 生产 Docker”里程碑。
