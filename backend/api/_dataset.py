"""数据集上传与预处理的共享实现。

training / automl / analysis 三个路由原先各自维护一份相同的
读取、目标列识别、键列识别、训练矩阵构建与指标计算代码，统一收敛于此。

训练侧使用 sklearn Pipeline（先划分、再仅在训练集拟合），pipeline 随
artifact 一起保存；推理侧通过 predict_with_artifact 复用同一 pipeline，
从构造上保证训练/推理一致。
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from core.automl.problem_spec import (
    ProblemSpec,
    ProblemSpecError,
    ProblemType,
    resolve_problem_spec,
    validate_train_test_split,
)

logger = logging.getLogger(__name__)

LABEL_NAME_CANDIDATES = ("label", "target", "y", "class", "is_fraud", "是否违约", "标签", "目标")
KEY_COLUMN_TOKENS = ("id", "uuid", "key", "编号", "序号", "订单号", "用户号")
MAX_UPLOAD_BYTES = int(os.getenv("ML_MAX_UPLOAD_MB", "50")) * 1024 * 1024
SUPPORTED_DATA_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls"}


def _normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """执行不依赖标签的轻量清洗，并保留可解释的数据问题。"""
    if df.empty and not len(df.columns):
        raise ValueError("数据文件为空")

    columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    if any(not col for col in columns):
        raise ValueError("数据中存在空字段名，请补全后重试")
    duplicates = sorted({col for col in columns if columns.count(col) > 1})
    if duplicates:
        raise ValueError(f"数据中存在重复字段名：{', '.join(duplicates[:5])}")

    clean = df.copy()
    clean.columns = columns
    clean = clean.dropna(axis=0, how="all").reset_index(drop=True)
    if clean.empty:
        raise ValueError("数据文件没有有效记录")
    return clean


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """兼容 UTF-8/GB18030 等常见编码，并自动识别逗号、制表符和竖线。"""
    import chardet

    detected = chardet.detect(raw[: min(len(raw), 1_000_000)]).get("encoding")
    encodings = ["utf-8-sig", detected, "gb18030"]
    for encoding in dict.fromkeys(item for item in encodings if item):
        try:
            text = raw.decode(encoding)
            return pd.read_csv(StringIO(text), sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("无法识别 CSV 编码或分隔符；请使用 UTF-8、GB18030 编码的规范 CSV")


async def read_uploaded_dataframe(file: UploadFile) -> pd.DataFrame:
    """读取上传的 CSV/Excel 文件为 DataFrame。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_DATA_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 CSV、TXT、XLSX 或 XLS 数据文件")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件超过 {limit_mb} MB 上传限制")
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(BytesIO(raw))
        else:
            df = _read_csv_bytes(raw)
        return _normalise_dataframe(df)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"数据文件读取失败：{exc}") from exc


def dataframe_quality_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """生成供 API 和前端复用的数据质量摘要，不修改原始数据。"""
    feature_df = df.drop(columns=[target_col], errors="ignore") if target_col else df
    numeric = feature_df.select_dtypes(include=[np.number])
    infinite_values = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(numeric.columns) else 0
    missing_rates = feature_df.isna().mean()
    constant_columns = [str(col) for col in feature_df.columns if feature_df[col].nunique(dropna=True) <= 1]
    high_missing_columns = [str(col) for col, rate in missing_rates.items() if float(rate) >= 0.5]
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate": round(float(df.isna().sum().sum()) / max(df.size, 1), 6),
        "infinite_values": infinite_values,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
    }


def detect_target_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    """按请求值或常见标签名识别目标列，缺省取最后一列。"""
    if requested:
        if requested not in df.columns:
            raise HTTPException(status_code=400, detail=f"目标列不存在：{requested}")
        return requested
    label_names = {name.lower() for name in LABEL_NAME_CANDIDATES}
    for col in df.columns:
        if str(col).strip().lower() in label_names:
            return str(col)
    return str(df.columns[-1])


def detect_key_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """识别高基数的 ID/编号类键列（训练时应剔除）。"""
    keys: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        name = str(col).strip().lower()
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.85 and any(token in name for token in KEY_COLUMN_TOKENS):
            keys.append(str(col))
    return keys


@dataclass
class TrainBundle:
    """prepare_train_pipeline 的输出：划分后的原始特征 + 未拟合的预处理器。"""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    label_encoder: Any
    problem_spec: ProblemSpec
    preprocessor: Any  # 未 fit 的 ColumnTransformer，仅在 X_train 上构建
    feature_columns: List[str]  # 原始输入列名（推理时的对齐基准）


def build_preprocessor(X: pd.DataFrame, *, ohe_min_frequency: int = 2, ohe_max_categories: int = 100):
    """按列类型构建预处理 ColumnTransformer（调用方负责在训练集上 fit）。

    数值列：中位数填充 + 标准化；类别列：常量填充 + One-Hot
    （handle_unknown='ignore' 容忍推理期新类别，min_frequency / max_categories
    限制高基数展开规模，并尽量保留稀疏矩阵以降低内存占用）。
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=ohe_min_frequency,
                                max_categories=ohe_max_categories,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("未找到可训练特征列")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def prepare_train_pipeline(
    df: pd.DataFrame,
    target_col: str,
    key_cols: List[str],
    *,
    test_size: Optional[float] = None,
    random_state: int = 42,
    problem_spec: Optional[ProblemSpec] = None,
) -> TrainBundle:
    """清洗 → 剔键列 → 按任务处理目标 → 划分特征 → 构建预处理器。

    与旧 get_dummies 方案的关键差异：类别集合、缺失值统计量、缩放参数
    都只在训练集上学习，测试集/推理期只做 transform，杜绝划分前泄漏。
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if problem_spec is None:
        problem_spec = resolve_problem_spec(df[target_col], target_col, "auto")

    clean_df = df.copy()
    clean_df = clean_df.loc[clean_df[target_col].notna()].reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("目标列全部为空，无法训练")

    feature_df = clean_df.drop(columns=[target_col, *key_cols], errors="ignore")
    if feature_df.shape[1] == 0:
        raise ValueError("未找到可训练特征列")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        feature_df.loc[:, numeric_cols] = feature_df.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    if test_size is None:
        test_size = 0.25 if len(feature_df) >= 20 else 0.3

    if problem_spec.resolved_type == ProblemType.REGRESSION:
        y = pd.to_numeric(clean_df[target_col], errors="raise").astype(float)
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("回归任务的目标列必须是有限数值")
        label_encoder = None
        stratify = None
    else:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(clean_df[target_col]), name=target_col)
        if y.nunique() < 2:
            raise ValueError("目标变量至少需要 2 个类别")
        class_counts = y.value_counts()
        if int(class_counts.min()) < 2:
            rare_labels = label_encoder.inverse_transform(class_counts[class_counts < 2].index.to_numpy())
            raise ValueError(f"每个类别至少需要 2 条记录；样本不足的类别：{', '.join(map(str, rare_labels))}")
        stratify = y

    validate_train_test_split(problem_spec, y, test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    preprocessor = build_preprocessor(X_train)
    return TrainBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        problem_spec=problem_spec,
        preprocessor=preprocessor,
        feature_columns=[str(col) for col in feature_df.columns],
    )


def with_runtime_params(algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """合并各算法的运行时安全参数（限制线程数、避免并行嵌套等）。"""
    runtime = dict(params)
    if algorithm in {"lightgbm", "xgboost", "random_forest"}:
        runtime["n_jobs"] = 1
    if algorithm == "logistic_regression":
        runtime.setdefault("solver", "liblinear")
        runtime.setdefault("max_iter", 1000)
        runtime.pop("n_jobs", None)
    if algorithm == "catboost":
        runtime.setdefault("thread_count", 1)
        runtime.setdefault("verbose", False)
    return runtime


def calibrate_binary_threshold(model, X, y, *, random_state: int = 42) -> Dict[str, Any]:
    """从训练集内部留出的校准集选择 F1 阈值，避免使用测试集调阈值。"""
    from sklearn.base import clone
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split

    counts = pd.Series(y).value_counts()
    if len(counts) != 2 or int(counts.min()) < 3:
        return {"threshold": 0.5, "source": "default", "calibration_size": 0}
    calibration_size = max(len(counts), math.ceil(len(y) * 0.2))
    if len(y) - calibration_size < len(counts):
        return {"threshold": 0.5, "source": "default", "calibration_size": 0}
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X,
        y,
        test_size=calibration_size,
        random_state=random_state,
        stratify=y,
    )
    candidate = clone(model)
    candidate.fit(X_fit, y_fit)
    if not hasattr(candidate, "predict_proba"):
        return {"threshold": 0.5, "source": "default", "calibration_size": 0}
    scores = candidate.predict_proba(X_cal)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_cal, scores)
    if not len(thresholds):
        return {"threshold": 0.5, "source": "default", "calibration_size": 0}
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.nanargmax(f1_scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "source": "training_calibration_split",
        "calibration_size": int(len(y_cal)),
    }


def compute_classification_metrics(model, X_test, y_test, *, decision_threshold: float = 0.5) -> Dict[str, Any]:
    """计算分类指标；二分类阈值必须由测试集之外的数据预先确定。"""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    if proba is not None and proba.shape[1] == 2:
        y_pred = (proba[:, 1] >= decision_threshold).astype(int)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "best_threshold": float(decision_threshold),
    }
    if proba is not None and proba.shape[1] == 2:
        y_score = proba[:, 1]
        try:
            # 单类别时 sklearn 仅告警并返回 nan：nan 同样省略，不伪装成 accuracy
            value = float(roc_auc_score(y_test, y_score))
            if not np.isnan(value):
                metrics["roc_auc"] = value
        except Exception:
            pass
    elif proba is not None:
        try:
            value = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
            if not np.isnan(value):
                metrics["roc_auc"] = value
        except Exception:
            pass
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def compute_regression_metrics(model, X_test, y_test) -> Dict[str, float]:
    """计算回归指标。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:  # pragma: no cover - 仅覆盖旧版 sklearn
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    else:
        rmse = root_mean_squared_error(y_test, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = float(r2_score(y_test, y_pred, force_finite=False))
    if not np.isfinite(r2):
        raise ValueError("R² 无法计算：测试集目标值没有方差，无法生成有限分数")
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": r2,
    }


def save_job_artifact(artifact_dir: Path, prefix: str, job_id: str, artifact: Any, project_root: Path) -> str:
    """将任务产物保存到 artifact_dir 内（含目录约束校验），返回项目相对路径。"""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (artifact_dir / f"{prefix}_{job_id}.joblib").resolve()
    if artifact_dir.resolve() not in artifact_path.parents:
        raise HTTPException(status_code=400, detail="无效的 artifact 路径")
    joblib.dump(artifact, artifact_path)
    return str(artifact_path.relative_to(project_root))


def feature_importance_rows(model: Any, feature_names: List[str], top_n: int = 30) -> List[Dict[str, Any]]:
    """提取特征重要性（One-Hot 展开后的全名），按重要性降序取 Top-N。

    长度失配时按最短对齐并告警（不伪造数据）；模型无重要性属性时返回空列表。
    """
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
    else:
        return []
    names = [str(name) for name in feature_names]
    if len(values) != len(names):
        logger.warning("特征重要性长度(%d)与特征名数量(%d)不一致，按最短对齐", len(values), len(names))
        limit = min(len(values), len(names))
        values, names = values[:limit], names[:limit]
    rows = [{"feature": name, "importance": float(value)} for name, value in zip(names, values)]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:top_n]


@dataclass(frozen=True)
class ArtifactProblemMetadata:
    schema_version: int
    problem_type: ProblemType
    problem_spec: ProblemSpec
    metric_directions: Dict[str, str]


def _artifact_metadata_error(code: str, field: str, message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": code, "field": field, "message": message})


def parse_artifact_problem_metadata(artifact: Dict[str, Any]) -> Optional[ArtifactProblemMetadata]:
    """Strictly parse schema-v3 metadata; return ``None`` only for legacy artifacts."""
    if "artifact_schema_version" not in artifact:
        return None
    schema_version = artifact["artifact_schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, (int, np.integer)):
        raise _artifact_metadata_error(
            "invalid_artifact_schema_version", "artifact_schema_version", "模型产物的 schema 版本必须是整数"
        )
    if int(schema_version) < 3:
        return None
    if int(schema_version) != 3:
        raise _artifact_metadata_error(
            "unsupported_artifact_schema_version", "artifact_schema_version", "不支持该模型产物 schema 版本"
        )

    persisted_type = artifact.get("problem_type")
    canonical_resolved_types = {item.value for item in ProblemType if item != ProblemType.AUTO}
    if not isinstance(persisted_type, str) or persisted_type not in canonical_resolved_types:
        raise _artifact_metadata_error(
            "invalid_artifact_problem_type", "problem_type", "schema-v3 模型产物必须保存规范的已解析任务类型"
        )
    resolved = ProblemType(persisted_type)

    saved_spec = artifact.get("problem_spec")
    try:
        if isinstance(saved_spec, ProblemSpec):
            spec = ProblemSpec.from_dict(saved_spec.to_dict())
        elif isinstance(saved_spec, dict):
            spec = ProblemSpec.from_dict(saved_spec)
        else:
            raise ProblemSpecError("invalid_problem_spec", "problem_spec", "problem_spec 缺失")
    except (ProblemSpecError, KeyError, TypeError, ValueError) as exc:
        raise _artifact_metadata_error(
            "invalid_artifact_problem_spec", "problem_spec", "schema-v3 模型产物的 problem_spec 不完整或无效"
        ) from exc
    if resolved != spec.resolved_type:
        raise _artifact_metadata_error(
            "inconsistent_artifact_problem_type", "problem_type", "模型产物的重复任务类型字段不一致"
        )

    metric_directions = artifact.get("metric_directions")
    if not isinstance(metric_directions, dict) or metric_directions != spec.metric_directions:
        raise _artifact_metadata_error(
            "invalid_artifact_metric_directions", "metric_directions", "模型产物的指标方向缺失或与任务类型不一致"
        )
    return ArtifactProblemMetadata(int(schema_version), resolved, spec, dict(metric_directions))


def resolve_artifact_problem_type(artifact: Dict[str, Any]) -> Optional[ProblemType]:
    """校验 schema-v3 的持久化任务语义；旧产物返回 None。"""
    metadata = parse_artifact_problem_metadata(artifact)
    return metadata.problem_type if metadata is not None else None


def predict_with_artifact(artifact: Dict[str, Any], features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """统一推理入口，按 artifact 形态路由：

    - v2（含 ``preprocessor``）：原始输入列 reindex（缺失→NaN 交 imputer）后
      走训练时保存的 ColumnTransformer，与训练严格一致；
    - workflow（含 ``feature_executor``）：仅 transform（绝不重新 fit），
      残留非数值列兜底转数值，按执行器输出列对齐补 0；
    - legacy：保持旧 get_dummies + 补 0 对齐行为（兼容历史 artifact）。

    每行返回 prediction（有 label_encoder 时为原始标签）、encoded_prediction、
    probability（预测类别的概率，无 predict_proba 时为 None）与键级诊断。
    """
    model = artifact.get("model")
    if model is None:
        raise HTTPException(status_code=400, detail="模型产物缺少可预测模型")
    feature_columns = [str(col) for col in (artifact.get("feature_columns") or [])]
    label_encoder = artifact.get("label_encoder")
    preprocessor = artifact.get("preprocessor")
    executor = artifact.get("feature_executor")
    artifact_problem_type = resolve_artifact_problem_type(artifact)
    is_regression = artifact_problem_type == ProblemType.REGRESSION

    frame = pd.DataFrame(features_list)
    unknown_keys: List[str] = []

    if preprocessor is not None:
        unknown_keys = [str(col) for col in frame.columns if str(col) not in feature_columns]
        missing = [col for col in feature_columns if col not in frame.columns]
        frame = frame.reindex(columns=feature_columns)
        matrix = preprocessor.transform(frame)
        # 训练期启用特征选择时，推理必须复用同一选择器（随 artifact 持久化）
        feature_selector = artifact.get("feature_selector")
        if feature_selector is not None and hasattr(feature_selector, "transform"):
            matrix = feature_selector.transform(matrix)
    else:
        if executor is not None and hasattr(executor, "transform"):
            is_fitted = getattr(executor, "is_fitted", None)
            if not callable(is_fitted) or is_fitted():
                frame = executor.transform(frame)
        else:
            frame = pd.get_dummies(frame, dummy_na=True)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
        missing = [col for col in feature_columns if col not in frame.columns]
        if feature_columns:
            frame = frame.reindex(columns=feature_columns, fill_value=0)
        matrix = frame

    raw_predictions = np.asarray(model.predict(matrix)).ravel()
    if is_regression:
        diagnostics = {
            "unknown_input_keys": unknown_keys,
            "missing_columns_imputed": [str(col) for col in missing],
        }
        return [
            {
                "prediction": float(raw_prediction),
                "encoded_prediction": None,
                "probability": None,
                "diagnostics": diagnostics,
            }
            for raw_prediction in raw_predictions
        ]

    proba = model.predict_proba(matrix) if hasattr(model, "predict_proba") else None
    encoded = raw_predictions
    decision_threshold = artifact.get("decision_threshold")
    classes = list(getattr(model, "classes_", []))
    if proba is not None and proba.shape[1] == 2 and decision_threshold is not None and 1 in classes:
        positive_idx = classes.index(1)
        negative_class = next(item for item in classes if item != 1)
        encoded = np.where(proba[:, positive_idx] >= float(decision_threshold), 1, negative_class)

    diagnostics = {
        "unknown_input_keys": unknown_keys,
        "missing_columns_imputed": [str(col) for col in missing],
    }
    results: List[Dict[str, Any]] = []
    for i in range(len(encoded)):
        enc = int(encoded[i])
        probability = None
        if proba is not None and getattr(proba, "ndim", 1) == 2 and enc in classes:
            probability = float(proba[i, classes.index(enc)])
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([enc])[0]
            if hasattr(prediction, "item"):
                prediction = prediction.item()
        else:
            prediction = float(enc)
        results.append(
            {
                "prediction": prediction,
                "encoded_prediction": float(enc),
                "probability": probability,
                "diagnostics": diagnostics,
            }
        )
    return results
