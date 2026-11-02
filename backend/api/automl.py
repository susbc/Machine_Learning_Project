"""
AutoML jobs API.

The Vue frontend creates a job by uploading a dataset. The backend owns the
whole training lifecycle: target detection, model recommendation, Optuna search,
status recovery, SSE progress, cancellation, result retrieval, and artifact save.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from ._common import validate_entity_id
from backend.runtime_state import TaskQueueFullError, background_tasks, runtime_state
from ._dataset import (
    calibrate_binary_threshold,
    compute_classification_metrics,
    compute_regression_metrics,
    dataframe_quality_report,
    detect_key_columns,
    detect_target_col,
    feature_importance_rows,
    prepare_train_pipeline,
    read_uploaded_dataframe,
    save_job_artifact,
    with_runtime_params,
)
from core.automl.problem_spec import (
    ProblemSpec,
    ProblemSpecError,
    ProblemType,
    resolve_metric,
    resolve_problem_spec,
    validate_algorithm,
    validate_cv,
    validate_train_test_split,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/automl", tags=["AutoML"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "saved" / "automl"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class AutoMLJob:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    n_trials: int
    cv_folds: int
    optimization_metric: str
    target_col: str
    decision_summary: Dict[str, Any]
    problem_spec: Dict[str, Any] = field(default_factory=dict)
    candidate_models: List[str] = field(default_factory=list)
    early_stopping_rounds: int = 0
    min_improvement: float = 0.001
    enable_feature_selection: bool = False
    enable_sampling: bool = False
    sampling_negative_ratio: float = 1.0
    progress: int = 0
    current_trial: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


automl_jobs: Dict[str, AutoMLJob] = {}
jobs_lock = threading.RLock()
JOB_NAMESPACE = "automl_jobs"
INPUT_BLOB_NAMESPACE = "automl_job_inputs"
RESULT_BLOB_NAMESPACE = "automl_job_results"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _persist_job(job: AutoMLJob) -> None:
    has_result = job.result is not None
    if has_result:
        runtime_state.save_blob(RESULT_BLOB_NAMESPACE, job.job_id, job.result)
    runtime_state.put(
        JOB_NAMESPACE,
        job.job_id,
        {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "n_trials": job.n_trials,
            "cv_folds": job.cv_folds,
            "optimization_metric": job.optimization_metric,
            "target_col": job.target_col,
            "decision_summary": job.decision_summary,
            "problem_spec": job.problem_spec,
            "candidate_models": job.candidate_models,
            "early_stopping_rounds": job.early_stopping_rounds,
            "min_improvement": job.min_improvement,
            "enable_feature_selection": job.enable_feature_selection,
            "enable_sampling": job.enable_sampling,
            "sampling_negative_ratio": job.sampling_negative_ratio,
            "progress": job.progress,
            "current_trial": job.current_trial,
            "error": job.error,
            "artifact_path": job.artifact_path,
            "events": job.events,
            "has_result": has_result,
        },
    )


def _restore_job(job_id: str) -> Optional[AutoMLJob]:
    payload = runtime_state.get(JOB_NAMESPACE, job_id)
    if payload is None:
        return None
    restored_payload = dict(payload)
    has_result = bool(restored_payload.pop("has_result", False))
    result = None
    if has_result:
        try:
            result = runtime_state.load_blob(RESULT_BLOB_NAMESPACE, job_id)
        except FileNotFoundError:
            logger.error("Persisted AutoML result is incomplete: %s", job_id)
    if "problem_spec" not in restored_payload:
        restored_payload["problem_spec"] = _legacy_classification_problem_spec()
    if isinstance(result, dict) and "problem_spec" not in result:
        result["problem_spec"] = restored_payload["problem_spec"]
    return AutoMLJob(result=result, **restored_payload)


def _legacy_classification_problem_spec() -> Dict[str, Any]:
    """Provide task metadata for persisted pre-schema-v3 AutoML jobs."""
    payload = resolve_problem_spec(pd.Series([0, 1]), "legacy_target", "binary").to_dict()
    payload["source"] = "legacy_compatibility"
    return payload


def recover_interrupted_automl_jobs() -> int:
    """Mark process-local searches interrupted by a prior server exit as failed."""
    recovered = 0
    for payload in runtime_state.list(JOB_NAMESPACE):
        if payload.get("status") not in {"queued", "running", "cancelling"}:
            continue
        payload["status"] = "failed"
        payload["error"] = "服务曾重启，原 AutoML 进程已中断，请重新提交任务"
        payload["updated_at"] = _now()
        problem_spec = payload.setdefault("problem_spec", _legacy_classification_problem_spec())
        events = payload.setdefault("events", [])
        events.append(
            {
                "timestamp": payload["updated_at"],
                "problem_spec": problem_spec,
                "event": "error",
                "status": "failed",
                "message": payload["error"],
            }
        )
        runtime_state.put(JOB_NAMESPACE, payload["job_id"], payload)
        recovered += 1
    return recovered


def prune_automl_runtime(retention_days: int) -> int:
    """Remove expired terminal AutoML metadata, inputs, events, and results."""
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for payload in runtime_state.list(JOB_NAMESPACE):
        try:
            expired = datetime.fromisoformat(payload["updated_at"]) < cutoff
        except (KeyError, TypeError, ValueError):
            expired = False
        if payload.get("status") not in TERMINAL_STATUSES or not expired:
            continue
        job_id = payload["job_id"]
        runtime_state.delete(JOB_NAMESPACE, job_id, (INPUT_BLOB_NAMESPACE, RESULT_BLOB_NAMESPACE))
        with jobs_lock:
            automl_jobs.pop(job_id, None)
        removed += 1
    return removed


def _safe_job(job_id: str) -> AutoMLJob:
    validate_entity_id(job_id, "job_id")
    with jobs_lock:
        job = automl_jobs.get(job_id)
        if job is None:
            job = _restore_job(job_id)
            if job is not None:
                automl_jobs[job_id] = job
    if job is None:
        raise HTTPException(status_code=404, detail="未找到 AutoML 任务")
    return job


def _job_snapshot(job: AutoMLJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "current_trial": job.current_trial,
        "n_trials": job.n_trials,
        "cv_folds": job.cv_folds,
        "optimization_metric": job.optimization_metric,
        "target_col": job.target_col,
        "decision_summary": job.decision_summary,
        "problem_spec": job.problem_spec,
        "early_stopping_rounds": job.early_stopping_rounds,
        "min_improvement": job.min_improvement,
        "enable_feature_selection": job.enable_feature_selection,
        "enable_sampling": job.enable_sampling,
        "sampling_negative_ratio": job.sampling_negative_ratio,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "artifact_path": job.artifact_path,
    }


def _append_event(job: AutoMLJob, event: Dict[str, Any]) -> None:
    event = {"timestamp": _now(), "problem_spec": job.problem_spec, **event}
    with jobs_lock:
        job.events.append(event)
        job.updated_at = event["timestamp"]
    _persist_job(job)


def _eda_summary(df: pd.DataFrame, target_col: str, key_cols: List[str]) -> Dict[str, Any]:
    feature_df = df.drop(columns=[target_col, *key_cols], errors="ignore")
    target = df[target_col]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_col": target_col,
        "feature_count": int(feature_df.shape[1]),
        "numeric_features": int(len(feature_df.select_dtypes(include=[np.number]).columns)),
        "categorical_features": int(len(feature_df.select_dtypes(include=["object", "category"]).columns)),
        "missing_rate": round(float(feature_df.isna().sum().sum()) / max(feature_df.size, 1), 4),
        "target_classes": int(target.nunique(dropna=True)),
        "key_columns": key_cols,
        "quality": dataframe_quality_report(df, target_col),
    }


def _parse_candidate_models(candidate_models: Optional[str]) -> Optional[List[str]]:
    if not candidate_models:
        return None
    try:
        parsed = json.loads(candidate_models)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="candidate_models 必须是 JSON 数组") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=400, detail="candidate_models 必须是字符串数组")
    if not parsed:
        raise HTTPException(status_code=400, detail="candidate_models 不能为空数组")
    return parsed


def _recommend_models(
    feature_df: pd.DataFrame,
    target: pd.Series,
    requested: Optional[List[str]],
    allowed_models: List[str],
) -> Dict[str, Any]:
    if requested is not None:
        return {
            "candidate_models": requested,
            "recommendation_reason": "使用前端指定候选模型。",
            "recommendation_source": "user",
        }

    try:
        from model_code.auto_selector import AutoModelSelector

        recommendation = AutoModelSelector.recommend(feature_df, target)
        candidates = [item for item in recommendation["candidates"] if item in allowed_models]
        return {
            "candidate_models": candidates[:3] or allowed_models[:3],
            "recommendation_reason": recommendation.get("reason", "启发式模型推荐。"),
            "recommendation_source": "heuristic",
            "selector_analysis": recommendation.get("analysis", {}),
        }
    except Exception as exc:
        logger.warning("AutoML model recommendation failed: %s", exc)
        return {
            "candidate_models": allowed_models[:3],
            "recommendation_reason": f"启发式推荐失败，使用稳健默认候选模型：{exc}",
            "recommendation_source": "fallback",
        }


def _suggest_params(trial, model_name: str) -> Dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "solver": "liblinear",
            "max_iter": 1000,
            "n_jobs": None,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 140),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": 1,
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 240),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 96),
            "n_jobs": 1,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "n_jobs": 1,
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 80, 220),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 9),
            "thread_count": 1,
            "verbose": False,
        }
    if model_name == "linear_regression":
        return {}
    return {}


def _scoring(metric: str, resolved_type: str) -> str:
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    if metric == "roc_auc":
        return "roc_auc" if resolved_type == ProblemType.BINARY.value else "roc_auc_ovr_weighted"
    return mapping[metric]


def _readable_cv_value(metric: str, scores: np.ndarray) -> float:
    mean = float(np.mean(scores))
    return -mean if metric in {"rmse", "mae"} else mean


def _optimization_direction(spec: ProblemSpec, metric: str) -> str:
    return spec.metric_directions[metric]


def _cross_validator(resolved_type: str, cv_folds: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    if resolved_type == ProblemType.REGRESSION.value:
        return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)


def _build_selector_step(task_type: str):
    """特征选择步骤：以随机森林重要性为基准的 SelectFromModel。

    选择器作为 Pipeline 的一环参与 CV，每一折重新 fit，杜绝折间泄漏；
    刻意选用 sklearn 内置的随机森林而不是 LightGBM，避免可选依赖缺失
    时整个训练链路不可用。移植自历史 AutoTrainer.feature_selection。
    """
    from sklearn.feature_selection import SelectFromModel
    from model_code.model_factory import ModelFactory

    base = ModelFactory.build(
        "random_forest",
        task_type=task_type,
        params=with_runtime_params("random_forest", {"n_estimators": 50, "max_depth": 8}),
    )
    return SelectFromModel(base, threshold="median")


def _training_steps(preprocessor, task_type: str, model, enable_feature_selection: bool) -> List:
    """组装训练 Pipeline 步骤；特征选择开启时插入 selector 环节。"""
    steps = [("prep", preprocessor)]
    if enable_feature_selection:
        steps.append(("select", _build_selector_step(task_type)))
    steps.append(("model", model))
    return steps


def _apply_negative_sampling(X_train, y_train, negative_ratio: float, random_state: int = 42):
    """二分类负样本降采样（移植自历史 smart_sampling / adjust_sampling_ratio）。

    只作用训练集：测试集保持原始分布，指标评估不受采样影响。
    返回 (X, y, info)；无法采样（单类样本为空或比例不缩减）时原样返回。
    """
    y = pd.Series(y_train).reset_index(drop=True)
    X = X_train.reset_index(drop=True)
    positive_mask = (y == 1).to_numpy()
    n_positive = int(positive_mask.sum())
    n_negative = int((~positive_mask).sum())
    keep_negative = max(1, int(np.floor(n_negative * negative_ratio)))
    info: Dict[str, Any] = {
        "enabled": True,
        "negative_ratio": float(negative_ratio),
        "original_rows": int(len(y)),
        "positive_rows": n_positive,
        "negative_rows": n_negative,
    }
    if n_positive == 0 or n_negative == 0 or keep_negative >= n_negative:
        info.update({"applied": False, "sampled_rows": int(len(y)), "kept_negative_rows": n_negative})
        return X_train, y_train, info
    rng = np.random.default_rng(random_state)
    kept_negative_idx = rng.choice(np.flatnonzero(~positive_mask), size=keep_negative, replace=False)
    keep_idx = np.concatenate([np.flatnonzero(positive_mask), kept_negative_idx])
    rng.shuffle(keep_idx)
    info.update({"applied": True, "sampled_rows": int(len(keep_idx)), "kept_negative_rows": keep_negative})
    return X.iloc[keep_idx], y.iloc[keep_idx], info


def _run_job(
    job_id: str,
    df: Optional[pd.DataFrame] = None,
    candidate_models: Optional[List[str]] = None,
) -> None:
    import optuna
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from model_code.model_factory import ModelFactory

    job = _safe_job(job_id)
    try:
        if df is None:
            df = runtime_state.load_blob(INPUT_BLOB_NAMESPACE, job_id)
        candidate_models = candidate_models or job.candidate_models or list(job.decision_summary["candidate_models"])
        with jobs_lock:
            job.status = "running"
            job.updated_at = _now()
        _append_event(job, {"event": "status", "status": "running", "message": "AutoML 任务开始"})

        key_cols = job.decision_summary.get("key_columns", [])
        spec = ProblemSpec.from_dict(job.problem_spec) if job.problem_spec else resolve_problem_spec(
            df[job.target_col], job.target_col, "auto"
        )
        if not job.problem_spec:
            job.problem_spec = spec.to_dict()
        bundle = prepare_train_pipeline(df, job.target_col, key_cols, problem_spec=spec)
        task_type = spec.resolved_type.value

        # 智能采样（可选，仅二分类）：只降采样训练集的负样本，测试集保持原始分布
        X_train, y_train = bundle.X_train, bundle.y_train
        sampling_info: Optional[Dict[str, Any]] = None
        cv_folds = job.cv_folds
        if job.enable_sampling and spec.resolved_type == ProblemType.BINARY:
            X_train, y_train, sampling_info = _apply_negative_sampling(
                bundle.X_train, bundle.y_train, job.sampling_negative_ratio
            )
            _append_event(
                job,
                {
                    "event": "status",
                    "status": "running",
                    "message": (
                        f"智能采样：训练集 {sampling_info['original_rows']} → {sampling_info['sampled_rows']} 条"
                        f"（负样本保留 {sampling_info['negative_ratio']:.0%}）"
                        if sampling_info["applied"]
                        else "智能采样：负样本比例不缩减，使用完整训练集"
                    ),
                },
            )
            min_class_count = int(pd.Series(y_train).value_counts().min())
            if min_class_count < cv_folds:
                clamped = max(2, min_class_count)
                _append_event(
                    job,
                    {
                        "event": "status",
                        "status": "running",
                        "message": f"采样后最小类别仅剩 {min_class_count} 条，交叉验证折数从 {cv_folds} 调整为 {clamped}",
                    },
                )
                cv_folds = clamped

        cv = _cross_validator(task_type, cv_folds)
        direction = _optimization_direction(spec, job.optimization_metric)
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        trial_history: List[Dict[str, Any]] = []

        def objective(trial) -> float:
            if job.cancel_event.is_set():
                raise optuna.TrialPruned("用户取消任务")
            model_name = trial.suggest_categorical("model_type", candidate_models)
            params = _suggest_params(trial, model_name)
            model = ModelFactory.build(model_name, task_type=task_type, params=params)
            # 预处理器（和可选的特征选择器）进 CV：每一折重新 fit，杜绝折间泄漏
            pipeline = Pipeline(
                _training_steps(clone(bundle.preprocessor), task_type, model, job.enable_feature_selection)
            )
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=_scoring(job.optimization_metric, task_type),
                n_jobs=1,
                error_score="raise",
            )
            return _readable_cv_value(job.optimization_metric, scores)

        def callback(study_obj, trial) -> None:
            if job.cancel_event.is_set():
                study_obj.stop()
            if trial.value is None:
                return
            params = dict(trial.params)
            event = {
                "event": "trial",
                "status": "running",
                "trial_number": int(trial.number),
                "value": float(trial.value),
                "best_value": float(study_obj.best_value),
                "params": params,
                "model_type": params.get("model_type"),
                "progress": int(((trial.number + 1) / job.n_trials) * 100),
            }
            trial_history.append(event)
            with jobs_lock:
                job.current_trial = trial.number + 1
                job.progress = min(99, event["progress"])
            _append_event(job, event)

        # 平台期早停（移植自历史 AutoTrainer.check_early_stopping）：
        # 连续 early_stopping_rounds 轮改进幅度不足 min_improvement 即停止搜索
        early_stop = {"triggered": False, "best": None, "stale": 0}

        def early_stopping_callback(study_obj, trial) -> None:
            if job.early_stopping_rounds <= 0 or trial.value is None:
                return
            value = float(trial.value)
            best = early_stop["best"]
            if direction == "minimize":
                improved = best is None or value < best - job.min_improvement
            else:
                improved = best is None or value > best + job.min_improvement
            if improved:
                early_stop["best"] = value
                early_stop["stale"] = 0
                return
            early_stop["stale"] += 1
            if early_stop["stale"] >= job.early_stopping_rounds:
                early_stop["triggered"] = True
                study_obj.stop()

        study.optimize(
            objective,
            n_trials=job.n_trials,
            callbacks=[callback, early_stopping_callback],
            show_progress_bar=False,
        )

        if job.cancel_event.is_set():
            with jobs_lock:
                job.status = "cancelled"
                job.progress = min(job.progress, 99)
            _append_event(job, {"event": "done", "status": "cancelled", "message": "AutoML 任务已取消"})
            return

        if early_stop["triggered"]:
            _append_event(
                job,
                {
                    "event": "status",
                    "status": "running",
                    "message": (
                        f"触发早停：连续 {job.early_stopping_rounds} 轮改进小于 {job.min_improvement}，"
                        f"共完成 {len(study.trials)}/{job.n_trials} 次 Trial"
                    ),
                },
            )

        best_params = dict(study.best_params)
        best_model_name = best_params.pop("model_type", candidate_models[0])
        best_model = ModelFactory.build(best_model_name, task_type=task_type, params=with_runtime_params(best_model_name, best_params))
        calibration = None if spec.resolved_type == ProblemType.REGRESSION else {
            "threshold": 0.5,
            "source": "default",
            "calibration_size": 0,
        }
        if spec.resolved_type == ProblemType.BINARY:
            calibration = calibrate_binary_threshold(
                Pipeline(
                    _training_steps(
                        clone(bundle.preprocessor), task_type, clone(best_model), job.enable_feature_selection
                    )
                ),
                X_train,
                y_train,
                random_state=42,
            )
        preprocessor = clone(bundle.preprocessor).fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(bundle.X_test)
        transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]

        # 特征选择（可选）：在全部训练集上拟合选择器并随 artifact 持久化，
        # 推理侧 predict_with_artifact 会复用同一 selector，保证训练/推理一致
        feature_selector = None
        feature_selection_info: Optional[Dict[str, Any]] = None
        if job.enable_feature_selection:
            selector = _build_selector_step(task_type)
            selector.fit(X_train_transformed, y_train)
            support = selector.get_support()
            if int(support.sum()) > 0:
                feature_selector = selector
                X_train_transformed = selector.transform(X_train_transformed)
                X_test_transformed = selector.transform(X_test_transformed)
                transformed_names = [name for name, keep in zip(transformed_names, support) if keep]
                feature_selection_info = {
                    "enabled": True,
                    "applied": True,
                    "threshold": "median",
                    "original_feature_count": int(len(support)),
                    "selected_feature_count": int(support.sum()),
                }
            else:
                logger.warning("特征选择未保留任何特征，回退为使用全部特征")
                feature_selection_info = {
                    "enabled": True,
                    "applied": False,
                    "reason": "选择器未保留任何特征，已回退为全部特征",
                }

        best_model.fit(X_train_transformed, y_train)
        if spec.resolved_type == ProblemType.REGRESSION:
            metrics = compute_regression_metrics(best_model, X_test_transformed, bundle.y_test)
        else:
            metrics = compute_classification_metrics(
                best_model,
                X_test_transformed,
                bundle.y_test,
                decision_threshold=float(calibration["threshold"]) if calibration else 0.5,
            )
        feature_importance = feature_importance_rows(best_model, transformed_names)

        result = {
            "job_id": job_id,
            "status": "completed",
            "best_model": best_model_name,
            "best_params": {"model_type": best_model_name, **best_params},
            "best_value": float(study.best_value),
            "metrics": metrics,
            "trial_history": trial_history,
            "decision_summary": job.decision_summary,
            "problem_spec": spec.to_dict(),
            "feature_importance": feature_importance,
            "feature_columns": bundle.feature_columns,
            "threshold_calibration": calibration,
            "early_stopped": early_stop["triggered"],
            "feature_selection": feature_selection_info,
            "sampling": sampling_info,
            "_artifact": {
                "artifact_schema_version": 3,
                "problem_type": spec.resolved_type.value,
                "problem_spec": spec.to_dict(),
                "metric_directions": spec.metric_directions,
                "model": best_model,
                "preprocessor": preprocessor,
                "feature_selector": feature_selector,
                "label_encoder": bundle.label_encoder,
                "feature_columns": bundle.feature_columns,
                "decision_threshold": float(calibration["threshold"]) if calibration else None,
                "threshold_calibration": calibration,
                "decision_summary": job.decision_summary,
                "metrics": metrics,
            },
        }
        with jobs_lock:
            job.status = "completed"
            job.progress = 100
            job.current_trial = len(study.trials)
            job.result = result
        _append_event(job, {"event": "done", "status": "completed", "progress": 100, "message": "AutoML 全流程完成"})
    except Exception as exc:
        logger.exception("AutoML job failed: %s", exc)
        with jobs_lock:
            job.status = "failed"
            job.error = str(exc)
        _append_event(job, {"event": "error", "status": "failed", "message": str(exc)})


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_automl_job(
    file: UploadFile = File(...),
    target_col: Optional[str] = Form(None),
    n_trials: int = Form(10),
    cv_folds: int = Form(3),
    candidate_models: Optional[str] = Form(None),
    optimization_metric: str = Form("auto"),
    problem_type: str = Form("auto"),
    use_llm: bool = Form(True),
    business_context: str = Form("通用机器学习建模"),
    early_stopping_rounds: int = Form(0),
    min_improvement: float = Form(0.001),
    enable_feature_selection: bool = Form(False),
    enable_sampling: bool = Form(False),
    sampling_negative_ratio: float = Form(1.0),
):
    if not 1 <= n_trials <= 100:
        raise HTTPException(status_code=400, detail="n_trials 必须在 1 到 100 之间")
    if not 2 <= cv_folds <= 10:
        raise HTTPException(status_code=400, detail="cv_folds 必须在 2 到 10 之间")
    if not 0 <= early_stopping_rounds <= 50:
        raise HTTPException(status_code=400, detail="early_stopping_rounds 必须在 0 到 50 之间（0 表示关闭早停）")
    if not 0.0 <= min_improvement <= 0.5:
        raise HTTPException(status_code=400, detail="min_improvement 必须在 0 到 0.5 之间")
    if not 0.01 <= sampling_negative_ratio <= 1.0:
        raise HTTPException(status_code=400, detail="sampling_negative_ratio 必须在 0.01 到 1.0 之间")
    df = await read_uploaded_dataframe(file)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="数据至少需要 1 个特征列和 1 个目标列")

    target = detect_target_col(df, target_col)
    try:
        spec = resolve_problem_spec(df[target], target, problem_type)
        resolved_metric = resolve_metric(spec, optimization_metric)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc
    if enable_sampling and spec.resolved_type != ProblemType.BINARY:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_sampling_task",
                "field": "enable_sampling",
                "message": "智能采样当前仅支持二分类任务",
            },
        )
    key_cols = detect_key_columns(df, target)
    feature_df = df.drop(columns=[target, *key_cols], errors="ignore")
    requested_models = _parse_candidate_models(candidate_models)
    try:
        for model_name in requested_models or []:
            validate_algorithm(spec, model_name)
        valid_target_rows = int(df[target].notna().sum())
        test_size = 0.25 if valid_target_rows >= 20 else 0.3
        validate_train_test_split(spec, df[target], test_size)
        bundle = prepare_train_pipeline(df, target, key_cols, test_size=test_size, problem_spec=spec)
        validate_cv(spec, bundle.y_train, cv_folds)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc
    recommendation = _recommend_models(feature_df, df[target], requested_models, list(spec.algorithms))
    selected_models = recommendation["candidate_models"]
    summary = {
        **_eda_summary(df, target, key_cols),
        **recommendation,
        "optimization_metric": resolved_metric,
        "use_llm_requested": use_llm,
        # AutoML 任务始终使用确定性启发式推荐，LLM 不参与调参；如实标注避免误导
        "llm_status": "not_used" if use_llm else "disabled",
        "business_context": business_context,
        "early_stopping": {"rounds": early_stopping_rounds, "min_improvement": min_improvement},
        "feature_selection": enable_feature_selection,
        "sampling": {"enabled": enable_sampling, "negative_ratio": sampling_negative_ratio},
    }

    job_id = uuid.uuid4().hex
    now = _now()
    job = AutoMLJob(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        n_trials=n_trials,
        cv_folds=cv_folds,
        optimization_metric=resolved_metric,
        target_col=target,
        decision_summary=summary,
        problem_spec=spec.to_dict(),
        candidate_models=selected_models,
        early_stopping_rounds=early_stopping_rounds,
        min_improvement=min_improvement,
        enable_feature_selection=enable_feature_selection,
        enable_sampling=enable_sampling,
        sampling_negative_ratio=sampling_negative_ratio,
    )
    runtime_state.save_blob(INPUT_BLOB_NAMESPACE, job_id, df)
    _persist_job(job)
    with jobs_lock:
        automl_jobs[job_id] = job

    try:
        background_tasks.submit(f"automl-{job_id}", _run_job, job_id, None, selected_models)
    except TaskQueueFullError as exc:
        with jobs_lock:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = _now()
        _persist_job(job)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _job_snapshot(job)


@router.get("/jobs/{job_id}")
async def get_automl_job(job_id: str):
    return _job_snapshot(_safe_job(job_id))


@router.get("/jobs/{job_id}/stream")
async def stream_automl_job(job_id: str, cursor: int = Query(0, ge=0)):
    job = _safe_job(job_id)

    async def generator():
        index = cursor
        while True:
            with jobs_lock:
                events = list(job.events[index:])
                current_status = job.status
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            index += len(events)
            if current_status in TERMINAL_STATUSES and not events:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/cancel")
async def cancel_automl_job(job_id: str):
    job = _safe_job(job_id)
    if job.status in TERMINAL_STATUSES:
        return _job_snapshot(job)
    job.cancel_event.set()
    with jobs_lock:
        job.status = "cancelling"
        job.updated_at = _now()
    _append_event(job, {"event": "status", "status": "cancelling", "message": "正在取消 AutoML 任务"})
    return _job_snapshot(job)


@router.get("/jobs/{job_id}/result")
async def get_automl_result(job_id: str):
    job = _safe_job(job_id)
    if job.status != "completed" or job.result is None:
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态：{job.status}")
    return {key: value for key, value in job.result.items() if key != "_artifact"}


@router.post("/jobs/{job_id}/save")
async def save_automl_result(job_id: str):
    job = _safe_job(job_id)
    if job.status != "completed" or job.result is None:
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态：{job.status}")

    relative_path = save_job_artifact(ARTIFACT_DIR, "automl", job_id, job.result["_artifact"], PROJECT_ROOT)
    with jobs_lock:
        job.artifact_path = relative_path
        job.updated_at = _now()
    _persist_job(job)
    return {"status": "saved", "job_id": job_id, "artifact_path": relative_path}


@router.get("/jobs/{job_id:path}")
async def reject_invalid_job_path(job_id: str):
    validate_entity_id(job_id, "job_id")
    raise HTTPException(status_code=404, detail="未找到 AutoML 任务")
