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
    dataframe_quality_report,
    detect_key_columns,
    detect_target_col,
    feature_importance_rows,
    prepare_train_pipeline,
    read_uploaded_dataframe,
    save_job_artifact,
    with_runtime_params,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/automl", tags=["AutoML"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "saved" / "automl"
SUPPORTED_METRICS = {"roc_auc", "accuracy", "f1"}
SUPPORTED_MODELS = {
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "logistic_regression",
}
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
    candidate_models: List[str] = field(default_factory=list)
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
            "candidate_models": job.candidate_models,
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
    has_result = bool(payload.pop("has_result", False))
    result = None
    if has_result:
        try:
            result = runtime_state.load_blob(RESULT_BLOB_NAMESPACE, job_id)
        except FileNotFoundError:
            logger.error("Persisted AutoML result is incomplete: %s", job_id)
    return AutoMLJob(result=result, **payload)


def recover_interrupted_automl_jobs() -> int:
    """Mark process-local searches interrupted by a prior server exit as failed."""
    recovered = 0
    for payload in runtime_state.list(JOB_NAMESPACE):
        if payload.get("status") not in {"queued", "running", "cancelling"}:
            continue
        payload["status"] = "failed"
        payload["error"] = "服务曾重启，原 AutoML 进程已中断，请重新提交任务"
        payload["updated_at"] = _now()
        events = payload.setdefault("events", [])
        events.append(
            {
                "timestamp": payload["updated_at"],
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
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "artifact_path": job.artifact_path,
    }


def _append_event(job: AutoMLJob, event: Dict[str, Any]) -> None:
    event = {"timestamp": _now(), **event}
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
    invalid = [item for item in parsed if item not in SUPPORTED_MODELS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"不支持的候选模型：{', '.join(invalid)}")
    return parsed


def _recommend_models(feature_df: pd.DataFrame, target: pd.Series, requested: Optional[List[str]]) -> Dict[str, Any]:
    if requested:
        return {
            "candidate_models": requested,
            "recommendation_reason": "使用前端指定候选模型。",
            "recommendation_source": "user",
        }

    try:
        from model_code.auto_selector import AutoModelSelector

        recommendation = AutoModelSelector.recommend(feature_df, target)
        candidates = [item for item in recommendation["candidates"] if item in SUPPORTED_MODELS]
        return {
            "candidate_models": candidates[:3] or ["random_forest", "logistic_regression"],
            "recommendation_reason": recommendation.get("reason", "启发式模型推荐。"),
            "recommendation_source": "heuristic",
            "selector_analysis": recommendation.get("analysis", {}),
        }
    except Exception as exc:
        logger.warning("AutoML model recommendation failed: %s", exc)
        return {
            "candidate_models": ["random_forest", "logistic_regression"],
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
    return {}


def _scoring(metric: str, n_classes: int) -> str:
    if metric == "accuracy":
        return "accuracy"
    if metric == "f1":
        return "f1_weighted"
    return "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"


def _run_job(
    job_id: str,
    df: Optional[pd.DataFrame] = None,
    candidate_models: Optional[List[str]] = None,
) -> None:
    import optuna
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score
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
        bundle = prepare_train_pipeline(df, job.target_col, key_cols)
        n_classes = int(bundle.y_train.nunique())
        task_type = "binary" if n_classes == 2 else "multiclass"
        safe_cv = max(2, min(job.cv_folds, int(bundle.y_train.value_counts().min())))
        cv = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42)
        direction = "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        trial_history: List[Dict[str, Any]] = []

        def objective(trial) -> float:
            if job.cancel_event.is_set():
                raise optuna.TrialPruned("用户取消任务")
            model_name = trial.suggest_categorical("model_type", candidate_models)
            params = _suggest_params(trial, model_name)
            model = ModelFactory.build(model_name, task_type=task_type, params=params)
            # 预处理器进 CV：每一折重新 fit，杜绝折间泄漏
            pipeline = Pipeline([("prep", clone(bundle.preprocessor)), ("model", model)])
            scores = cross_val_score(
                pipeline,
                bundle.X_train,
                bundle.y_train,
                cv=cv,
                scoring=_scoring(job.optimization_metric, n_classes),
                n_jobs=1,
                error_score=0.0,
            )
            return float(np.mean(scores))

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

        study.optimize(objective, n_trials=job.n_trials, callbacks=[callback], show_progress_bar=False)

        if job.cancel_event.is_set():
            with jobs_lock:
                job.status = "cancelled"
                job.progress = min(job.progress, 99)
            _append_event(job, {"event": "done", "status": "cancelled", "message": "AutoML 任务已取消"})
            return

        best_params = dict(study.best_params)
        best_model_name = best_params.pop("model_type", candidate_models[0])
        best_model = ModelFactory.build(best_model_name, task_type=task_type, params=with_runtime_params(best_model_name, best_params))
        calibration = {"threshold": 0.5, "source": "default", "calibration_size": 0}
        if task_type == "binary":
            calibration = calibrate_binary_threshold(
                Pipeline([("prep", clone(bundle.preprocessor)), ("model", clone(best_model))]),
                bundle.X_train,
                bundle.y_train,
                random_state=42,
            )
        preprocessor = clone(bundle.preprocessor).fit(bundle.X_train)
        X_train_transformed = preprocessor.transform(bundle.X_train)
        X_test_transformed = preprocessor.transform(bundle.X_test)
        best_model.fit(X_train_transformed, bundle.y_train)
        metrics = compute_classification_metrics(
            best_model,
            X_test_transformed,
            bundle.y_test,
            decision_threshold=float(calibration["threshold"]),
        )
        feature_importance = feature_importance_rows(best_model, list(preprocessor.get_feature_names_out()))

        result = {
            "job_id": job_id,
            "status": "completed",
            "best_model": best_model_name,
            "best_params": {"model_type": best_model_name, **best_params},
            "best_value": float(study.best_value),
            "metrics": metrics,
            "trial_history": trial_history,
            "decision_summary": job.decision_summary,
            "feature_importance": feature_importance,
            "feature_columns": bundle.feature_columns,
            "threshold_calibration": calibration,
            "_artifact": {
                "artifact_version": 2,
                "model": best_model,
                "preprocessor": preprocessor,
                "label_encoder": bundle.label_encoder,
                "feature_columns": bundle.feature_columns,
                "decision_threshold": float(calibration["threshold"]),
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
    optimization_metric: str = Form("roc_auc"),
    use_llm: bool = Form(True),
    business_context: str = Form("通用机器学习建模"),
):
    if not 1 <= n_trials <= 100:
        raise HTTPException(status_code=400, detail="n_trials 必须在 1 到 100 之间")
    if not 2 <= cv_folds <= 10:
        raise HTTPException(status_code=400, detail="cv_folds 必须在 2 到 10 之间")
    if optimization_metric not in SUPPORTED_METRICS:
        raise HTTPException(status_code=400, detail=f"不支持的优化指标：{optimization_metric}")

    df = await read_uploaded_dataframe(file)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="数据至少需要 1 个特征列和 1 个目标列")

    target = detect_target_col(df, target_col)
    key_cols = detect_key_columns(df, target)
    feature_df = df.drop(columns=[target, *key_cols], errors="ignore")
    requested_models = _parse_candidate_models(candidate_models)
    recommendation = _recommend_models(feature_df, df[target], requested_models)
    selected_models = recommendation["candidate_models"]
    summary = {
        **_eda_summary(df, target, key_cols),
        **recommendation,
        "optimization_metric": optimization_metric,
        "use_llm_requested": use_llm,
        "llm_status": "not_configured_fallback" if use_llm else "disabled",
        "business_context": business_context,
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
        optimization_metric=optimization_metric,
        target_col=target,
        decision_summary=summary,
        candidate_models=selected_models,
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
