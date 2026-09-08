"""
Manual training API.

This module provides the first Vue-facing manual-training workflow:
dataset upload, inferred field config, EDA, single-model training jobs, result
recovery, and artifact save.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

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
    capabilities_payload,
    resolve_problem_spec,
    validate_algorithm,
    validate_train_test_split,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Manual Training"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "saved" / "manual_training"

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class TrainingDataset:
    dataset_id: str
    filename: str
    df: pd.DataFrame = field(repr=False)
    target_col: str
    fields: List[Dict[str, Any]]
    created_at: str
    problem_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    job_id: str
    dataset_id: str
    algorithm: str
    params: Dict[str, Any]
    status: str
    progress: int
    created_at: str
    updated_at: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    problem_spec: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.3
    random_state: int = 42
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


training_datasets: Dict[str, TrainingDataset] = {}
training_jobs: Dict[str, TrainingJob] = {}
training_lock = threading.RLock()
DATASET_NAMESPACE = "training_datasets"
DATASET_BLOB_NAMESPACE = "training_dataset_frames"
JOB_NAMESPACE = "training_jobs"
RESULT_BLOB_NAMESPACE = "training_job_results"


class TrainingJobRequest(BaseModel):
    dataset_id: str
    algorithm: str
    problem_type: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0, le=10000)


class EdaRequest(BaseModel):
    dataset_id: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _persist_dataset(dataset: TrainingDataset) -> None:
    runtime_state.save_blob(DATASET_BLOB_NAMESPACE, dataset.dataset_id, dataset.df)
    runtime_state.put(
        DATASET_NAMESPACE,
        dataset.dataset_id,
        {
            "dataset_id": dataset.dataset_id,
            "filename": dataset.filename,
            "target_col": dataset.target_col,
            "fields": dataset.fields,
            "created_at": dataset.created_at,
            "problem_spec": dataset.problem_spec,
        },
    )


def _restore_dataset(dataset_id: str) -> Optional[TrainingDataset]:
    payload = runtime_state.get(DATASET_NAMESPACE, dataset_id)
    if payload is None:
        return None
    try:
        frame = runtime_state.load_blob(DATASET_BLOB_NAMESPACE, dataset_id)
    except FileNotFoundError:
        logger.error("Persisted training dataset is incomplete: %s", dataset_id)
        return None
    restored_payload = dict(payload)
    if "problem_spec" not in restored_payload:
        try:
            restored_payload["problem_spec"] = resolve_problem_spec(
                frame[restored_payload["target_col"]], restored_payload["target_col"], "auto"
            ).to_dict()
        except (KeyError, ProblemSpecError):
            restored_payload["problem_spec"] = _legacy_classification_problem_spec()
    return TrainingDataset(df=frame, **restored_payload)


def _persist_job(job: TrainingJob) -> None:
    has_result = job.result is not None
    if has_result:
        runtime_state.save_blob(RESULT_BLOB_NAMESPACE, job.job_id, job.result)
    runtime_state.put(
        JOB_NAMESPACE,
        job.job_id,
        {
            "job_id": job.job_id,
            "dataset_id": job.dataset_id,
            "algorithm": job.algorithm,
            "params": job.params,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": job.error,
            "artifact_path": job.artifact_path,
            "test_size": job.test_size,
            "random_state": job.random_state,
            "problem_spec": job.problem_spec,
            "has_result": has_result,
        },
    )


def _restore_job(job_id: str) -> Optional[TrainingJob]:
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
            logger.error("Persisted training result is incomplete: %s", job_id)
    if "problem_spec" not in restored_payload:
        try:
            dataset = _get_dataset(restored_payload["dataset_id"])
        except HTTPException:
            restored_payload["problem_spec"] = _legacy_classification_problem_spec()
        else:
            restored_payload["problem_spec"] = dataset.problem_spec or _legacy_classification_problem_spec()
    if isinstance(result, dict) and "problem_spec" not in result:
        result["problem_spec"] = restored_payload["problem_spec"]
    return TrainingJob(result=result, **restored_payload)


def _legacy_classification_problem_spec() -> Dict[str, Any]:
    """Provide a safe snapshot for pre-v3 jobs that have no recoverable target."""
    spec = resolve_problem_spec(pd.Series([0, 1]), "legacy_target", "binary")
    payload = spec.to_dict()
    payload["source"] = "legacy_compatibility"
    return payload


def _resolve_problem_spec_or_422(target: pd.Series, target_col: str, problem_type: Any) -> ProblemSpec:
    try:
        return resolve_problem_spec(target, target_col, problem_type)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc


def recover_interrupted_training_jobs() -> int:
    """Mark process-local jobs interrupted by a prior server exit as failed."""
    recovered = 0
    for payload in runtime_state.list(JOB_NAMESPACE):
        if payload.get("status") not in {"queued", "running", "cancelling"}:
            continue
        payload["status"] = "failed"
        payload["error"] = "服务曾重启，原训练进程已中断，请重新提交任务"
        payload["updated_at"] = _now()
        runtime_state.put(JOB_NAMESPACE, payload["job_id"], payload)
        recovered += 1
    return recovered


def prune_training_runtime(retention_days: int) -> int:
    """Remove expired datasets and terminal job snapshots from local storage."""
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for payload in runtime_state.list(DATASET_NAMESPACE):
        try:
            expired = datetime.fromisoformat(payload["created_at"]) < cutoff
        except (KeyError, TypeError, ValueError):
            expired = False
        if expired:
            dataset_id = payload["dataset_id"]
            runtime_state.delete(DATASET_NAMESPACE, dataset_id, (DATASET_BLOB_NAMESPACE,))
            with training_lock:
                training_datasets.pop(dataset_id, None)
            removed += 1

    for payload in runtime_state.list(JOB_NAMESPACE):
        try:
            expired = datetime.fromisoformat(payload["updated_at"]) < cutoff
        except (KeyError, TypeError, ValueError):
            expired = False
        if payload.get("status") in TERMINAL_STATUSES and expired:
            job_id = payload["job_id"]
            runtime_state.delete(JOB_NAMESPACE, job_id, (RESULT_BLOB_NAMESPACE,))
            with training_lock:
                training_jobs.pop(job_id, None)
            removed += 1
    return removed


def _get_dataset(dataset_id: str) -> TrainingDataset:
    validate_entity_id(dataset_id, "dataset_id")
    with training_lock:
        dataset = training_datasets.get(dataset_id)
        if dataset is None:
            dataset = _restore_dataset(dataset_id)
            if dataset is not None:
                training_datasets[dataset_id] = dataset
    if dataset is None:
        raise HTTPException(status_code=404, detail="未找到训练数据集")
    return dataset


def _get_job(job_id: str) -> TrainingJob:
    validate_entity_id(job_id, "job_id")
    with training_lock:
        job = training_jobs.get(job_id)
        if job is None:
            job = _restore_job(job_id)
            if job is not None:
                training_jobs[job_id] = job
    if job is None:
        raise HTTPException(status_code=404, detail="未找到训练任务")
    return job


def _infer_fields(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    key_columns = set(detect_key_columns(df, target_col))
    for col in df.columns:
        if col == target_col:
            field_type = "label"
        elif str(col) in key_columns:
            field_type = "key"
        elif pd.api.types.is_numeric_dtype(df[col]):
            field_type = "num"
        else:
            field_type = "str"
        fields.append(
            {
                "field_name": str(col),
                "field_type": field_type,
                "dtype": str(df[col].dtype),
                "missing_rate": round(float(df[col].isna().mean()), 6),
                "unique_count": int(df[col].nunique(dropna=True)),
                "encoding": "one-hot" if field_type == "str" else None,
                "scaling": "standard" if field_type == "num" else None,
            }
        )
    return fields


def _dataset_snapshot(dataset: TrainingDataset) -> Dict[str, Any]:
    preview = dataset.df.head(20).replace([np.inf, -np.inf], np.nan).astype(object)
    preview = preview.where(pd.notna(preview), None)
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "rows": int(dataset.df.shape[0]),
        "columns": int(dataset.df.shape[1]),
        "target_col": dataset.target_col,
        "problem_spec": dataset.problem_spec,
        "created_at": dataset.created_at,
        "quality": dataframe_quality_report(dataset.df, dataset.target_col),
        "preview": preview.to_dict(orient="records"),
    }


def _job_snapshot(job: TrainingJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_id": job.dataset_id,
        "algorithm": job.algorithm,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "artifact_path": job.artifact_path,
        "problem_spec": job.problem_spec,
    }


def _eda_summary(dataset: TrainingDataset) -> Dict[str, Any]:
    df = dataset.df
    feature_df = df.drop(columns=[dataset.target_col], errors="ignore")
    numeric = feature_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    quality = dataframe_quality_report(df, dataset.target_col)
    return {
        "dataset_id": dataset.dataset_id,
        "summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_col": dataset.target_col,
            "missing_rate": round(float(df.isna().sum().sum()) / max(df.size, 1), 6),
            "numeric_features": int(len(numeric.columns)),
            "categorical_features": int(len(feature_df.select_dtypes(include=["object", "category"]).columns)),
            "target_unique": int(df[dataset.target_col].nunique(dropna=True)),
        },
        "columns": dataset.fields,
        "quality": quality,
        "correlations": numeric.corr(numeric_only=True).round(4).fillna(0).to_dict() if len(numeric.columns) else {},
    }


def _prepare_data(dataset: TrainingDataset, test_size: float, random_state: int, problem_spec: ProblemSpec):
    key_cols = [item["field_name"] for item in dataset.fields if item["field_type"] == "key"]
    return prepare_train_pipeline(
        dataset.df,
        dataset.target_col,
        key_cols,
        test_size=test_size,
        random_state=random_state,
        problem_spec=problem_spec,
    )


def _mark_cancelled(job: TrainingJob) -> bool:
    """阶段边界取消检查：已置位则转为 cancelled 并返回 True。"""
    if not job.cancel_event.is_set():
        return False
    with training_lock:
        job.status = "cancelled"
        job.progress = min(job.progress, 99)
        job.updated_at = _now()
    _persist_job(job)
    return True


def _job_log_context(job: TrainingJob) -> tuple[str, str, str, str]:
    """Return non-sensitive lifecycle log fields from the durable job snapshot."""
    saved_spec = job.problem_spec if isinstance(job.problem_spec, dict) else {}
    capabilities = saved_spec.get("capabilities") if isinstance(saved_spec.get("capabilities"), dict) else {}
    return (
        job.job_id,
        str(saved_spec.get("resolved_type", "unknown")),
        job.algorithm,
        str(capabilities.get("default_metric", "unknown")),
    )


def _run_training_job(job_id: str, test_size: float, random_state: int) -> None:
    from model_code.model_factory import ModelFactory
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    job = _get_job(job_id)
    try:
        dataset = _get_dataset(job.dataset_id)
        with training_lock:
            job.status = "running"
            job.progress = 20
            job.updated_at = _now()
        _persist_job(job)
        if _mark_cancelled(job):
            return

        spec = ProblemSpec.from_dict(job.problem_spec) if job.problem_spec else _resolve_problem_spec_or_422(
            dataset.df[dataset.target_col], dataset.target_col, "auto"
        )
        bundle = _prepare_data(dataset, test_size, random_state, spec)
        with training_lock:
            job.progress = 45
            job.updated_at = _now()
        _persist_job(job)
        if _mark_cancelled(job):
            return

        task_type = spec.resolved_type.value
        preprocessor = bundle.preprocessor
        model = ModelFactory.build(job.algorithm, task_type=task_type, params=with_runtime_params(job.algorithm, job.params))
        calibration = None if spec.resolved_type == ProblemType.REGRESSION else {
            "threshold": 0.5,
            "source": "default",
            "calibration_size": 0,
        }
        if spec.resolved_type == ProblemType.BINARY:
            calibration = calibrate_binary_threshold(
                Pipeline([("prep", clone(preprocessor)), ("model", clone(model))]),
                bundle.X_train,
                bundle.y_train,
                random_state=random_state,
            )
        X_train = preprocessor.fit_transform(bundle.X_train)
        X_test = preprocessor.transform(bundle.X_test)
        model.fit(X_train, bundle.y_train)
        with training_lock:
            job.progress = 75
            job.updated_at = _now()
        _persist_job(job)
        if _mark_cancelled(job):
            return

        if spec.resolved_type == ProblemType.REGRESSION:
            metrics = compute_regression_metrics(model, X_test, bundle.y_test)
            confusion_matrix = None
            threshold_calibration = None
        else:
            metrics = compute_classification_metrics(
                model,
                X_test,
                bundle.y_test,
                decision_threshold=float(calibration["threshold"]) if calibration else 0.5,
            )
            confusion_matrix = metrics["confusion_matrix"]
            threshold_calibration = calibration
        feature_importance = feature_importance_rows(model, list(preprocessor.get_feature_names_out()))
        display_metric = "r2" if spec.resolved_type == ProblemType.REGRESSION else (
            "roc_auc" if "roc_auc" in metrics else "accuracy"
        )
        result = {
            "job_id": job.job_id,
            "dataset_id": job.dataset_id,
            "algorithm": job.algorithm,
            "params": job.params,
            "problem_spec": spec.to_dict(),
            "metrics": {key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": confusion_matrix,
            "feature_importance": feature_importance,
            "train_size": int(len(bundle.X_train)),
            "test_size": int(len(bundle.X_test)),
            "feature_columns": bundle.feature_columns,
            "threshold_calibration": threshold_calibration,
            "_artifact": {
                "artifact_schema_version": 3,
                "problem_type": spec.resolved_type.value,
                "problem_spec": spec.to_dict(),
                "metric_directions": spec.metric_directions,
                "best_metric": display_metric,
                "model": model,
                "preprocessor": preprocessor,
                "label_encoder": bundle.label_encoder,
                "feature_columns": bundle.feature_columns,
                "decision_threshold": float(calibration["threshold"]) if calibration else None,
                "threshold_calibration": threshold_calibration,
                "field_config": dataset.fields,
                "metrics": metrics,
            },
        }
        with training_lock:
            job.result = result
            job.status = "completed"
            job.progress = 100
            job.updated_at = _now()
        _persist_job(job)
        logger.info(
            "Manual training job completed: job_id=%s problem_type=%s algorithm=%s metric=%s",
            *_job_log_context(job),
        )
    except Exception as exc:
        logger.error(
            "Manual training job failed: job_id=%s problem_type=%s algorithm=%s metric=%s error_type=%s",
            *_job_log_context(job),
            type(exc).__name__,
        )
        with training_lock:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = _now()
        _persist_job(job)


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
async def upload_training_dataset(
    file: UploadFile = File(...), target_col: Optional[str] = Form(None), problem_type: str = Form("auto")
):
    df = await read_uploaded_dataframe(file)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="数据至少需要 1 个特征列和 1 个目标列")
    target = detect_target_col(df, target_col)
    spec = _resolve_problem_spec_or_422(df[target], target, problem_type)
    dataset_id = uuid.uuid4().hex
    dataset = TrainingDataset(
        dataset_id=dataset_id,
        filename=file.filename or "dataset",
        df=df,
        target_col=target,
        fields=_infer_fields(df, target),
        created_at=_now(),
        problem_spec=spec.to_dict(),
    )
    _persist_dataset(dataset)
    with training_lock:
        training_datasets[dataset_id] = dataset
    return _dataset_snapshot(dataset)


@router.get("/capabilities")
async def get_training_capabilities():
    return capabilities_payload()


@router.get("/datasets/{dataset_id}")
async def get_training_dataset(dataset_id: str):
    return _dataset_snapshot(_get_dataset(dataset_id))


@router.get("/datasets/{dataset_id}/field-config")
async def get_field_config(dataset_id: str):
    dataset = _get_dataset(dataset_id)
    return {"dataset_id": dataset.dataset_id, "fields": dataset.fields}


@router.get("/datasets/{dataset_id:path}")
async def reject_invalid_dataset_path(dataset_id: str):
    validate_entity_id(dataset_id, "dataset_id")
    raise HTTPException(status_code=404, detail="未找到训练数据集")


@router.post("/eda")
async def run_training_eda(payload: EdaRequest):
    return _eda_summary(_get_dataset(payload.dataset_id))


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_training_job(payload: TrainingJobRequest):
    dataset = _get_dataset(payload.dataset_id)
    if payload.problem_type is None:
        try:
            spec = ProblemSpec.from_dict(dataset.problem_spec)
        except (KeyError, TypeError, ProblemSpecError):
            spec = _resolve_problem_spec_or_422(dataset.df[dataset.target_col], dataset.target_col, "auto")
    else:
        spec = _resolve_problem_spec_or_422(dataset.df[dataset.target_col], dataset.target_col, payload.problem_type)
    try:
        validate_algorithm(spec, payload.algorithm)
        validate_train_test_split(spec, dataset.df[dataset.target_col], payload.test_size)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc
    job_id = uuid.uuid4().hex
    now = _now()
    job = TrainingJob(
        job_id=job_id,
        dataset_id=dataset.dataset_id,
        algorithm=payload.algorithm,
        params=payload.params,
        status="queued",
        progress=0,
        created_at=now,
        updated_at=now,
        test_size=payload.test_size,
        random_state=payload.random_state,
        problem_spec=spec.to_dict(),
    )
    _persist_job(job)
    with training_lock:
        training_jobs[job_id] = job
    try:
        background_tasks.submit(f"training-{job_id}", _run_training_job, job_id, payload.test_size, payload.random_state)
    except TaskQueueFullError as exc:
        with training_lock:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = _now()
        _persist_job(job)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _job_snapshot(job)


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    return _job_snapshot(_get_job(job_id))


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    job = _get_job(job_id)
    with training_lock:
        if job.status in TERMINAL_STATUSES:
            return _job_snapshot(job)
        job.cancel_event.set()
        job.status = "cancelling"
        job.updated_at = _now()
    _persist_job(job)
    return _job_snapshot(job)


@router.get("/jobs/{job_id}/result")
async def get_training_result(job_id: str):
    job = _get_job(job_id)
    if job.status != "completed" or job.result is None:
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态：{job.status}")
    return {key: value for key, value in job.result.items() if key != "_artifact"}


@router.post("/jobs/{job_id}/save")
async def save_training_result(job_id: str):
    job = _get_job(job_id)
    if job.status != "completed" or job.result is None:
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态：{job.status}")
    relative_path = save_job_artifact(ARTIFACT_DIR, "manual_training", job_id, job.result["_artifact"], PROJECT_ROOT)
    with training_lock:
        job.artifact_path = relative_path
        job.updated_at = _now()
    _persist_job(job)
    return {"status": "saved", "job_id": job_id, "artifact_path": relative_path}


@router.get("/jobs/{job_id:path}")
async def reject_invalid_job_path(job_id: str):
    validate_entity_id(job_id, "job_id")
    raise HTTPException(status_code=404, detail="未找到训练任务")
