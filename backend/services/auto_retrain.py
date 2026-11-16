"""自动重训：漂移严重时用可用标注数据重跑训练链路，并按指标决定是否保存新模型。

数据优先级：绑定的数据库表（需含目标列且样本充足）＞ 原训练数据集。
仅支持手动训练 / AutoML 来源的模型；新模型指标优于旧模型才落盘保存，
否则如实回报对比结果，绝不伪造"重训完成"。
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from backend.api import training as training_api
from backend.api.models import list_model_records
from backend.runtime_state import background_tasks
from backend.services import mlops_monitoring as monitoring

logger = logging.getLogger(__name__)

AUTOML_RESULT_BLOB_NAMESPACE = "automl_job_results"
RETRAIN_TIMEOUT_SECONDS = 3600
MIN_RETRAIN_ROWS = 100


def _source_job(record: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    artifact = record.get("artifact_path")
    if not artifact:
        return None
    store = training_api.runtime_state
    for payload in store.list(monitoring.TRAINING_JOB_NAMESPACE):
        if payload.get("artifact_path") == artifact:
            return "manual", payload
    for payload in store.list(monitoring.AUTOML_JOB_NAMESPACE):
        if payload.get("artifact_path") == artifact:
            return "automl", payload
    return None


def _retrain_frame(record: Dict[str, Any], target_col: str) -> Optional[pd.DataFrame]:
    binding = monitoring.get_online_source(record["model_id"])
    if binding:
        try:
            frame = monitoring.online_frame_from_binding(binding, days=30)
        except Exception as exc:
            logger.warning("绑定巡检源读取失败 model=%s：%s", record["model_id"], exc)
            frame = pd.DataFrame()
        if target_col in frame.columns and len(frame) >= MIN_RETRAIN_ROWS:
            return frame
    source = monitoring._source_frame(record, training_api.runtime_state)
    if source is None:
        return None
    frame, stored_target, _ = source
    if not isinstance(frame, pd.DataFrame) or stored_target != target_col or target_col not in frame.columns:
        return None
    return frame if len(frame) >= MIN_RETRAIN_ROWS else None


def _metric_direction(spec: Dict[str, Any], metric_name: str) -> str:
    directions = (spec.get("capabilities") or {}).get("metric_directions") or {}
    return directions.get(metric_name, "maximize")


def _wait_for_job(job_id: str, timeout: int = RETRAIN_TIMEOUT_SECONDS) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = training_api._get_job(job_id)
        if job.status in training_api.TERMINAL_STATUSES:
            return job.status
        time.sleep(2)
    return "timeout"


def run_auto_retrain(model_id: str) -> Dict[str, Any]:
    record = next((item for item in list_model_records() if item["model_id"] == model_id), None)
    if record is None:
        return {"status": "model_missing", "message": f"未找到模型 {model_id}，自动重训未执行"}

    found = _source_job(record)
    if found is None:
        return {
            "status": "unsupported",
            "message": f"模型 {model_id} 的来源不支持自动重训（仅手动训练 / AutoML 来源可重训）",
        }
    kind, payload = found

    if kind == "manual":
        algorithm = payload.get("algorithm")
        params = payload.get("params") or {}
        test_size = float(payload.get("test_size", 0.3))
        random_state = int(payload.get("random_state", 42))
        spec_dict = payload.get("problem_spec") or record.get("problem_spec") or {}
        dataset_payload = training_api.runtime_state.get(
            monitoring.TRAINING_DATASET_NAMESPACE, payload.get("dataset_id", "")
        )
        target_col = (dataset_payload or {}).get("target_col") or payload.get("target_col") or ""
    else:
        try:
            result_blob = training_api.runtime_state.load_blob(AUTOML_RESULT_BLOB_NAMESPACE, payload.get("job_id", ""))
        except FileNotFoundError:
            result_blob = None
        if not isinstance(result_blob, dict):
            return {"status": "no_data", "message": "AutoML 结果已不存在，无法取得重训配置"}
        algorithm = result_blob.get("best_model")
        params = result_blob.get("best_params") or {}
        test_size, random_state = 0.3, 42
        spec_dict = payload.get("problem_spec") or record.get("problem_spec") or {}
        target_col = payload.get("target_col") or ""

    if not algorithm or not target_col:
        return {"status": "no_data", "message": "缺少算法或目标列信息，自动重训未执行"}

    frame = _retrain_frame(record, target_col)
    if frame is None:
        return {
            "status": "no_data",
            "message": "无可用标注重训数据：绑定表需含目标列且不少于 "
            f"{MIN_RETRAIN_ROWS} 行，或原训练数据集已不存在",
        }

    if not spec_dict:
        try:
            spec_dict = training_api._resolve_problem_spec_or_422(frame[target_col], target_col, "auto").to_dict()
        except Exception:
            return {"status": "no_data", "message": "无法解析任务类型，自动重训未执行"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    dataset = training_api.TrainingDataset(
        dataset_id=dataset_id,
        filename=f"retrain_{monitoring.model_slug(model_id)}_{stamp}.csv",
        df=frame,
        target_col=target_col,
        fields=training_api._infer_fields(frame, target_col),
        created_at=training_api._now(),
        problem_spec=spec_dict,
    )
    training_api._persist_dataset(dataset)
    with training_api.training_lock:
        training_api.training_datasets[dataset_id] = dataset
        job = training_api.TrainingJob(
            job_id=job_id,
            dataset_id=dataset_id,
            algorithm=algorithm,
            params=params,
            status="queued",
            progress=0,
            created_at=training_api._now(),
            updated_at=training_api._now(),
            test_size=test_size,
            random_state=random_state,
            problem_spec=spec_dict,
        )
        training_api.training_jobs[job_id] = job
    training_api._persist_job(job)
    background_tasks.submit(f"training-{job_id}", training_api._run_training_job, job_id, test_size, random_state)
    logger.info("自动重训任务已提交：model=%s job=%s algorithm=%s rows=%d", model_id, job_id, algorithm, len(frame))

    final_status = _wait_for_job(job_id)
    if final_status != "completed":
        return {
            "status": "retrain_failed",
            "message": f"重训任务结束状态 {final_status}：{training_api._get_job(job_id).error or '未产生结果'}",
        }

    job = training_api._get_job(job_id)
    metric_name = record.get("best_metric") or record.get("optimization_metric")
    if not metric_name:
        metric_name = (spec_dict.get("capabilities") or {}).get("default_metric")
    new_metrics = job.result.get("metrics") or {}
    new_value = new_metrics.get(metric_name)
    old_value = (record.get("metrics") or {}).get(metric_name)
    if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float)):
        return {
            "status": "retrain_failed",
            "message": f"缺少可比指标 {metric_name}，未保存新模型",
        }
    direction = _metric_direction(spec_dict, metric_name)
    improved = new_value > old_value if direction == "maximize" else new_value < old_value
    comparison = f"{metric_name} 旧 {old_value:.4f} → 新 {new_value:.4f}（{'更优' if improved else '未超越'}）"
    if not improved:
        return {"status": "not_improved", "message": f"新模型未超越旧模型，保留旧模型：{comparison}"}

    relative_path = training_api.save_job_artifact(
        training_api.ARTIFACT_DIR, "manual_training", job_id, job.result["_artifact"], training_api.PROJECT_ROOT
    )
    with training_api.training_lock:
        job.artifact_path = relative_path
        job.updated_at = training_api._now()
    training_api._persist_job(job)
    new_model_id = f"manual:{Path(relative_path).stem}"
    key_cols = [
        item.get("field_name")
        for item in dataset.fields
        if isinstance(item, dict) and item.get("field_type") == "key"
    ]
    monitoring.write_baseline_for_saved_model(new_model_id, frame, target_col, [c for c in key_cols if c])
    return {
        "status": "retrained",
        "new_model_id": new_model_id,
        "message": f"自动重训完成并保存新模型 {new_model_id}：{comparison}",
    }
