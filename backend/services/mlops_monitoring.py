"""MLOps 真实数据监控：预测日志、训练基线、漂移巡检与报告管理。"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.paths import ARTIFACT_ROOT, LOGS_DIR, SAVED_DIR
from model_code.mlops_monitor import DataDriftMonitor

from backend.runtime_state import RuntimeStateStore, runtime_state

logger = logging.getLogger(__name__)

# 与 backend/api/training.py、backend/api/automl.py 的命名空间常量保持一致。
# 这里刻意不导入 API 模块：API 层反向依赖本服务写基线，避免循环导入。
TRAINING_JOB_NAMESPACE = "training_jobs"
TRAINING_DATASET_NAMESPACE = "training_datasets"
TRAINING_DATASET_BLOB_NAMESPACE = "training_dataset_frames"
AUTOML_JOB_NAMESPACE = "automl_jobs"
AUTOML_INPUT_BLOB_NAMESPACE = "automl_job_inputs"

PREDICTION_LOG_PATH = LOGS_DIR / "prediction_log.jsonl"
# 巡检线上数据源绑定（模型 → 数据库表），使生产落库数据可直接参与 PSI 巡检。
ONLINE_SOURCES_PATH = ARTIFACT_ROOT / "config" / "mlops_online_sources.json"
MLOPS_SETTINGS_PATH = ARTIFACT_ROOT / "config" / "mlops_settings.json"
DEFAULT_PSI_LOW = 0.1
DEFAULT_PSI_HIGH = 0.2
# PSI 在小样本下会把抽样噪声误判为漂移：10 分箱 + 30 行时噪声 PSI 常超 0.2。
# 因此巡检门槛设为 100 行，并按样本量自适应分箱（n//50，2~10 箱）。
MIN_ONLINE_ROWS = 100
# 预测日志轮转：超过保留窗口的条目无巡检价值；文件超上限时保留最近一半行。
PREDICTION_LOG_MAX_BYTES = 20 * 1024 * 1024
PREDICTION_LOG_RETENTION_DAYS = 30
_log_lock = threading.Lock()
_SAFE_SLUG = re.compile(r"[^A-Za-z0-9_.-]")


def model_slug(model_id: str) -> str:
    return _SAFE_SLUG.sub("_", model_id)


def baseline_path(model_id: str) -> Path:
    return SAVED_DIR / f"baseline_{model_slug(model_id)}.json"


def report_paths(model_id: Optional[str] = None) -> List[Path]:
    if not LOGS_DIR.is_dir():
        return []
    prefix = f"drift_report_{model_slug(model_id)}_" if model_id else "drift_report_"
    return sorted(path for path in LOGS_DIR.glob(f"{prefix}*.json") if path.is_file())


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def append_predictions(model_id: str, feature_rows: Iterable[Dict[str, Any]]) -> None:
    """把线上预测的输入特征追加到预测日志，作为漂移巡检的真实线上数据源。"""
    rows = [row for row in feature_rows if isinstance(row, dict)]
    if not rows:
        return
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with _log_lock, PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
            for features in rows:
                handle.write(
                    json.dumps({"ts": stamp, "model_id": model_id, "features": features}, ensure_ascii=False) + "\n"
                )
        if PREDICTION_LOG_PATH.stat().st_size > PREDICTION_LOG_MAX_BYTES:
            prune_prediction_log()
    except OSError as exc:
        logger.warning("预测日志写入失败 model=%s：%s", model_id, exc)


def prune_prediction_log() -> int:
    """清理预测日志：丢弃超过保留窗口/损坏的行；文件仍超上限时保留最近一半行。"""
    if not PREDICTION_LOG_PATH.is_file():
        return 0
    with _log_lock:
        try:
            lines = PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0
        cutoff = datetime.now() - timedelta(days=PREDICTION_LOG_RETENTION_DAYS)
        kept: List[str] = []
        removed = 0
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                removed += 1
                continue
            stamp = _parse_time(entry.get("ts"))
            if stamp is not None and stamp < cutoff:
                removed += 1
                continue
            kept.append(line)
        oversized = PREDICTION_LOG_PATH.stat().st_size > PREDICTION_LOG_MAX_BYTES
        if removed == 0 and not oversized:
            return 0
        if oversized and removed == 0:
            dropped = len(kept) // 2
            kept = kept[dropped:]
            removed += dropped
        temporary = PREDICTION_LOG_PATH.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        os.replace(temporary, PREDICTION_LOG_PATH)
        logger.info("预测日志轮转：移除 %d 行，保留 %d 行", removed, len(kept))
        return removed


def prediction_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not PREDICTION_LOG_PATH.is_file():
        return counts
    with PREDICTION_LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            model_id = entry.get("model_id")
            if isinstance(model_id, str):
                counts[model_id] = counts.get(model_id, 0) + 1
    return counts


def online_frame(model_id: str, days: int) -> pd.DataFrame:
    """近 N 天该模型的线上数据：优先数据库绑定表，其次预测日志。"""
    binding = get_online_source(model_id)
    if binding is not None:
        return online_frame_from_binding(binding, days)
    return online_frame_from_log(model_id, days)


def online_frame_from_log(model_id: str, days: int) -> pd.DataFrame:
    """近 N 天该模型的预测输入特征表（线上数据）。"""
    if not PREDICTION_LOG_PATH.is_file():
        return pd.DataFrame()
    cutoff = datetime.now() - timedelta(days=days)
    records: List[Dict[str, Any]] = []
    with PREDICTION_LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("model_id") != model_id:
                continue
            stamp = _parse_time(entry.get("ts"))
            if stamp is None or stamp < cutoff:
                continue
            features = entry.get("features")
            if isinstance(features, dict):
                records.append(features)
    return pd.DataFrame(records)


def get_psi_thresholds() -> Tuple[float, float]:
    """读取 PSI 阈值配置（轻微/严重），损坏或缺失时回退默认 0.1/0.2。"""
    try:
        payload = json.loads(MLOPS_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_PSI_LOW, DEFAULT_PSI_HIGH
    if not isinstance(payload, dict):
        return DEFAULT_PSI_LOW, DEFAULT_PSI_HIGH
    low = payload.get("psi_threshold_low", DEFAULT_PSI_LOW)
    high = payload.get("psi_threshold_high", DEFAULT_PSI_HIGH)
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not 0 < low < high:
        return DEFAULT_PSI_LOW, DEFAULT_PSI_HIGH
    return float(low), float(high)


def set_psi_thresholds(low: float, high: float) -> Tuple[float, float]:
    if not 0 < low < high:
        raise ValueError("PSI 阈值需满足 0 < 轻微 < 严重")
    MLOPS_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLOPS_SETTINGS_PATH.write_text(
        json.dumps({"psi_threshold_low": low, "psi_threshold_high": high}, indent=2),
        encoding="utf-8",
    )
    return float(low), float(high)


def _load_online_sources() -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.loads(ONLINE_SOURCES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_online_sources(mapping: Dict[str, Dict[str, Any]]) -> None:
    ONLINE_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONLINE_SOURCES_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def get_online_source(model_id: str) -> Optional[Dict[str, Any]]:
    binding = _load_online_sources().get(model_id)
    return binding if isinstance(binding, dict) else None


def set_online_source(model_id: str, binding: Dict[str, Any]) -> Dict[str, Any]:
    mapping = _load_online_sources()
    mapping[model_id] = binding
    _save_online_sources(mapping)
    return binding


def clear_online_source(model_id: str) -> None:
    mapping = _load_online_sources()
    if mapping.pop(model_id, None) is not None:
        _save_online_sources(mapping)


def list_online_sources() -> Dict[str, Dict[str, Any]]:
    return _load_online_sources()


def online_frame_from_binding(binding: Dict[str, Any], days: int) -> pd.DataFrame:
    """从绑定的数据库表读取线上数据；配置 time_column 时按近 N 天过滤。"""
    from backend.api.database import get_connector

    connector = get_connector(binding["connection_id"])
    try:
        frame = connector.read_table(binding["table_name"], limit=int(binding.get("limit", 5000)))
    finally:
        connector.disconnect()
    time_column = binding.get("time_column")
    if time_column and time_column in frame.columns:
        parsed = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame[parsed >= datetime.now() - timedelta(days=days)]
    return frame


def today_report_exists(model_id: str) -> bool:
    stamp = datetime.now().strftime("%Y%m%d")
    return any(path.stem.endswith(stamp) for path in report_paths(model_id))


def write_baseline(model_id: str, df: pd.DataFrame, features: List[str]) -> Path:
    monitor = DataDriftMonitor()
    baseline = monitor.extract_baseline(df, features)
    path = baseline_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    monitor.save_baseline(baseline, str(path))
    return path


def _source_frame(record: Dict[str, Any], store: RuntimeStateStore) -> Optional[Tuple[pd.DataFrame, str, List[str]]]:
    """按 registry 记录的 artifact 路径回溯训练数据集（手动训练 / AutoML）。

    返回 (数据框, 目标列, 需排除的 key 列)：ID/键列取值随业务递增，
    参与 PSI 只会产生必然漂移的噪声告警。
    """
    artifact = record.get("artifact_path")
    if not artifact:
        return None
    for payload in store.list(TRAINING_JOB_NAMESPACE):
        if payload.get("artifact_path") != artifact:
            continue
        dataset_id = payload.get("dataset_id")
        dataset = store.get(TRAINING_DATASET_NAMESPACE, dataset_id) if dataset_id else None
        if not dataset:
            return None
        try:
            frame = store.load_blob(TRAINING_DATASET_BLOB_NAMESPACE, dataset_id)
        except FileNotFoundError:
            return None
        key_cols = [
            field.get("field_name")
            for field in dataset.get("fields") or []
            if isinstance(field, dict) and field.get("field_type") == "key"
        ]
        return frame, dataset.get("target_col", ""), [col for col in key_cols if isinstance(col, str)]
    for payload in store.list(AUTOML_JOB_NAMESPACE):
        if payload.get("artifact_path") != artifact:
            continue
        job_id = payload.get("job_id")
        try:
            frame = store.load_blob(AUTOML_INPUT_BLOB_NAMESPACE, job_id)
        except FileNotFoundError:
            return None
        return frame, payload.get("target_col", ""), []
    return None


def write_baseline_for_saved_model(
    model_id: str, frame: pd.DataFrame, target_col: str, exclude_columns: Iterable[str] = ()
) -> Optional[Path]:
    """模型保存成功后写入监控基线；特征列 = 数据列去除目标列与 key 列。"""
    if not isinstance(frame, pd.DataFrame):
        return None
    excluded = set(exclude_columns)
    features = [col for col in frame.columns if col != target_col and col not in excluded]
    if not features:
        return None
    try:
        return write_baseline(model_id, frame, features)
    except (OSError, ValueError) as exc:
        logger.warning("监控基线写入失败 model=%s：%s", model_id, exc)
        return None


def backfill_baseline(record: Dict[str, Any], store: RuntimeStateStore = runtime_state) -> Optional[Path]:
    """基线缺失时从仍保留的训练数据集回填，使存量模型可被监控。"""
    existing = baseline_path(record["model_id"])
    if existing.is_file():
        return existing
    source = _source_frame(record, store)
    if source is None:
        return None
    frame, target_col, exclude_columns = source
    if not isinstance(frame, pd.DataFrame):
        return None
    try:
        return write_baseline_for_saved_model(record["model_id"], frame, target_col, exclude_columns)
    except (OSError, ValueError) as exc:
        logger.warning("基线回填失败 model=%s：%s", record.get("model_id"), exc)
        return None


def inspect_model(model_id: str, days: int, source: str) -> Dict[str, Any]:
    """基线 vs 预测日志计算 PSI 并落盘报告；样本不足或无基线时如实返回。"""
    path = baseline_path(model_id)
    if not path.is_file():
        return {
            "model_id": model_id,
            "status": "no_baseline",
            "message": "缺少训练基线：请重新保存模型，或确认训练数据集仍可回溯",
        }
    try:
        frame = online_frame(model_id, days)
    except Exception as exc:
        logger.warning("巡检线上数据源读取失败 model=%s：%s", model_id, exc)
        return {
            "model_id": model_id,
            "status": "online_source_error",
            "message": f"绑定的数据库巡检源读取失败：{exc}",
        }
    if len(frame) < MIN_ONLINE_ROWS:
        return {
            "model_id": model_id,
            "status": "insufficient_online",
            "message": f"近 {days} 天线上预测样本不足（{len(frame)}/{MIN_ONLINE_ROWS}），跳过巡检",
        }
    monitor = DataDriftMonitor(bins=max(2, min(10, len(frame) // 50)))
    baseline = monitor.load_baseline(str(path))
    low, high = get_psi_thresholds()
    report = monitor.generate_drift_report(baseline, frame, threshold_low=low, threshold_high=high)
    if report["summary"]["total_features"] == 0:
        return {
            "model_id": model_id,
            "status": "feature_mismatch",
            "message": "预测输入字段与基线特征无交集，无法计算 PSI：请确认预测请求携带训练特征字段",
        }
    report["model_id"] = model_id
    report["source"] = source
    report["thresholds"] = {"low": low, "high": high}
    target = LOGS_DIR / f"drift_report_{model_slug(model_id)}_{datetime.now().strftime('%Y%m%d')}.json"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("漂移巡检完成 model=%s report=%s", model_id, target)
    return {"model_id": model_id, "status": "ok", "report": report, "path": str(target), "message": "巡检完成"}


def latest_report(model_id: str) -> Optional[Dict[str, Any]]:
    paths = report_paths(model_id)
    if not paths:
        return None
    try:
        return json.loads(paths[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def health_status(model_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    report = latest_report(model_id)
    if report is not None:
        summary = report.get("summary", {})
        if summary.get("needs_retrain"):
            return "需重训", report
        if summary.get("drifted_features"):
            return "漂移告警", report
        return "健康", report
    if baseline_path(model_id).is_file():
        return "待巡检", None
    return "未巡检", None
