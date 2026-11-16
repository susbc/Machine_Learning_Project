"""Overview dashboard aggregation API: composes real runtime data for the home page."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from core.paths import OUTPUT_DIR
from model_code.database.connection_manager import ConnectionManager
from model_code.llm.llm_connector import LLMManager

from backend.runtime_state import RuntimeStateStore, runtime_state

from .automl import JOB_NAMESPACE as AUTOML_JOB_NAMESPACE
from .models import list_model_records
from .training import DATASET_NAMESPACE, JOB_NAMESPACE as TRAINING_JOB_NAMESPACE
from .workflow import WORKFLOW_STATE_FILENAME

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/overview", tags=["Overview"])

_ACTIVE_STATUSES = {"queued", "running", "cancelling"}
_WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
_TYPE_LABELS = {
    "binary_classification": "分类",
    "multiclass_classification": "分类",
    "regression": "回归",
}
_WORKFLOW_SOURCE = "Agent 工作流"


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _job_rows(store: RuntimeStateStore) -> List[Dict[str, Any]]:
    """汇总手动训练 / AutoML / Agent 工作流三类任务的真实运行记录。"""
    datasets = {item.get("dataset_id"): item for item in store.list(DATASET_NAMESPACE)}
    rows: List[Dict[str, Any]] = []
    for payload in store.list(TRAINING_JOB_NAMESPACE):
        spec = payload.get("problem_spec") or {}
        dataset = datasets.get(payload.get("dataset_id")) or {}
        rows.append(
            {
                "source": "手动训练",
                "name": dataset.get("filename") or payload.get("dataset_id") or payload.get("job_id", ""),
                "type": _TYPE_LABELS.get(spec.get("resolved_type"), "其他"),
                "model": payload.get("algorithm") or "-",
                "status": payload.get("status") or "failed",
                "progress": payload.get("progress") or 0,
                "created_at": _parse_time(payload.get("created_at")),
                "updated_at": _parse_time(payload.get("updated_at")),
            }
        )
    for payload in store.list(AUTOML_JOB_NAMESPACE):
        spec = payload.get("problem_spec") or {}
        rows.append(
            {
                "source": "AutoML",
                "name": f"AutoML · {payload.get('target_col') or payload.get('job_id', '')}",
                "type": _TYPE_LABELS.get(spec.get("resolved_type"), "其他"),
                "model": payload.get("best_model") or "-",
                "status": payload.get("status") or "failed",
                "progress": payload.get("progress") or 0,
                "created_at": _parse_time(payload.get("created_at")),
                "updated_at": _parse_time(payload.get("updated_at")),
            }
        )
    if OUTPUT_DIR.is_dir():
        for task_dir in sorted(OUTPUT_DIR.iterdir()):
            state_path = task_dir / WORKFLOW_STATE_FILENAME
            if not task_dir.is_dir() or not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("工作流状态文件不可读：%s", state_path)
                continue
            if not isinstance(state, dict):
                continue
            spec = state.get("problem_spec") or {}
            modified = datetime.fromtimestamp(state_path.stat().st_mtime)
            status_value = state.get("status") or "failed"
            rows.append(
                {
                    "source": _WORKFLOW_SOURCE,
                    "name": task_dir.name,
                    "type": _TYPE_LABELS.get(spec.get("resolved_type"), "其他"),
                    "model": state.get("selected_algorithm") or "-",
                    "status": status_value,
                    "progress": 100 if status_value == "completed" else 0,
                    "created_at": modified,
                    "updated_at": modified,
                }
            )
    return rows


def build_summary(store: RuntimeStateStore) -> Dict[str, Any]:
    now = datetime.now()
    week_ago = now - timedelta(days=7)

    records = list_model_records()
    distribution = {"分类模型": 0, "回归模型": 0, "其他模型": 0}
    for record in records:
        problem_type = record.get("problem_type")
        if problem_type in {"binary_classification", "multiclass_classification"}:
            distribution["分类模型"] += 1
        elif problem_type == "regression":
            distribution["回归模型"] += 1
        else:
            distribution["其他模型"] += 1
    recent_models = 0
    for record in records:
        updated_at = record.get("updated_at")
        if isinstance(updated_at, (int, float)) and datetime.fromtimestamp(updated_at) >= week_ago:
            recent_models += 1

    rows = _job_rows(store)
    running = sum(1 for row in rows if row["status"] in _ACTIVE_STATUSES)
    queued = sum(1 for row in rows if row["status"] == "queued")
    completed = sum(1 for row in rows if row["status"] == "completed")

    trend: List[Dict[str, Any]] = []
    for offset in range(6, -1, -1):
        day = (now - timedelta(days=offset)).date()
        trend.append(
            {
                "label": _WEEKDAY_LABELS[day.weekday()],
                "submitted": sum(1 for row in rows if row["created_at"] is not None and row["created_at"].date() == day),
                "completed": sum(
                    1
                    for row in rows
                    if row["status"] == "completed" and row["updated_at"] is not None and row["updated_at"].date() == day
                ),
            }
        )

    recent_tasks = sorted(rows, key=lambda row: row["created_at"] or datetime.min, reverse=True)[:8]

    connection_manager = ConnectionManager(storage_path=os.getenv("DATABASE_CONNECTIONS_PATH"))
    db_types = sorted(
        {config.get("db_type") for config in connection_manager.connections.values() if config.get("db_type")}
    )

    llm_manager = LLMManager(storage_path=os.getenv("LLM_CONFIG_PATH"))
    provider_names = [
        config.get("name") or config.get("type") or provider_id
        for provider_id, config in llm_manager.configs.items()
    ]

    workflow_rows = [row for row in rows if row["source"] == _WORKFLOW_SOURCE]
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "models": {
            "total": len(records),
            "recent_7d": recent_models,
            "distribution": [{"name": name, "count": count} for name, count in distribution.items() if count > 0],
        },
        "training": {"running": running, "queued": queued, "completed": completed},
        "connections": {"total": len(connection_manager.connections), "db_types": db_types},
        "llm": {"total": len(llm_manager.configs), "names": provider_names},
        "workflow": {
            "total": len(workflow_rows),
            "running": sum(1 for row in workflow_rows if row["status"] in _ACTIVE_STATUSES),
        },
        "trend": trend,
        "recent_tasks": [
            {
                "name": row["name"],
                "source": row["source"],
                "type": row["type"],
                "model": row["model"],
                "status": row["status"],
                "submitted_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else None,
                "progress": row["progress"],
            }
            for row in recent_tasks
        ],
    }


@router.get("/summary")
def get_summary() -> Dict[str, Any]:
    return build_summary(runtime_state)
