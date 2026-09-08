"""Unified model registry API for saved training artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.automl.problem_spec import ProblemSpec, ProblemType

from ._common import validate_entity_id
from ._dataset import predict_with_artifact, resolve_artifact_problem_type

router = APIRouter(prefix="/api/models", tags=["Model Registry"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class ModelPredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


class ModelBatchPredictionRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list, max_length=1000)


def validate_model_id(model_id: str) -> str:
    return validate_entity_id(model_id, "model_id", MODEL_ID_PATTERN)


def _safe_relative(path: Path) -> str:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise HTTPException(status_code=400, detail="模型路径不在项目目录内")
    return str(resolved.relative_to(root))


def _try_load_registry_metadata(path: Path) -> Dict[str, Any]:
    """Load the one persisted artifact needed to render a registry record.

    Only schema-v3 artifacts expose their resolved problem semantics.  Older
    artifacts intentionally remain classification-compatible at prediction
    time, but are marked as legacy instead of assigning them new metadata.
    """
    metadata: Dict[str, Any] = {
        "metrics": {},
        "optimization_metric": None,
        "best_metric": None,
        "problem_type": None,
        "problem_spec": None,
        "legacy": True,
        "metadata_error": None,
    }
    try:
        artifact = joblib.load(path)
    except Exception:
        return metadata
    if not isinstance(artifact, dict):
        return metadata

    metrics = artifact.get("metrics") or {}
    if isinstance(metrics, dict):
        metadata["metrics"] = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
    else:
        best_value = artifact.get("best_value")
        if isinstance(best_value, (int, float)):
            metadata["metrics"] = {"best_value": float(best_value)}

    try:
        problem_type = resolve_artifact_problem_type(artifact)
    except HTTPException as exc:
        metadata["legacy"] = False
        metadata["metadata_error"] = exc.detail
        return metadata

    metadata["legacy"] = problem_type is None
    if problem_type is None:
        return metadata

    decision_summary = artifact.get("decision_summary")
    persisted_optimization_metric = artifact.get("optimization_metric")
    if not isinstance(persisted_optimization_metric, str) and isinstance(decision_summary, dict):
        persisted_optimization_metric = decision_summary.get("optimization_metric")
    if isinstance(persisted_optimization_metric, str) and persisted_optimization_metric.strip():
        metadata["optimization_metric"] = persisted_optimization_metric.strip()

    persisted_best_metric = artifact.get("best_metric")
    if not isinstance(persisted_best_metric, str):
        persisted_best_metric = artifact.get("metric")
    if isinstance(persisted_best_metric, str) and persisted_best_metric.strip():
        metadata["best_metric"] = persisted_best_metric.strip()

    problem_spec = artifact.get("problem_spec")
    if isinstance(problem_spec, ProblemSpec):
        metadata["problem_spec"] = problem_spec.to_dict()
    elif isinstance(problem_spec, dict):
        metadata["problem_spec"] = problem_spec
    metadata["problem_type"] = problem_type.value
    return metadata


def _record(path: Path, source: str, prefix: str) -> Dict[str, Any]:
    metadata = _try_load_registry_metadata(path)
    return {
        "model_id": f"{prefix}:{path.stem}",
        "source": source,
        "name": path.stem,
        "artifact_path": _safe_relative(path),
        **metadata,
        "updated_at": path.stat().st_mtime,
    }


def _iter_existing(patterns: Iterable[tuple[Path, str, str, str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for directory, glob_pattern, source, prefix in patterns:
        if not directory.exists():
            continue
        for path in sorted(directory.glob(glob_pattern)):
            if path.is_file():
                records.append(_record(path, source, prefix))
    return sorted(records, key=lambda item: item["updated_at"], reverse=True)


def list_model_records() -> List[Dict[str, Any]]:
    return _iter_existing(
        [
            (PROJECT_ROOT / "saved" / "automl", "*.joblib", "automl", "automl"),
            (PROJECT_ROOT / "saved" / "manual_training", "*.joblib", "manual_training", "manual"),
            (PROJECT_ROOT / "saved", "model_task_*.pkl", "agent_workflow", "workflow"),
            (PROJECT_ROOT / "model_code" / "saved", "model_*.joblib", "legacy", "legacy"),
            (PROJECT_ROOT / "model_code" / "saved", "*_model_*.joblib", "legacy", "legacy"),
        ]
    )


def _find_record(model_id: str) -> Dict[str, Any]:
    validate_model_id(model_id)
    for record in list_model_records():
        if record["model_id"] == model_id:
            return record
    raise HTTPException(status_code=404, detail="未找到模型")


def _load_artifact(record: Dict[str, Any]) -> Dict[str, Any]:
    path = (PROJECT_ROOT / record["artifact_path"]).resolve()
    root = PROJECT_ROOT.resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="模型路径不在项目目录内")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"模型产物读取失败：{exc}") from exc
    if not isinstance(artifact, dict):
        raise HTTPException(status_code=400, detail="暂不支持该 legacy 模型产物格式")
    return artifact


def _n_classes_of(artifact: Dict[str, Any]) -> int:
    """推断类别数：优先 label_encoder，其次模型 classes_，缺省按二分类。"""
    label_encoder = artifact.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        return int(len(label_encoder.classes_))
    classes = getattr(artifact.get("model"), "classes_", None)
    if classes is not None:
        return int(len(classes))
    return 2


def _summarize_predictions(
    rows: List[Dict[str, Any]], n_classes: int, problem_type: ProblemType | None
) -> Dict[str, float | int | None]:
    """汇总批量预测：positive_rate 仅二分类有定义（encoded==1 占比），多分类为 None。"""
    if not rows:
        return {"count": 0, "positive_rate": None, "average_probability": None}
    if problem_type == ProblemType.REGRESSION:
        return {"count": len(rows), "positive_rate": None, "average_probability": None}
    encoded = [float(row["encoded_prediction"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows if row.get("probability") is not None]
    positive_rate = None
    if n_classes == 2:
        positive_rate = sum(1 for value in encoded if value == 1.0) / len(encoded)
    return {
        "count": len(rows),
        "positive_rate": positive_rate,
        "average_probability": (sum(probabilities) / len(probabilities)) if probabilities else None,
    }


@router.get("")
async def list_models():
    return {"models": list_model_records()}


@router.post("/{model_id:path}/predict")
async def predict_with_model(model_id: str, payload: ModelPredictionRequest):
    record = _find_record(model_id)
    artifact = _load_artifact(record)
    result = predict_with_artifact(artifact, [payload.features])[0]
    return {
        "model_id": record["model_id"],
        "source": record["source"],
        "input_features": payload.features,
        **result,
    }


@router.post("/{model_id:path}/batch-predict")
async def batch_predict_with_model(model_id: str, payload: ModelBatchPredictionRequest):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="批量预测至少需要一行特征")
    record = _find_record(model_id)
    artifact = _load_artifact(record)
    problem_type = resolve_artifact_problem_type(artifact)
    predictions = predict_with_artifact(artifact, payload.rows)
    rows = [
        {
            "index": index,
            "features": features,
            **prediction,
        }
        for index, (features, prediction) in enumerate(zip(payload.rows, predictions))
    ]
    return {
        "model_id": record["model_id"],
        "source": record["source"],
        "summary": _summarize_predictions(rows, _n_classes_of(artifact), problem_type),
        "rows": rows,
    }


@router.get("/{model_id:path}")
async def get_model(model_id: str):
    return _find_record(model_id)
