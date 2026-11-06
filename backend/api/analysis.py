"""
AI analysis API.

These endpoints expose the existing deterministic diagnostics and report
registry to the TypeScript frontend. LLM analysis can be layered on top later;
rule diagnostics remain available even when no provider is configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services.datasets import detect_target_col, read_uploaded_dataframe
from model_code.analysis.registry import ensure_within_project, list_recent_records, scrub_sensitive
from model_code.analysis.report_builder import build_composite_report, save_composite_report
from model_code.analysis.rule_engine import diagnose_dataset, diagnose_prediction, diagnose_training
from model_code.analysis.schemas import AnalysisResult

router = APIRouter(prefix="/api/analysis", tags=["AI Analysis"])


class TrainingAnalysisRequest(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    feature_importance: Optional[List[List[Any]]] = None


class PredictionAnalysisRequest(BaseModel):
    predictions: List[Any]
    threshold: float = 0.5
    reference_predictions: Optional[List[Any]] = None


class CompositeAnalysisRequest(BaseModel):
    title: str = "AI 智能分析综合诊断"
    results: List[Dict[str, Any]] = Field(default_factory=list)
    llm_sections: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    save_report: bool = False


def _safe_result(result: AnalysisResult) -> Dict[str, Any]:
    return scrub_sensitive(result.to_dict())


@router.post("/dataset")
async def analyze_dataset(file: UploadFile = File(...), target_col: Optional[str] = Form(None)):
    df = await read_uploaded_dataframe(file)
    target = detect_target_col(df, target_col)
    result = diagnose_dataset(df, target)
    return _safe_result(result)


@router.post("/training")
async def analyze_training(payload: TrainingAnalysisRequest):
    feature_importance = None
    if payload.feature_importance:
        feature_importance = [(str(item[0]), float(item[1])) for item in payload.feature_importance if len(item) >= 2]
    result = diagnose_training(payload.metrics, scrub_sensitive(payload.params), feature_importance)
    return _safe_result(result)


@router.post("/prediction")
async def analyze_prediction(payload: PredictionAnalysisRequest):
    result = diagnose_prediction(
        payload.predictions,
        threshold=payload.threshold,
        reference_predictions=payload.reference_predictions,
    )
    return _safe_result(result)


@router.post("/composite")
async def analyze_composite(payload: CompositeAnalysisRequest):
    rule_results = [AnalysisResult.from_dict(item) for item in payload.results]
    metadata = scrub_sensitive(payload.metadata)
    llm_sections = scrub_sensitive(payload.llm_sections)
    result = build_composite_report(
        payload.title,
        rule_results,
        llm_sections=llm_sections,
        metadata=metadata,
    )
    report_path = None
    if payload.save_report:
        report_path = save_composite_report(result, llm_sections=llm_sections)
    return {"result": _safe_result(result), "report_path": report_path}


@router.get("/reports")
async def list_analysis_reports(limit: int = 20):
    return {"reports": scrub_sensitive(list_recent_records(limit=limit))}


@router.get("/reports/{analysis_id}")
async def get_analysis_report(analysis_id: str):
    records = list_recent_records(limit=200)
    record = next((item for item in records if item.get("analysis_id") == analysis_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="未找到分析报告")
    report_path = record.get("report_path")
    markdown = ""
    if report_path:
        # ensure_within_project 已对越界路径抛出异常，无需重复校验
        path = ensure_within_project(Path(str(report_path)))
        if path.exists():
            markdown = path.read_text(encoding="utf-8")
    return {"record": scrub_sensitive(record), "markdown": markdown}
