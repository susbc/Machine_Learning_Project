"""
MLOps API - 模型监控与漂移检测接口

数据全部来自真实运行数据：
1. PSI 漂移检测：训练基线（模型保存时写入 / 存量回填）vs 预测日志（每次预测落盘）
2. 模型健康度：模型 registry + 预测调用次数 + 最新漂移报告结论
3. 巡检日志：落盘的漂移巡检报告（每日 03:00 调度 + 手动巡检）
4. 手动巡检与重训触发
"""

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field, validator
from sqlalchemy import inspect as sqlalchemy_inspect

from backend.services import mlops_monitoring as monitoring

from .database import TABLE_NAME_PATTERN, get_connector
from .models import list_model_records

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mlops", tags=["MLOps 监控"])


class InspectRequest(BaseModel):
    model_id: Optional[str] = None
    days: int = Field(default=7, ge=1, le=30)


class OnlineSourcePayload(BaseModel):
    connection_id: str
    table_name: str
    time_column: Optional[str] = None
    limit: int = Field(default=5000, ge=1, le=50000)


class PsiThresholdPayload(BaseModel):
    psi_threshold_low: float = Field(gt=0, lt=100)
    psi_threshold_high: float = Field(gt=0, le=100)

    @validator("psi_threshold_high")
    def validate_order(cls, value: float, values: Dict[str, Any]) -> float:
        low = values.get("psi_threshold_low")
        if low is not None and value <= low:
            raise ValueError("严重阈值必须大于轻微阈值")
        return value


@router.get("/drift-report")
async def get_drift_report(
    model_version: str = Query(..., description="模型版本 (model_id)"),
    baseline_path: str = Query(None, description="基线数据路径 (可选)"),
):
    """返回指定模型最近一次巡检报告；无报告时给出可操作的提示。"""
    report = monitoring.latest_report(model_version)
    if report is not None:
        return report
    empty_summary = {
        "total_features": 0,
        "drifted_features": 0,
        "critical_features": 0,
        "needs_retrain": False,
    }
    if baseline_path or monitoring.baseline_path(model_version).is_file():
        return {
            "features": {},
            "summary": empty_summary,
            "message": "基线已存在但尚无巡检报告，请点击「立即巡检」或等待每日 03:00 巡检",
        }
    return {
        "features": {},
        "summary": empty_summary,
        "message": "未找到基线数据或巡检报告：重新保存模型可写入基线，随后执行巡检",
    }


@router.post("/inspect")
async def run_inspect(payload: InspectRequest) -> Dict[str, Any]:
    """手动触发漂移巡检：缺基线时先尝试从留存训练数据回填。"""
    records = list_model_records()
    if payload.model_id:
        records = [record for record in records if record["model_id"] == payload.model_id]
        if not records:
            raise HTTPException(status_code=404, detail="未找到模型")
    results = []
    for record in records:
        monitoring.backfill_baseline(record)
        results.append(monitoring.inspect_model(record["model_id"], payload.days, source="手动巡检"))
    return {
        "inspected": len(results),
        "reports": sum(1 for item in results if item["status"] == "ok"),
        "results": [
            {
                "model_id": item["model_id"],
                "status": item["status"],
                "message": item["message"],
                "summary": item.get("report", {}).get("summary"),
            }
            for item in results
        ],
    }


@router.get("/settings")
async def get_settings() -> Dict[str, Any]:
    low, high = monitoring.get_psi_thresholds()
    return {"psi_threshold_low": low, "psi_threshold_high": high}


@router.put("/settings")
async def put_settings(payload: PsiThresholdPayload) -> Dict[str, Any]:
    try:
        low, high = monitoring.set_psi_thresholds(payload.psi_threshold_low, payload.psi_threshold_high)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"psi_threshold_low": low, "psi_threshold_high": high}


@router.get("/online-sources")
async def list_online_sources() -> Dict[str, Any]:
    """模型 → 数据库巡检源绑定列表。"""
    return {"sources": monitoring.list_online_sources()}


@router.put("/online-sources/{model_id:path}")
async def bind_online_source(model_id: str, payload: OnlineSourcePayload) -> Dict[str, Any]:
    """把已保存连接的数据库表绑定为该模型巡检的线上数据源。"""
    if not TABLE_NAME_PATTERN.fullmatch(payload.table_name):
        raise HTTPException(status_code=400, detail="无效的表名：仅支持字母/数字/下划线，可带 schema 前缀")
    connector = get_connector(payload.connection_id)
    try:
        if connector.engine is None and not connector.connect():
            raise HTTPException(status_code=400, detail=f"数据库连接失败：{connector.last_error or '请检查连接配置'}")
        schema, _, name = payload.table_name.rpartition(".")
        if not sqlalchemy_inspect(connector.engine).has_table(name, schema=schema or None):
            raise HTTPException(status_code=404, detail="未找到数据表")
    finally:
        connector.disconnect()
    binding = monitoring.set_online_source(model_id, payload.dict(exclude_none=True))
    return {"model_id": model_id, "source": binding}


@router.delete("/online-sources/{model_id:path}", status_code=204)
async def unbind_online_source(model_id: str) -> Response:
    monitoring.clear_online_source(model_id)
    return Response(status_code=204)


@router.get("/health")
async def get_model_health():
    """模型健康度：registry 指标 + 真实预测调用次数 + 最新漂移结论。"""
    counts = monitoring.prediction_counts()
    models = []
    for record in list_model_records():
        model_id = record["model_id"]
        metrics = record.get("metrics") or {}
        metric_name = record.get("best_metric") or record.get("optimization_metric")
        value = metrics.get(metric_name) if isinstance(metric_name, str) else None
        if value is None and metrics:
            metric_name = metric_name or next(iter(metrics))
            value = metrics.get(metric_name)
        health, _ = monitoring.health_status(model_id)
        models.append(
            {
                "模型": model_id,
                "最佳指标": None if not isinstance(value, (int, float)) else round(float(value), 4),
                "指标名": metric_name,
                "健康度": health,
                "调用次数": counts.get(model_id, 0),
                "预测延迟 (ms)": None,
            }
        )
    return {"models": models}


@router.get("/logs")
async def get_inspection_logs(days: int = Query(7, description="获取最近 N 条巡检日志", ge=1, le=30)):
    logs = []
    for path in reversed(monitoring.report_paths()):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("巡检报告不可读：%s", path)
            continue
        summary = report.get("summary", {})
        if summary.get("needs_retrain"):
            result = "重训"
            detail = f"发现 {summary.get('critical_features', 0)} 个特征严重漂移"
        elif summary.get("drifted_features", 0) > 0:
            result = "告警"
            detail = f"发现 {summary.get('drifted_features', 0)} 个特征轻微漂移"
        else:
            result = "正常"
            detail = "所有特征 PSI < 0.1"
        logs.append(
            {
                "时间": report.get("timestamp", "")[:19].replace("T", " "),
                "事件": report.get("source", "每日巡检"),
                "结果": result,
                "详情": f"{report.get('model_id', path.stem)}：{detail}",
            }
        )
        if len(logs) >= days:
            break
    return {"logs": logs}


@router.post("/retrain")
async def trigger_retrain(
    model_version: str = Query(..., description="模型版本"),
    force: bool = Query(False, description="是否强制重训"),
):
    """触发自动重训：真实调度重训任务（稍后执行），结果以告警/日志回执。"""
    import re as _re
    from datetime import datetime as _datetime
    from datetime import timedelta as _timedelta

    from backend.scheduler import auto_retrain_job, scheduler

    job_id = "manual_retrain_" + _re.sub(r"[^A-Za-z0-9_-]", "_", model_version)[:64]
    scheduler.add_job(
        auto_retrain_job,
        "date",
        run_date=_datetime.now() + _timedelta(seconds=5),
        args=[model_version],
        id=job_id,
        replace_existing=True,
        misfire_grace_time=600,
    )
    logger.info("触发重训任务：model_version=%s, force=%s, job=%s", model_version, force, job_id)
    return {
        "status": "success",
        "message": f"已调度模型 {model_version} 的重训任务，稍后执行并以告警/巡检日志回执",
        "job_id": job_id,
    }


@router.get("/status")
async def get_mlops_status():
    from backend.scheduler import scheduler

    jobs = scheduler.get_jobs()
    next_runs = [job.next_run_time for job in jobs if job.next_run_time]
    next_run = min(next_runs) if next_runs else None

    return {
        "scheduler_running": scheduler.running,
        "active_task_id": None,
        "next_run": str(next_run) if next_run else None,
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            }
            for job in jobs
        ],
    }
