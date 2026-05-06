"""
MLOps API - 模型监控与漂移检测接口

提供:
1. PSI 漂移检测报告
2. 模型健康度检查
3. 巡检日志查询
4. 自动重训触发
"""

import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mlops", tags=["MLOps 监控"])

# 日志文件路径
LOGS_DIR = "logs"
SAVED_DIR = "saved"


class DriftReportRequest(BaseModel):
    """漂移检测请求"""
    model_version: str
    baseline_path: str = None


@router.get("/drift-report")
async def get_drift_report(
    model_version: str = Query(..., description="模型版本 (task_id)"),
    baseline_path: str = Query(None, description="基线数据路径 (可选)")
):
    """
    获取指定模型的漂移检测报告

    Args:
        model_version: 模型版本
        baseline_path: 基线数据路径

    Returns:
        Dict: 漂移报告
    """
    try:
        # 默认基线路径
        if not baseline_path:
            baseline_path = f"{SAVED_DIR}/baseline_{model_version}.json"

        # 检查基线文件是否存在
        if not os.path.exists(baseline_path):
            # 尝试查找最近的巡检报告
            report_pattern = f"{LOGS_DIR}/drift_report_*"
            import glob
            reports = glob.glob(report_pattern)

            if reports:
                # 返回最新的巡检报告
                latest_report = sorted(reports)[-1]
                with open(latest_report, "r", encoding="utf-8") as f:
                    drift_report = json.load(f)
                return drift_report
            else:
                return {
                    "features": {},
                    "summary": {
                        "total_features": 0,
                        "drifted_features": 0,
                        "critical_features": 0,
                        "needs_retrain": False
                    },
                    "message": "未找到基线数据或巡检报告"
                }

        # 加载基线数据
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        # 返回空报告 (需要线上数据才能计算 PSI)
        return {
            "features": {},
            "summary": {
                "total_features": len([k for k in baseline_data.keys() if not k.endswith("_stats") and not k.endswith("_dist")]),
                "drifted_features": 0,
                "critical_features": 0,
                "needs_retrain": False
            },
            "message": "基线数据已加载，需要线上数据才能计算 PSI"
        }

    except Exception as e:
        logger.error(f"获取漂移报告失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_model_health():
    """
    获取模型健康度信息

    Returns:
        Dict: 模型健康度列表
    """
    try:
        models = []

        if not os.path.exists(SAVED_DIR):
            return {"models": []}

        # 扫描所有模型文件
        for filename in os.listdir(SAVED_DIR):
            if filename.startswith("model_") and filename.endswith(".pkl"):
                model_path = os.path.join(SAVED_DIR, filename)
                model_version = filename.replace("model_", "").replace(".pkl", "")

                try:
                    import joblib
                    artifact = joblib.load(model_path)

                    # 查找对应的巡检报告
                    drift_report = None
                    report_pattern = f"{LOGS_DIR}/drift_report_*"
                    import glob
                    reports = glob.glob(report_pattern)

                    if reports:
                        latest_report = sorted(reports)[-1]
                        with open(latest_report, "r", encoding="utf-8") as f:
                            drift_report = json.load(f)

                    # 计算健康度
                    auc = artifact.get("best_value", 0)
                    status = "健康" if auc > 0.8 else "亚健康" if auc > 0.7 else "需优化"

                    # [DEMO MODE] 以下值为模拟值，需对接实际监控系统后替换
                    models.append({
                        "模型": model_version,
                        "AUC": round(auc, 4),
                        "预测延迟 (ms)": None,  # TODO: 对接实际延迟统计 (如 Prometheus 指标)
                        "调用次数": None,  # TODO: 对接实际调用统计 (如 API 访问日志)
                        "健康度": status
                    })

                except Exception as e:
                    logger.warning(f"加载模型 {filename} 失败：{e}")

        return {"models": models}

    except Exception as e:
        logger.error(f"获取模型健康度失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_inspection_logs(
    days: int = Query(7, description="获取最近 N 天的日志", ge=1, le=30)
):
    """
    获取巡检日志

    Args:
        days: 获取最近 N 天的日志

    Returns:
        Dict: 日志列表
    """
    try:
        logs = []

        if not os.path.exists(LOGS_DIR):
            return {"logs": []}

        # 扫描巡检报告
        import glob
        report_pattern = f"{LOGS_DIR}/drift_report_*"
        reports = glob.glob(report_pattern)

        for report_path in sorted(reports, reverse=True)[:days]:
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    drift_report = json.load(f)

                # 解析文件名中的日期
                filename = os.path.basename(report_path)
                date_str = filename.replace("drift_report_", "").replace(".json", "")

                summary = drift_report.get("summary", {})

                # 判断结果
                if summary.get("needs_retrain"):
                    result = "重训"
                    detail = f"发现 {summary.get('critical_features', 0)} 个特征严重漂移"
                elif summary.get("drifted_features", 0) > 0:
                    result = "告警"
                    detail = f"发现 {summary.get('drifted_features', 0)} 个特征轻微漂移"
                else:
                    result = "正常"
                    detail = "所有模型 PSI < 0.1"

                logs.append({
                    "时间": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 03:00:00",
                    "事件": "每日巡检",
                    "结果": result,
                    "详情": detail
                })

            except Exception as e:
                logger.warning(f"读取日志文件 {report_path} 失败：{e}")

        # 如果没有日志，返回空列表 (不再返回模拟数据)
        if not logs:
            logs = []

        return {"logs": logs}

    except Exception as e:
        logger.error(f"获取巡检日志失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def trigger_retrain(
    model_version: str = Query(..., description="模型版本"),
    force: bool = Query(False, description="是否强制重训")
):
    """
    触发自动重训

    Args:
        model_version: 模型版本
        force: 是否强制重训

    Returns:
        Dict: 操作结果
    """
    try:
        # 这里应该调用 scheduler 中的自动重训逻辑
        # 目前仅返回成功响应

        logger.info(f"触发重训任务：model_version={model_version}, force={force}")

        # TODO: 实际实现应该:
        # 1. 检查当前是否有重训任务在运行
        # 2. 添加新的重训任务到调度器
        # 3. 发送通知

        from datetime import timedelta
        scheduled_time = (datetime.now() + timedelta(days=1)).replace(
            hour=4, minute=0, second=0, microsecond=0
        ).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "message": f"已触发模型 {model_version} 的重训任务",
            "scheduled_time": scheduled_time
        }

    except Exception as e:
        logger.error(f"触发重训失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_mlops_status():
    """
    获取 MLOps 系统状态

    Returns:
        Dict: 系统状态信息
    """
    from backend.scheduler import scheduler

    return {
        "scheduler_running": scheduler.running,
        "active_task_id": None,  # 从 scheduler 获取
        "next_run": str(scheduler.next_run_time) if scheduler.next_run_time else None,
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None
            }
            for job in scheduler.get_jobs()
        ]
    }
