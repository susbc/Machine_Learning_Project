"""
MLOps Scheduler - 定时监控任务调度

使用 APScheduler 实现:
1. 每日数据漂移巡检
2. 自动重训触发
3. 告警通知发送

调度配置:
- 每日凌晨 3:00 执行漂移检测
- PSI ≥ 0.2 自动触发重训
- 告警通过飞书/邮件发送

当前状态:
- 巡检使用真实数据：训练基线（保存/回填）vs 预测日志（每次预测落盘）或绑定的数据库表。
- 自动重训已接入训练链路：用可用标注数据重跑原算法，指标更优才保存新模型；
  数据/来源不可用时如实告警，不会误报成功。
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.api.models import list_model_records
from backend.services import mlops_monitoring as monitoring
from backend.services.auto_retrain import run_auto_retrain

logger = logging.getLogger("MLOps_Monitor")

# 全局调度器实例
scheduler = AsyncIOScheduler()

# 告警配置（PSI 阈值已移至 config/mlops_settings.json，由巡检统一读取）
ALERT_CONFIG = {
    "feishu_webhook": os.getenv("FEISHU_WEBHOOK", ""),
    "email_recipients": [
        addr.strip() for addr in os.getenv("ALERT_EMAIL", "").split(",") if addr.strip()
    ],
}


async def check_data_drift_job():
    """
    每日执行的漂移巡检任务

    流程:
    1. 遍历已保存模型，基线缺失时从留存训练数据回填
    2. 以预测日志（近 7 天真实预测输入）作为线上数据
    3. 计算各特征 PSI 并落盘巡检报告
    4. 触发告警 (PSI ≥ 0.1) 与自动重训调度 (PSI ≥ 0.2)
    """
    logger.info("🔍 开始执行 MLOps 数据漂移巡检任务...")

    monitoring.prune_prediction_log()
    records = list_model_records()
    if not records:
        logger.info("无已保存模型，跳过巡检")
        return

    results = []
    for record in records:
        monitoring.backfill_baseline(record)
        results.append(monitoring.inspect_model(record["model_id"], days=7, source="每日巡检"))

    completed = [item for item in results if item["status"] == "ok"]
    logger.info("巡检完成：%d/%d 个模型产出报告", len(completed), len(results))
    for item in results:
        if item["status"] != "ok":
            logger.info("巡检跳过 model=%s：%s", item["model_id"], item["message"])

    for item in completed:
        summary = item["report"]["summary"]
        if summary["needs_retrain"]:
            logger.critical(
                "🚨 严重漂移告警！model=%s 漂移特征数：%d/%d, 严重漂移：%d 个",
                item["model_id"],
                summary["drifted_features"],
                summary["total_features"],
                summary["critical_features"],
            )
            await send_alert(
                "🚨 MLOps 严重漂移告警",
                f"模型 {item['model_id']} 检测到 {summary['critical_features']} 个特征严重漂移 (PSI ≥ 0.2)，"
                f"共 {summary['drifted_features']} 个特征发生漂移。系统将在凌晨自动触发重训。",
                level="critical",
                drift_report=item["report"],
            )
            await schedule_auto_retrain(item["model_id"])
        elif summary["drifted_features"] > 0:
            logger.warning(
                "⚠️ 轻微漂移警告！model=%s 漂移特征数：%d",
                item["model_id"],
                summary["drifted_features"],
            )
            await send_alert(
                "⚠️ MLOps 漂移警告",
                f"模型 {item['model_id']} 检测到 {summary['drifted_features']} 个特征发生轻微漂移 (PSI ≥ 0.1)，"
                "请关注业务变化。",
                level="warning",
                drift_report=item["report"],
            )


async def send_alert(
    title: str,
    message: str,
    level: str = "info",
    drift_report: Optional[Dict] = None
):
    """
    发送告警通知
    
    支持:
    - 飞书 webhook
    - 邮件通知
    - 日志记录
    
    Args:
        title: 告警标题
        message: 告警内容
        level: 告警级别 (info/warning/error/critical)
        drift_report: 漂移报告详情 (可选)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建告警消息
    alert_message = f"""
【MLOps 监控告警】

时间：{timestamp}
级别：{level.upper()}
标题：{title}

内容：
{message}
"""
    
    if drift_report:
        summary = drift_report.get("summary", {})
        alert_message += f"""

漂移详情:
- 总特征数：{summary.get('total_features', 0)}
- 漂移特征：{summary.get('drifted_features', 0)}
- 严重漂移：{summary.get('critical_features', 0)}
- 需要重训：{'是' if summary.get('needs_retrain') else '否'}

Top 漂移特征:
"""
        features = drift_report.get("features", {})
        sorted_features = sorted(
            features.items(),
            key=lambda x: x[1].get("psi", 0),
            reverse=True
        )[:5]
        
        for feat, info in sorted_features:
            alert_message += f"- {feat}: PSI={info.get('psi', 0):.4f} ({info.get('severity', 'unknown')})\n"
    
    # 记录日志
    if level == "critical":
        logger.critical(alert_message)
    elif level == "error":
        logger.error(alert_message)
    elif level == "warning":
        logger.warning(alert_message)
    else:
        logger.info(alert_message)
    
    # 发送飞书通知
    feishu_webhook = ALERT_CONFIG.get("feishu_webhook")
    if feishu_webhook:
        try:
            import requests
            
            payload = {
                "msg_type": "text",
                "content": {
                    "text": alert_message
                }
            }

            # 同步 requests 在事件循环线程中直接调用会阻塞整个服务，
            # 移到工作线程执行
            response = await asyncio.to_thread(
                requests.post, feishu_webhook, json=payload, timeout=10
            )
            response.raise_for_status()
            logger.info("飞书告警已发送")
            
        except Exception as e:
            logger.error(f"飞书告警发送失败：{e}")
    
    # TODO: 邮件通知
    # email_recipients = ALERT_CONFIG.get("email_recipients")
    # if email_recipients and level in ["error", "critical"]:
    #     send_email_alert(email_recipients, title, alert_message)


async def schedule_auto_retrain(task_id: str):
    """
    调度自动重训任务
    
    在业务低峰期 (凌晨 4:00) 执行重训
    
    Args:
        task_id: 当前任务 ID (用于重新训练)
    """
    logger.info(f"🤖 已调度自动重训任务，将在凌晨 4:00 执行 (task_id: {task_id})")
    
    # 安排在凌晨 4:00 执行
    scheduler.add_job(
        auto_retrain_job,
        CronTrigger(hour=4, minute=0),
        args=[task_id],
        id=f"auto_retrain_{task_id}",
        replace_existing=True,
        max_instances=1
    )


async def auto_retrain_job(task_id: str):
    """
    自动重训任务

    流程:
    1. 选取可用标注数据（绑定数据库表优先，其次原训练数据集）
    2. 以原任务的算法与参数重跑训练链路
    3. 与旧模型指标对比，更优才保存新模型并写基线
    4. 任一环节不可用都如实告警，绝不伪造"重训完成"
    """
    logger.info("🤖 开始执行自动重训任务 (model: %s)", task_id)
    result = await asyncio.to_thread(run_auto_retrain, task_id)
    levels = {
        "retrained": "info",
        "not_improved": "info",
        "no_data": "warning",
        "unsupported": "warning",
        "model_missing": "warning",
        "retrain_failed": "error",
    }
    titles = {
        "retrained": "✅ MLOps 自动重训完成",
        "not_improved": "ℹ️ MLOps 自动重训：新模型未超越旧模型",
        "no_data": "⚠️ MLOps 自动重训未执行",
        "unsupported": "⚠️ MLOps 自动重训未执行",
        "model_missing": "⚠️ MLOps 自动重训未执行",
        "retrain_failed": "❌ MLOps 自动重训失败",
    }
    await send_alert(
        titles.get(result["status"], "MLOps 自动重训"),
        result["message"],
        level=levels.get(result["status"], "warning"),
    )


async def catchup_inspection_job():
    """启动补跑巡检：机器在 03:00 睡眠/停机导致当日巡检缺失时，按同样逻辑补一次。"""
    records = list_model_records()
    completed = 0
    for record in records:
        if monitoring.today_report_exists(record["model_id"]):
            continue
        monitoring.backfill_baseline(record)
        result = monitoring.inspect_model(record["model_id"], days=7, source="补跑巡检")
        if result["status"] == "ok":
            completed += 1
        else:
            logger.info("补跑巡检跳过 model=%s：%s", record["model_id"], result["message"])
    if completed:
        logger.info("补跑巡检完成：%d 个模型产出报告", completed)


def setup_scheduler():
    """
    配置调度器任务

    调用此函数添加所有定时任务
    """
    # 每日凌晨 3:00 执行漂移检测；机器短时睡眠唤醒后 1 小时内仍补执行
    scheduler.add_job(
        check_data_drift_job,
        CronTrigger(hour=3, minute=0),
        id="daily_drift_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    # 启动 10 秒后补跑当日缺失的巡检（覆盖整夜关机等超出宽限期的场景）
    scheduler.add_job(
        catchup_inspection_job,
        "date",
        run_date=datetime.now() + timedelta(seconds=10),
        id="startup_catchup_inspection",
        replace_existing=True,
        misfire_grace_time=600,
    )

    logger.info("MLOps 调度器配置完成")
    logger.info("  - 每日 03:00: 数据漂移巡检（misfire 宽限 1 小时 + 启动补跑）")
    logger.info("  - 自动重训：PSI ≥ 0.2 时触发")


def start_scheduler():
    """启动调度器（幂等：已在运行时不重复注册任务）"""
    if scheduler.running:
        logger.info("MLOps 调度器已在运行，跳过重复启动")
        return
    setup_scheduler()
    scheduler.start()
    logger.info("🚀 MLOps 后台监控任务已启动")


def stop_scheduler():
    """停止调度器（未启动时静默返回，避免关闭路径二次报错）"""
    if not scheduler.running:
        return
    scheduler.shutdown(wait=False)
    logger.info("🛑 MLOps 后台监控任务已关闭")


__all__ = [
    "scheduler",
    "setup_scheduler",
    "start_scheduler",
    "stop_scheduler",
]
