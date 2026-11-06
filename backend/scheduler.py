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
- fetch_online_data() 为占位实现（返回 None），接入真实数据源前，
  每日巡检会自动跳过，自动重训触发后以失败/跳过告警收尾，不会误报成功。
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from model_code.mlops_monitor import DataDriftMonitor
from core.paths import SAVED_DIR

logger = logging.getLogger("MLOps_Monitor")

# 全局调度器实例
scheduler = AsyncIOScheduler()

# 当前监控的任务 ID (可从配置文件或数据库读取)
ACTIVE_TASK_ID: Optional[str] = None

# 告警配置
ALERT_CONFIG = {
    "feishu_webhook": os.getenv("FEISHU_WEBHOOK", ""),
    "email_recipients": [
        addr.strip() for addr in os.getenv("ALERT_EMAIL", "").split(",") if addr.strip()
    ],
    "psi_threshold_low": 0.1,
    "psi_threshold_high": 0.2
}


async def check_data_drift_job():
    """
    每日执行的漂移巡检任务
    
    流程:
    1. 加载训练时的基线数据
    2. 从生产数据库拉取过去 24 小时的线上数据
    3. 计算各特征的 PSI
    4. 触发告警 (PSI ≥ 0.1)
    5. 自动重训决策 (PSI ≥ 0.2)
    """
    logger.info("🔍 开始执行 MLOps 数据漂移巡检任务...")
    
    # 获取当前监控的任务 ID
    task_id = ACTIVE_TASK_ID or os.getenv("ACTIVE_MODEL_VERSION")
    
    if not task_id:
        logger.warning("未配置 ACTIVE_TASK_ID，跳过巡检")
        return
    
    try:
        # 1. 加载训练时的基线数据
        baseline_path = SAVED_DIR / f"baseline_{task_id}.json"
        if not os.path.exists(baseline_path):
            logger.error(f"基线数据文件不存在：{baseline_path}")
            await send_alert(
                "MLOps 巡检错误",
                f"基线数据文件不存在：{baseline_path}",
                level="error"
            )
            return
        
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        
        logger.info(f"基线数据加载完成：{len(baseline_data)} 项")
        
        # 2. 从生产数据库拉取过去 24 小时的线上数据
        df_online = await fetch_online_data(hours=24)

        if df_online is None or len(df_online) == 0:
            logger.warning("未获取到线上数据，跳过巡检")
            return
        
        logger.info(f"线上数据获取完成：{len(df_online)} 条")
        
        # 3. 计算各特征的 PSI
        monitor = DataDriftMonitor()
        drift_report = monitor.generate_drift_report(
            baseline_data,
            df_online,
            threshold_low=ALERT_CONFIG["psi_threshold_low"],
            threshold_high=ALERT_CONFIG["psi_threshold_high"]
        )
        
        # 4. 保存巡检报告
        report_path = f"logs/drift_report_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs("logs", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(drift_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"巡检报告已保存：{report_path}")
        
        # 5. 分析结果并触发告警/重训
        summary = drift_report["summary"]
        
        if summary["needs_retrain"]:
            # 严重漂移，需要重训
            logger.critical(
                f"🚨 严重漂移告警！"
                f"漂移特征数：{summary['drifted_features']}/{summary['total_features']}, "
                f"严重漂移：{summary['critical_features']} 个"
            )
            
            # 发送告警
            await send_alert(
                "🚨 MLOps 严重漂移告警",
                f"检测到 {summary['critical_features']} 个特征严重漂移 (PSI ≥ 0.2)，"
                f"共 {summary['drifted_features']} 个特征发生漂移。"
                f"系统将在凌晨自动触发重训。",
                level="critical",
                drift_report=drift_report
            )
            
            # 触发自动重训 (在低峰期执行)
            await schedule_auto_retrain(task_id)
        
        elif summary["drifted_features"] > 0:
            # 轻微漂移，发送警告
            logger.warning(
                f"⚠️ 轻微漂移警告！"
                f"漂移特征数：{summary['drifted_features']}/{summary['total_features']}"
            )
            
            await send_alert(
                "⚠️ MLOps 漂移警告",
                f"检测到 {summary['drifted_features']} 个特征发生轻微漂移 (PSI ≥ 0.1)，"
                f"请关注业务变化。",
                level="warning",
                drift_report=drift_report
            )
        
        else:
            logger.info("✅ 巡检完成，未检测到显著漂移")
        
    except Exception as e:
        logger.error(f"巡检任务执行失败：{e}", exc_info=True)
        await send_alert(
            "MLOps 巡检错误",
            f"巡检任务执行失败：{str(e)}",
            level="error"
        )


async def fetch_online_data(hours: int = 24):
    """
    从生产数据库拉取线上数据

    TODO: 实现真实数据库查询
    示例:
        from model_code.database.db_connector import DatabaseConnector
        db = DatabaseConnector(config)
        db.connect()
        df = db.query(f"SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL {hours} HOUR")
        db.disconnect()
        return df

    当前返回 None，需接入数据源后启用。
    """
    return None


async def simulate_online_data():
    """
    模拟从生产数据库拉取线上数据 (仅供开发测试使用)

    生产环境请替换为 fetch_online_data()
    """
    import pandas as pd
    import numpy as np

    # 使用局部 RNG：np.random.seed 会改写进程级全局状态，污染其他流程
    rng = np.random.default_rng(42)
    n_samples = 1000

    df = pd.DataFrame({
        'age': rng.integers(20, 60, n_samples),
        'income': rng.uniform(3000, 50000, n_samples),
        'city_code': rng.choice(['北京', '上海', '广州', '深圳'], n_samples),
        'order_count': rng.integers(1, 100, n_samples),
        'is_fraud': rng.integers(0, 2, n_samples)
    })

    # 模拟一些漂移 (实际数据中会有真实漂移)
    if datetime.now().day % 5 == 0:
        df['income'] = df['income'] * 1.5  # 收入分布漂移

    return df


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
    1. 从数据库拉取最新数据
    2. 执行完整的 AutoML 训练流程
    3. 评估新模型效果
    4. 如果新模型更好，则替换旧模型
    
    Args:
        task_id: 当前任务 ID
    """
    logger.info(f"🤖 开始执行自动重训任务 (task_id: {task_id})")
    
    try:
        # 1. 拉取最新数据
        df_new = await fetch_online_data()

        if df_new is None or len(df_new) == 0:
            # 数据源未接入时如实告警收尾，绝不伪造"重训完成"
            logger.warning(
                "自动重训跳过：未获取到线上数据 (task_id: %s)，"
                "fetch_online_data 尚未接入数据源",
                task_id,
            )
            await send_alert(
                "⚠️ MLOps 自动重训跳过",
                f"任务 {task_id} 已触发重训，但线上数据源未接入"
                f"（fetch_online_data 返回空），本次未执行任何训练。",
                level="warning"
            )
            return

        # 2. 训练执行链路尚未实现。接入前以失败收尾，避免误报成功误导运维。
        #    接入点：复用训练/AutoML 任务接口（参考 backend/api/training.py），
        #    成功后再更新 ACTIVE_TASK_ID 并发送完成通知。
        raise NotImplementedError("自动重训训练链路尚未实现，请接入训练逻辑后启用")

    except Exception as e:
        logger.error(f"自动重训失败：{e}", exc_info=True)
        await send_alert(
            "❌ MLOps 自动重训失败",
            f"任务 {task_id} 的自动重训失败：{str(e)}",
            level="error"
        )


def setup_scheduler():
    """
    配置调度器任务
    
    调用此函数添加所有定时任务
    """
    # 每日凌晨 3:00 执行漂移检测
    scheduler.add_job(
        check_data_drift_job,
        CronTrigger(hour=3, minute=0),
        id="daily_drift_check",
        replace_existing=True,
        max_instances=1
    )
    
    logger.info("MLOps 调度器配置完成")
    logger.info("  - 每日 03:00: 数据漂移巡检")
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
    "ACTIVE_TASK_ID",
]
