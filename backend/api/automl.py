"""
AutoML API - FastAPI 后端接口

提供 Optuna 优化过程的 SSE 实时推送
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)

router = APIRouter()

# 线程池用于运行阻塞的 Optuna 优化任务
executor = ThreadPoolExecutor(max_workers=2)

# 全局存储运行中的优化任务
running_optimizations: Dict[str, Dict] = {}

STUDY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_study_id(study_id: str) -> str:
    if not STUDY_ID_PATTERN.fullmatch(study_id):
        raise HTTPException(status_code=400, detail="无效的 study_id")
    return study_id


class TrialCallback:
    """
    Optuna Trial 回调类

    在每次 Trial 结束时被调用，将结果推送到异步队列

    【修复】：在初始化时传入 event loop，避免在线程中调用 get_event_loop() 可能失败
    """

    def __init__(self, queue: asyncio.Queue, study_id: str, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.study_id = study_id
        self.loop = loop

    def __call__(self, study, trial):
        """
        Optuna 回调函数

        Args:
            study: Optuna Study 对象
            trial: Optuna Trial 对象
        """
        if trial.state.is_finished():
            payload = {
                "status": "RUNNING",
                "study_id": self.study_id,
                "trial_number": trial.number,
                "value": float(trial.value) if trial.value is not None else None,
                "best_value": float(study.best_value) if study.best_value is not None else None,
                "params": trial.params,
                "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None
            }

            # 将数据放入异步队列 (线程安全)
            try:
                asyncio.run_coroutine_threadsafe(
                    self.queue.put(payload),
                    self.loop
                )
            except Exception as e:
                logger.error(f"推送 Trial 数据失败：{e}")


async def event_generator(queue: asyncio.Queue, total_trials: int, study_id: str):
    """
    SSE 事件生成器

    持续从队列读取数据并推送给前端

    Args:
        queue: 异步队列
        total_trials: 总 Trial 次数
        study_id: 研究 ID

    【优化】：超时时间从 300 秒改为 60 秒，更快响应连接断开
    """
    try:
        while True:
            # 等待新数据
            data = await asyncio.wait_for(queue.get(), timeout=60)

            if data is None:
                # 结束信号
                yield f"data: {json.dumps({'status': 'DONE', 'study_id': study_id})}\n\n"
                break

            # 计算进度百分比
            progress = int((data['trial_number'] + 1) / total_trials * 100)
            data['progress'] = progress

            # SSE 标准格式
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    except asyncio.TimeoutError:
        # 超时，发送心跳 (60 秒无数据)
        yield f"data: {json.dumps({'status': 'HEARTBEAT', 'study_id': study_id})}\n\n"
    except Exception as e:
        logger.error(f"SSE 推送错误：{e}")
        yield f"data: {json.dumps({'status': 'ERROR', 'message': str(e)})}\n\n"


def run_optuna_optimization(
    objective_func,
    n_trials: int,
    queue: asyncio.Queue,
    study_id: str,
    loop: asyncio.AbstractEventLoop,
    X,
    y,
    config: Dict[str, Any]
):
    """
    在线程池中运行 Optuna 优化
    
    Args:
        objective_func: 目标函数
        n_trials: Trial 次数
        queue: 异步队列
        study_id: 研究 ID
        X: 特征矩阵
        y: 目标变量
        config: AutoML 配置
    """
    import optuna
    from core.automl import AutoMLOptimizer

    try:
        # 创建优化器
        optimizer = AutoMLOptimizer(X, y, config, cv_splits=5, n_jobs=-1)

        # 创建 Study
        direction = "maximize" if config.get("optimization_metric", "roc_auc") in ["roc_auc", "accuracy", "f1"] else "minimize"
        study = optuna.create_study(direction=direction, study_name=study_id)

        # 创建回调 (传入 loop)
        callback = TrialCallback(queue, study_id, loop)

        # 运行优化
        study.optimize(
            objective_func if objective_func else optimizer.objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False  # 禁用控制台进度条，使用前端推送
        )

        # 保存最佳结果
        running_optimizations[study_id]["best_params"] = study.best_params
        running_optimizations[study_id]["best_value"] = float(study.best_value) if study.best_value else None
        running_optimizations[study_id]["status"] = "COMPLETED"

        # 发送结束信号
        asyncio.run_coroutine_threadsafe(
            queue.put(None),
            loop
        )

    except Exception as e:
        logger.error(f"Optuna 优化失败：{e}")
        running_optimizations[study_id]["status"] = "FAILED"
        running_optimizations[study_id]["error"] = str(e)

        # 发送错误信号
        try:
            asyncio.run_coroutine_threadsafe(
                queue.put(None),
                loop
            )
        except Exception as notify_error:
            logger.warning(f"发送优化结束信号失败：{notify_error}")


@router.get("/automl/optimize/stream")
async def stream_optimization(
    study_id: Optional[str] = Query(None, min_length=1, max_length=64),
    n_trials: int = Query(30, ge=10, le=100),
    X_col: str = None,
    y_col: str = None
):
    """
    SSE 流式 Optuna 优化接口
    
    Args:
        study_id: 研究 ID (用于标识任务)
        n_trials: Trial 次数
        X_col: 特征列 (可选，用于从会话获取数据)
        y_col: 目标列 (可选)
        
    Returns:
        StreamingResponse: SSE 数据流
    """
    import uuid
    import pandas as pd
    
    # 生成唯一研究 ID
    if not study_id:
        study_id = str(uuid.uuid4())
    study_id = validate_study_id(study_id)
    
    # 创建异步队列
    queue_obj = asyncio.Queue()
    
    # 注册运行中的任务
    running_optimizations[study_id] = {
        "status": "STARTING",
        "n_trials": n_trials,
        "created_at": str(pd.Timestamp.now())
    }
    
    # TODO: 这里需要从数据库或会话获取实际数据
    # 下面是示例代码，实际使用时需要替换为真实数据加载逻辑
    try:
        # 示例：加载数据 (需要替换为实际数据源)
        # X, y = load_data_for_optimization(X_col, y_col)
        
        # 临时示例数据
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        # 默认配置
        config = {
            "task_type": "binary_classification",
            "optimization_metric": "roc_auc",
            "candidate_models": ["lightgbm", "catboost"],
            "n_trials": n_trials,
            "search_spaces": {
                "lightgbm": {
                    "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                    "num_leaves": {"type": "int", "low": 15, "high": 127}
                },
                "catboost": {
                    "depth": {"type": "int", "low": 4, "high": 10},
                    "l2_leaf_reg": {"type": "int", "low": 1, "high": 10}
                }
            }
        }

        # 在线程池中运行优化
        # 【修复】使用 get_running_loop() 替代 get_event_loop()
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            executor,
            run_optuna_optimization,
            None,  # objective_func (使用默认的)
            n_trials,
            queue_obj,
            study_id,
            loop,
            X,
            y,
            config
        )
        
        # 返回 SSE 流
        return StreamingResponse(
            event_generator(queue_obj, n_trials, study_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Nginx 禁用缓冲
            }
        )
        
    except Exception as e:
        logger.error(f"启动优化失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automl/optimize/{study_id}/status")
async def get_optimization_status(study_id: str):
    """
    获取优化任务状态
    
    Args:
        study_id: 研究 ID
        
    Returns:
        Dict: 任务状态
    """
    if study_id not in running_optimizations:
        raise HTTPException(status_code=404, detail="未找到该研究 ID")
    
    return running_optimizations[study_id]


@router.post("/automl/optimize/{study_id}/stop")
async def stop_optimization(study_id: str):
    """
    停止优化任务
    
    Args:
        study_id: 研究 ID
        
    Returns:
        Dict: 操作结果
    """
    if study_id not in running_optimizations:
        raise HTTPException(status_code=404, detail="未找到该研究 ID")
    
    # TODO: 实现停止逻辑 (需要 Optuna Study 对象引用)
    running_optimizations[study_id]["status"] = "STOPPED"
    
    return {"status": "success", "message": f"任务 {study_id} 已停止"}


@router.get("/automl/optimize/{study_id}/results")
async def get_optimization_results(study_id: str):
    """
    获取优化结果
    
    Args:
        study_id: 研究 ID
        
    Returns:
        Dict: 优化结果
    """
    if study_id not in running_optimizations:
        raise HTTPException(status_code=404, detail="未找到该研究 ID")
    
    result = running_optimizations[study_id]
    
    if result["status"] != "COMPLETED":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态：{result['status']}"
        )
    
    return {
        "study_id": study_id,
        "best_params": result.get("best_params"),
        "best_value": result.get("best_value"),
        "status": result["status"]
    }
