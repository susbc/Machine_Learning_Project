"""
Agentic AutoML Workflow - FastAPI 业务路由

整合所有核心模块，提供完整的生产级 ML 工作流 API。

核心设计:
1. task_id 管理有状态工作流
2. 文件系统作为中间存储 (output/{task_id}/)
3. SSE 实时推送训练进度
4. 模型 + 执行器一起持久化

API 端点:
- POST /api/workflow/step1_retrieve_data - 自然语言转 SQL 并提取数据
- POST /api/workflow/step2_generate_features - 生成特征工程策略
- GET /api/workflow/step3_train_stream - 数据清洗 + AutoML 调参 + SSE 推送
- POST /api/workflow/predict - 线上推理预测
"""

import os
import json
import uuid
import joblib
import asyncio
import time
import random
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from core.llm.agent_orchestrator import AgentOrchestrator, create_orchestrator
from model_code.feature_executor import FeatureEngineeringExecutor
from core.automl.automl_optimizer import AutoMLOptimizer
from core.llm.agent_schemas import FeatureStrategy

logger = logging.getLogger(__name__)

# ==================== 路由配置 ====================

router = APIRouter(prefix="/api/workflow", tags=["Agentic AutoML Workflow"])

# 线程池：专门用来跑阻塞的 ML 训练任务
executor_pool = ThreadPoolExecutor(max_workers=4)

# 初始化 LLM 编排器 (从环境变量读取 API Key)
orchestrator = create_orchestrator(
    model_name="qwen-plus",
    max_retries=3
)

# ==================== 请求体定义 ====================

class DataRetrievalRequest(BaseModel):
    """步骤 1：数据提取请求"""
    user_query: str = Field(..., description="用户自然语言查询", example="查询所有司机订单数据")
    target_col: str = Field(..., description="目标变量列名", example="is_fraud")


class FeatureGenerationRequest(BaseModel):
    """步骤 2：特征策略生成请求"""
    task_id: str = Field(..., description="任务 ID (从步骤 1 获取)")
    eda_report_text: str = Field("", description="可选：用户补充的业务先验知识")


class TrainRequest(BaseModel):
    """步骤 3：训练请求"""
    task_id: str = Field(..., description="任务 ID")
    n_trials: int = Field(30, description="Optuna Trial 次数", ge=10, le=100)


class PredictRequest(BaseModel):
    """步骤 4：线上预测请求"""
    model_version: str = Field(..., description="模型版本 (task_id)", example="task_abc123")
    features: Dict[str, Any] = Field(..., description="特征字典", example={"age": 35, "income": 5000})


# ==================== 步骤 1: 数据提取 ====================

@router.post("/step1_retrieve_data")
async def retrieve_data(req: DataRetrievalRequest):
    """
    步骤 1：自然语言转 SQL 并提取数据
    
    1. 从 Neo4j 获取图谱 Schema
    2. DataRetrieval_Agent 生成 SQL
    3. 执行 SQL 拉取数据
    4. 统计学压缩计算 (eda_summary)
    5. 落盘保存中间结果
    
    Returns:
        task_id: 任务 ID (用于后续步骤)
        sql_executed: 执行的 SQL
        eda_summary: 数据概览
    """
    try:
        # 生成唯一任务 ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        logger.info(f"创建新任务：{task_id}")
        
        # 1. 模拟从 Neo4j 获取图谱 Schema (实际使用时替换为真实查询)
        graph_schema = """
        - 表名：orders (订单表)
          字段：order_id (string), driver_id (bigint), order_fee (decimal), is_fraud (int)
          关联关系：orders.driver_id = driver_info.driver_id
        
        - 表名：driver_info (司机信息表)
          字段：driver_id (bigint), age (int), city_code (string)
        """
        
        # 2. DataRetrieval_Agent 生成 SQL
        logger.info("调用 DataRetrieval_Agent 生成 SQL...")
        sql_result = orchestrator.run_data_retrieval_agent(graph_schema, req.user_query)
        
        if not sql_result.is_executable:
            return {
                "status": "error",
                "msg": "数据不足，无法生成 SQL",
                "thought": sql_result.thought_process
            }
        
        logger.info(f"SQL 生成完成：{sql_result.sql[:100]}...")
        
        # 3. 执行 SQL 拉取数据 (模拟 - 实际使用时替换为真实数据库查询)
        # from model_code.database import DatabaseConnector
        # db = DatabaseConnector(db_config)
        # db.connect()
        # df = db.query(sql_result.sql)
        # db.disconnect()
        
        # 模拟数据
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'order_id': range(n_samples),
            'driver_id': np.random.randint(1, 100, n_samples),
            'order_fee': np.random.uniform(10, 500, n_samples),
            'age': np.random.randint(20, 60, n_samples),
            'city_code': np.random.choice(['北京', '上海', '广州', '深圳'], n_samples),
            'is_fraud': np.random.randint(0, 2, n_samples)
        })
        
        logger.info(f"数据提取完成：{df.shape}")
        
        # 4. 统计学压缩计算 (生成 eda_summary)
        eda_summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "target": req.target_col,
            "features": {}
        }
        
        for col in df.columns:
            col_data = df[col]
            feature_stats = {
                "type": str(col_data.dtype),
                "null_rate": float(col_data.isnull().mean()),
                "unique_values": int(col_data.nunique())
            }
            
            if col_data.dtype in ['int64', 'float64']:
                feature_stats["mean"] = float(col_data.mean())
                feature_stats["std"] = float(col_data.std())
            
            eda_summary["features"][col] = feature_stats
        
        # 5. 落盘保存中间结果
        os.makedirs(f"output/{task_id}", exist_ok=True)
        
        # 保存原始数据
        df.to_parquet(f"output/{task_id}/raw_data.parquet", index=False)
        
        # 保存 eda_summary
        with open(f"output/{task_id}/eda_summary.json", "w", encoding='utf-8') as f:
            json.dump(eda_summary, f, indent=2, ensure_ascii=False)
        
        # 保存 SQL
        with open(f"output/{task_id}/query.sql", "w") as f:
            f.write(sql_result.sql)
        
        logger.info(f"中间结果已保存：output/{task_id}/")
        
        return {
            "status": "success",
            "task_id": task_id,
            "sql_executed": sql_result.sql,
            "thought_process": sql_result.thought_process,
            "eda_summary": eda_summary,
            "data_shape": list(df.shape)
        }
        
    except Exception as e:
        logger.error(f"步骤 1 失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 步骤 2: 特征策略生成 ====================

@router.post("/step2_generate_features")
async def generate_features(req: FeatureGenerationRequest):
    """
    步骤 2：生成特征工程策略
    
    1. 读取上一步的统计画像
    2. EDA_Agent + Feature_Agent 生成策略
    3. 落盘保存策略 JSON (前端可渲染成表格让用户确认)
    
    Returns:
        strategy: 特征处理策略
    """
    try:
        logger.info(f"任务 {req.task_id}: 生成特征策略...")
        
        # 1. 读取上一步的统计画像
        eda_summary_path = f"output/{req.task_id}/eda_summary.json"
        if not os.path.exists(eda_summary_path):
            raise HTTPException(status_code=404, detail=f"未找到任务 {req.task_id} 的 eda_summary")
        
        with open(eda_summary_path, "r", encoding='utf-8') as f:
            eda_summary_json = f.read()
        
        # 2. Feature_Agent 生成策略
        logger.info("调用 Feature_Agent 生成策略...")
        strategy_obj = orchestrator.run_feature_agent(
            eda_summary_json,
            req.eda_report_text
        )
        
        logger.info(f"策略生成完成：{len(strategy_obj.actions)} 个动作")
        
        # 3. 落盘保存策略
        with open(f"output/{req.task_id}/feature_strategy.json", "w", encoding='utf-8') as f:
            json.dump(strategy_obj.dict(), f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "task_id": req.task_id,
            "strategy": strategy_obj.dict(),
            "actions_count": len(strategy_obj.actions)
        }
        
    except Exception as e:
        logger.error(f"步骤 2 失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 步骤 3: 训练 + SSE 推送 ====================

@router.get("/step3_train_stream")
async def train_stream(
    task_id: str = Query(..., description="任务 ID"),
    n_trials: int = Query(30, description="Optuna Trial 次数", ge=10, le=100)
):
    """
    步骤 3：数据清洗 + AutoML 调参 + SSE 实时推送
    
    注意：SSE 必须是 GET 请求，参数通过 Query 传递
    
    流程:
    1. 加载前置数据
    2. FeatureEngineeringExecutor.fit_transform
    3. AutoMLOptimizer 贝叶斯寻优
    4. SSE 实时推送 Trial 进度
    5. 持久化打包 (模型 + 执行器)
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def background_training_task():
        """后台训练任务 (在线程池中运行)"""
        try:
            logger.info(f"任务 {task_id}: 开始训练...")
            
            # 1. 加载前置数据
            data_path = f"output/{task_id}/raw_data.parquet"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"未找到数据文件：{data_path}")
            
            df_raw = pd.read_parquet(data_path)
            logger.info(f"数据加载完成：{df_raw.shape}")
            
            # 加载特征策略
            strategy_path = f"output/{task_id}/feature_strategy.json"
            if not os.path.exists(strategy_path):
                raise FileNotFoundError(f"未找到策略文件：{strategy_path}")
            
            with open(strategy_path, "r", encoding='utf-8') as f:
                strategy_json = f.read()
            
            strategy_obj = FeatureStrategy.parse_raw(strategy_json)
            logger.info(f"策略加载完成：{len(strategy_obj.actions)} 个动作")
            
            # 2. 执行特征工程 (fit_transform)
            feature_executor = FeatureEngineeringExecutor(target_col="is_fraud")
            df_clean = feature_executor.fit_transform(df_raw, strategy_obj)
            logger.info(f"特征工程完成：{df_clean.shape}")
            
            # 准备训练数据
            X = df_clean.drop(columns=["is_fraud"])
            y = df_clean["is_fraud"]
            
            # 3. AutoML 配置 (mock - 实际使用时调用 AutoML_Agent)
            optuna_config = {
                "task_type": "binary_classification",
                "optimization_metric": "roc_auc",
                "candidate_models": ["lightgbm", "catboost"],
                "n_trials": n_trials,
                "search_spaces": {
                    "lightgbm": {
                        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                        "num_leaves": {"type": "int", "low": 15, "high": 127},
                        "n_estimators": {"type": "int", "low": 100, "high": 500}
                    },
                    "catboost": {
                        "depth": {"type": "int", "low": 4, "high": 10},
                        "l2_leaf_reg": {"type": "int", "low": 1, "high": 10},
                        "n_estimators": {"type": "int", "low": 100, "high": 500}
                    }
                }
            }
            
            # 4. SSE 回调函数
            def sse_callback(study, trial):
                """Optuna Trial 结束时的回调，推送 SSE"""
                payload = {
                    "status": "RUNNING",
                    "trial": trial.number,
                    "value": float(trial.value) if trial.value else None,
                    "best_value": float(study.best_value) if study.best_value else None,
                    "params": trial.params
                }
                asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
            
            # 5. 启动 Optuna 寻优
            optimizer = AutoMLOptimizer(X, y, optuna_config, cv_splits=5, n_jobs=-1)
            
            # 运行优化 (带回调)
            best_val = 0.5
            for i in range(n_trials):
                time.sleep(0.2)  # 模拟训练耗时
                val = 0.6 + random.random() * 0.3
                best_val = max(best_val, val)
                
                # 模拟 callback
                class MockStudy:
                    best_value = best_val
                class MockTrial:
                    number = i
                    value = val
                    params = {"learning_rate": 0.1, "num_leaves": 50}
                
                sse_callback(MockStudy(), MockTrial())
            
            # 6. 持久化打包 (模型 + 执行器)
            os.makedirs("saved", exist_ok=True)
            artifact_path = f"saved/model_{task_id}.pkl"
            
            joblib.dump({
                "feature_executor": feature_executor,
                "config": optuna_config,
                "best_value": best_val,
                "feature_columns": list(X.columns)
            }, artifact_path)
            
            logger.info(f"模型已保存：{artifact_path}")
            
        except Exception as e:
            logger.error(f"训练失败：{e}")
            asyncio.run_coroutine_threadsafe(
                queue.put({"status": "ERROR", "msg": str(e)}),
                loop
            )
        finally:
            # 发送结束信号
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
    
    # 将阻塞的训练任务扔进线程池
    loop.run_in_executor(executor_pool, background_training_task)
    
    # SSE 事件生成器
    async def event_generator():
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=300)
                
                if data is None:
                    yield f"data: {json.dumps({'status': 'DONE'})}\n\n"
                    break
                
                yield f"data: {json.dumps(data)}\n\n"
                
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'status': 'HEARTBEAT'})}\n\n"
            except Exception as e:
                logger.error(f"SSE 推送错误：{e}")
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==================== 步骤 4: 线上预测 ====================

@router.post("/predict")
async def predict_online(req: PredictRequest):
    """
    步骤 4：线上推理预测
    
    1. 加载持久化的神器包 (模型 + 执行器)
    2. 将前端 JSON 转为 DataFrame
    3. 严格使用训练时的状态进行特征清洗 (transform, NOT fit_transform)
    4. 预测并返回
    
    Returns:
        prediction: 预测结果
        probability: 概率值
    """
    try:
        # 1. 加载持久化的神器包
        artifact_path = f"saved/model_{req.model_version}.pkl"
        if not os.path.exists(artifact_path):
            raise HTTPException(
                status_code=404,
                detail=f"模型未找到：{req.model_version}，请先训练模型"
            )
        
        artifact = joblib.load(artifact_path)
        feature_executor = artifact["feature_executor"]
        model = artifact.get("model")  # mock 模式下为 None
        feature_columns = artifact.get("feature_columns", [])
        
        logger.info(f"加载模型：{req.model_version}")
        
        # 2. 将前端 JSON 转为 DataFrame
        df_infer = pd.DataFrame([req.features])
        logger.info(f"输入数据：{df_infer.shape}")
        
        # 3. 严格使用训练时的状态进行特征清洗 (transform, NOT fit_transform)
        if feature_executor.is_fitted():
            df_clean = feature_executor.transform(df_infer)
            logger.info(f"特征清洗完成：{df_clean.shape}")
        else:
            df_clean = df_infer
        
        # 4. 预测 (mock - 实际使用时调用 model.predict)
        # pred = model.predict(df_clean)
        # prob = model.predict_proba(df_clean)[0][1]
        
        pred = [0.85]  # 模拟预测结果
        prob = pred[0]
        
        return {
            "status": "success",
            "model_version": req.model_version,
            "prediction": float(pred[0]),
            "probability": float(prob),
            "input_features": req.features
        }
        
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 辅助接口 ====================

@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_dir = f"output/{task_id}"
    
    if not os.path.exists(task_dir):
        return {"status": "not_found", "task_id": task_id}
    
    files = os.listdir(task_dir)
    
    status = {
        "task_id": task_id,
        "status": "in_progress",
        "files": files
    }
    
    if "raw_data.parquet" in files:
        status["step1"] = "done"
    if "feature_strategy.json" in files:
        status["step2"] = "done"
    if f"saved/model_{task_id}.pkl" in os.listdir("saved") if os.path.exists("saved") else False:
        status["step3"] = "done"
        status["status"] = "completed"
    
    return status


@router.get("/models")
async def list_models():
    """列出所有已训练的模型"""
    if not os.path.exists("saved"):
        return {"models": []}
    
    models = []
    for f in os.listdir("saved"):
        if f.startswith("model_task_") and f.endswith(".pkl"):
            task_id = f.replace("model_", "").replace(".pkl", "")
            models.append({
                "task_id": task_id,
                "model_path": f"saved/{f}"
            })
    
    return {"models": models}
