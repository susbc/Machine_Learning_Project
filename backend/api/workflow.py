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
import re
import threading
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# 修复导入路径 - 确保使用正确的模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.llm.agent_orchestrator import create_orchestrator
from core.automl.automl_agent import AutoMLAgent, extract_data_features
from core.automl.problem_spec import (
    ProblemSpec,
    ProblemSpecError,
    ProblemType,
    resolve_metric,
    resolve_problem_spec,
    validate_cv,
    validate_train_test_split,
)
from model_code.feature_executor import FeatureEngineeringExecutor, FeatureEngineeringTransformer
from core.automl.automl_optimizer import AutoMLOptimizer, OptimizationCancelled
from core.llm.agent_schemas import FeatureAction, FeatureStrategy

from ._dataset import compute_classification_metrics, compute_regression_metrics, predict_with_artifact

logger = logging.getLogger(__name__)

# ==================== 路由配置 ====================

router = APIRouter(prefix="/api/workflow", tags=["Agentic AutoML Workflow"])

# 线程池：专门用来跑阻塞的 ML 训练任务
executor_pool = ThreadPoolExecutor(max_workers=4)

TASK_ID_PATTERN = re.compile(r"^task_[a-f0-9]{8}$")
# 基于项目根目录（而非 CWD），从任意目录启动服务都能定位同一份数据
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
SAVED_DIR = PROJECT_ROOT / "saved"


class StaleWorkflowRun(RuntimeError):
    """A superseded/cancelled run attempted to mutate shared task state."""


@dataclass
class WorkflowRunControl:
    run_id: str
    cancel_event: threading.Event


workflow_run_lock = threading.RLock()
active_workflow_runs: Dict[str, WorkflowRunControl] = {}


def _claim_workflow_run(task_id: str) -> WorkflowRunControl:
    with workflow_run_lock:
        if task_id in active_workflow_runs:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "workflow_run_active",
                    "field": "task_id",
                    "message": "该任务已有训练运行，请等待其结束",
                },
            )
        control = WorkflowRunControl(uuid.uuid4().hex, threading.Event())
        active_workflow_runs[task_id] = control
        return control


def _require_current_run(task_id: str, run_id: str, *, allow_cancelled: bool = False) -> WorkflowRunControl:
    with workflow_run_lock:
        control = active_workflow_runs.get(task_id)
        if control is None or control.run_id != run_id:
            raise StaleWorkflowRun("工作流运行已失效")
        if control.cancel_event.is_set() and not allow_cancelled:
            raise OptimizationCancelled("工作流运行已取消")
        return control


def _write_run_state(
    task_id: str, run_id: str, payload: Dict[str, Any], *, allow_cancelled: bool = False
) -> None:
    with workflow_run_lock:
        _require_current_run(task_id, run_id, allow_cancelled=allow_cancelled)
        _write_workflow_state(task_id, {"run_id": run_id, **payload})


def _persist_run_artifact(task_id: str, run_id: str, artifact: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{run_id}.tmp")
    try:
        joblib.dump(artifact, temporary)
        with workflow_run_lock:
            _require_current_run(task_id, run_id)
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _release_workflow_run(task_id: str, run_id: str) -> None:
    with workflow_run_lock:
        control = active_workflow_runs.get(task_id)
        if control is not None and control.run_id == run_id:
            active_workflow_runs.pop(task_id, None)


def validate_task_id(task_id: str) -> str:
    """限制任务 ID 格式，避免路径穿越和任意文件访问。"""
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise HTTPException(status_code=400, detail="无效的 task_id")
    return task_id


def _safe_child_path(base_dir: Path, *parts: str) -> Path:
    base = base_dir.resolve()
    path = base.joinpath(*parts).resolve()
    if path != base and base not in path.parents:
        raise HTTPException(status_code=400, detail="无效的文件路径")
    return path


def get_task_dir(task_id: str) -> Path:
    return _safe_child_path(OUTPUT_DIR, validate_task_id(task_id))


def get_saved_path(filename: str) -> Path:
    return _safe_child_path(SAVED_DIR, filename)


WORKFLOW_STATE_FILENAME = "workflow_state.json"


def _write_workflow_state(task_id: str, payload: Dict[str, Any]) -> None:
    """Atomically persist one task's restart-safe workflow status."""
    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    state_path = _safe_child_path(task_dir, WORKFLOW_STATE_FILENAME)
    temporary = _safe_child_path(
        task_dir, f".workflow_state.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    state = {"task_id": validate_task_id(task_id), **payload}
    serialized = json.dumps(
        state, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_workflow_state(task_id: str) -> Optional[Dict[str, Any]]:
    """Load persisted task state without relying on process-local runtime data."""
    state_path = _safe_child_path(get_task_dir(task_id), WORKFLOW_STATE_FILENAME)
    if not state_path.is_file():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("读取工作流状态失败 task_id=%s error=%s", task_id, exc)
        return None
    return state if isinstance(state, dict) else None

# 初始化 LLM 编排器 (从环境变量读取 API Key)
orchestrator = create_orchestrator(
    model_name="qwen-plus",
    max_retries=3
)


def recommend_automl_config(data_features: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Return an Agent recommendation, with an offline-safe deterministic fallback."""
    try:
        return orchestrator.run_automl_agent(
            json.dumps(data_features, ensure_ascii=False), scenario
        ).model_dump()
    except Exception:
        logger.exception("AutoML Agent 调用失败，使用任务感知规则配置")
        return AutoMLAgent().generate_config(data_features, scenario)


def _coordinate_automl_config(
    suggested: Dict[str, Any], spec: ProblemSpec, n_trials: int
) -> tuple[Dict[str, Any], Optional[str], list[Dict[str, Any]]]:
    """Reconcile advisory LLM output with the canonical resolved problem spec."""
    resolved_type = spec.resolved_type.value
    suggested_type = suggested.get("task_type")
    corrections: list[Dict[str, Any]] = []

    if suggested_type != resolved_type:
        corrections.append(
            {
                "field": "task_type",
                "suggested": suggested_type,
                "resolved": resolved_type,
                "reason": "Agent 建议不能覆盖规范任务类型",
            }
        )

    suggested_metric = suggested.get("optimization_metric")
    try:
        metric = resolve_metric(spec, suggested_metric)
    except Exception:
        metric = resolve_metric(spec, "auto")
        corrections.append(
            {
                "field": "optimization_metric",
                "suggested": suggested_metric,
                "resolved": metric,
                "reason": "指标与规范任务类型不兼容",
            }
        )

    raw_candidates = suggested.get("candidate_models")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    candidates = []
    for model_name in raw_candidates:
        if model_name in spec.algorithms:
            if model_name not in candidates:
                candidates.append(model_name)
        else:
            corrections.append(
                {
                    "field": "candidate_models",
                    "suggested": model_name,
                    "resolved": None,
                    "reason": "模型与规范任务类型不兼容，已移除",
                }
            )

    if not candidates:
        baseline = "linear_regression" if spec.resolved_type == ProblemType.REGRESSION else "random_forest"
        candidates = [baseline]
        corrections.append(
            {
                "field": "candidate_models",
                "suggested": raw_candidates,
                "resolved": candidates,
                "reason": "没有兼容候选模型，使用确定性基线",
            }
        )

    suggested_spaces = suggested.get("search_spaces")
    if not isinstance(suggested_spaces, dict):
        suggested_spaces = {}
    if suggested_type != resolved_type:
        search_spaces = {model_name: {} for model_name in candidates}
        if suggested_spaces:
            corrections.append(
                {
                    "field": "search_spaces",
                    "suggested": suggested_spaces,
                    "resolved": search_spaces,
                    "reason": "Agent 任务类型不匹配，丢弃可能不兼容的搜索参数",
                }
            )
    else:
        search_spaces = {}
        for model_name in candidates:
            safe_space, space_corrections = _validate_search_space(
                model_name,
                resolved_type,
                suggested_spaces.get(model_name, {}),
            )
            search_spaces[model_name] = safe_space
            corrections.extend(space_corrections)
    return (
        {
            "task_type": resolved_type,
            "optimization_metric": metric,
            "candidate_models": candidates,
            "n_trials": n_trials,
            "search_spaces": search_spaces,
            "rationale": suggested.get("rationale", ""),
        },
        suggested_type,
        corrections,
    )


def _validate_search_space(
    model_name: str, resolved_type: str, search_space: Any
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Drop rules outside the estimator's canonical safe tuning domain."""
    from model_code.model_factory import ModelFactory

    corrections: list[Dict[str, Any]] = []
    if not isinstance(search_space, dict):
        return {}, [
            {
                "field": f"search_spaces.{model_name}",
                "suggested": search_space,
                "resolved": {},
                "reason": "搜索空间必须是参数到规则的对象",
            }
        ]

    parameter_domains = dict(ModelFactory.get_param_grid(model_name))
    if model_name == "catboost":
        parameter_domains["n_estimators"] = parameter_domains["iterations"]
        if resolved_type != ProblemType.REGRESSION.value:
            parameter_domains["auto_class_weights"] = ["Balanced", "SqrtBalanced"]
    if (
        model_name in {"lightgbm", "xgboost"}
        and resolved_type != ProblemType.REGRESSION.value
    ):
        # This parameter is task-specific and therefore is not in the shared
        # factory grid.  The bounds admit the documented Agent suggestions
        # while excluding non-positive or unbounded estimator inputs.
        parameter_domains["scale_pos_weight"] = [1e-6, 1e6]

    safe_space: Dict[str, Any] = {}
    for param_name, rule in search_space.items():
        field = f"search_spaces.{model_name}.{param_name}"
        domain = parameter_domains.get(param_name)
        reason = _search_rule_error(param_name, rule, parameter_domains)
        if reason is None:
            safe_rule = dict(rule)
            if safe_rule["type"] == "categorical" and _is_integer_domain(domain):
                safe_rule["choices"] = [int(value) for value in safe_rule["choices"]]
            safe_space[param_name] = safe_rule
        else:
            corrections.append(
                {
                    "field": field,
                    "suggested": rule,
                    "resolved": None,
                    "reason": reason,
                }
            )
    return safe_space, corrections


def _search_rule_error(
    param_name: str, rule: Any, parameter_domains: Dict[str, Any]
) -> Optional[str]:
    if param_name not in parameter_domains:
        return "参数不受该任务对应的 estimator 支持，已移除"
    if not isinstance(rule, dict):
        return "搜索规则必须是对象，已移除"
    rule_type = rule.get("type")
    if rule_type not in {"int", "float", "loguniform", "categorical"}:
        return "搜索规则类型无效，已移除"
    if rule_type == "categorical":
        choices = rule.get("choices")
        if not isinstance(choices, list) or not choices:
            return "categorical 规则必须提供非空 choices，已移除"
        return _categorical_domain_error(choices, parameter_domains[param_name])

    if _is_integer_domain(parameter_domains[param_name]) and rule_type != "int":
        return "整数 estimator 参数只支持 int 或整数 categorical 规则，已移除"

    low = rule.get("low")
    high = rule.get("high")
    if rule_type == "int":
        valid_bounds = (
            isinstance(low, int)
            and not isinstance(low, bool)
            and isinstance(high, int)
            and not isinstance(high, bool)
        )
    else:
        valid_bounds = (
            isinstance(low, (int, float))
            and not isinstance(low, bool)
            and isinstance(high, (int, float))
            and not isinstance(high, bool)
            and np.isfinite(low)
            and np.isfinite(high)
        )
    if not valid_bounds or low > high:
        return "数值搜索规则必须提供有效且有序的 low/high，已移除"
    if rule_type == "loguniform" and low <= 0:
        return "loguniform 的 low 必须大于 0，已移除"
    domain_error = _numeric_domain_error(low, high, parameter_domains[param_name])
    if domain_error is not None:
        return domain_error
    return None


def _canonical_domain_values(domain: Any) -> list[Any]:
    """Normalize ModelFactory grids without weakening numpy scalar checks."""
    if isinstance(domain, np.ndarray):
        return domain.tolist()
    if isinstance(domain, (list, tuple)):
        return list(domain)
    return [domain]


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _is_integer_domain(domain: Any) -> bool:
    supported = _canonical_domain_values(domain)
    return bool(supported) and all(
        _is_finite_number(value) and float(value).is_integer()
        for value in supported
    )


def _categorical_domain_error(choices: list[Any], domain: Any) -> Optional[str]:
    supported = _canonical_domain_values(domain)
    if supported and all(_is_finite_number(value) for value in supported):
        lower, upper = min(supported), max(supported)
        integer_domain = _is_integer_domain(domain)
        if all(
            _is_finite_number(choice) and lower <= choice <= upper
            and (not integer_domain or float(choice).is_integer())
            for choice in choices
        ):
            return None
    elif all(any(choice == value for value in supported) for choice in choices):
        return None
    return "categorical choices 超出该 estimator 的安全支持域，已移除"


def _numeric_domain_error(low: Any, high: Any, domain: Any) -> Optional[str]:
    supported = _canonical_domain_values(domain)
    if not supported or not all(_is_finite_number(value) for value in supported):
        return "数值规则不适用于该 estimator 参数，已移除"
    if low < min(supported) or high > max(supported):
        return "low/high 超出该 estimator 的安全支持域，已移除"
    return None


def _with_residual_categorical_encoding(
    df: pd.DataFrame, target_col: str, strategy: FeatureStrategy
) -> FeatureStrategy:
    """Ensure workflow estimators receive numeric features and inference reuses the encoding."""
    handled = {
        action.column
        for action in strategy.actions
        if action.action == "drop" or action.action.endswith("_encode")
    }
    automatic = [
        FeatureAction(
            column=column,
            action="frequency_encode",
            reason="工作流为残留类别特征应用确定性无监督编码",
        )
        for column in df.columns
        if column != target_col
        and column not in handled
        and (str(df[column].dtype) == "object" or "category" in str(df[column].dtype))
    ]
    if not automatic:
        return strategy
    return FeatureStrategy(actions=[*strategy.actions, *automatic])

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
        
        # =====================================================================
        # [DEMO MODE] 模拟数据 — 真实环境请取消下方注释并移除本标记块
        # from model_code.database import DatabaseConnector
        # db = DatabaseConnector(db_config)
        # db.connect()
        # df = db.query(sql_result.sql)
        # db.disconnect()
        # =====================================================================

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
        task_dir = get_task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始数据
        df.to_parquet(task_dir / "raw_data.parquet", index=False)
        
        # 保存 eda_summary
        with open(task_dir / "eda_summary.json", "w", encoding='utf-8') as f:
            json.dump(eda_summary, f, indent=2, ensure_ascii=False)
        
        # 保存 SQL
        with open(task_dir / "query.sql", "w", encoding='utf-8') as f:
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
        task_dir = get_task_dir(req.task_id)
        eda_summary_path = task_dir / "eda_summary.json"
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
        with open(task_dir / "feature_strategy.json", "w", encoding='utf-8') as f:
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

WORKFLOW_HOLDOUT_SIZE = 0.25
WORKFLOW_RANDOM_STATE = 42


@dataclass
class WorkflowTrainingContext:
    task_id: str
    task_dir: Path
    clean_frame: pd.DataFrame
    target_col: str
    strategy: FeatureStrategy
    execution_strategy: FeatureStrategy
    eda_summary: Dict[str, Any]
    spec: ProblemSpec
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_holdout: pd.Series
    cv_folds: int


def _workflow_detail(status_code: int, code: str, field: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "field": field, "message": message},
    )


def _prepare_workflow_training(
    task_id: str, problem_type: str, cv_folds: int
) -> WorkflowTrainingContext:
    """Resolve target semantics, holdout, and exact CV feasibility before SSE."""
    from sklearn.model_selection import train_test_split

    task_dir = get_task_dir(task_id)
    data_path = task_dir / "raw_data.parquet"
    if not data_path.is_file():
        raise _workflow_detail(404, "workflow_data_missing", "task_id", "工作流原始数据不存在")
    strategy_path = task_dir / "feature_strategy.json"
    if not strategy_path.is_file():
        raise _workflow_detail(404, "workflow_strategy_missing", "task_id", "工作流特征策略不存在")

    try:
        frame = pd.read_parquet(data_path)
    except Exception as exc:
        logger.exception("读取工作流数据失败 task_id=%s", task_id)
        raise _workflow_detail(400, "workflow_data_invalid", "task_id", "工作流原始数据无法读取") from exc
    try:
        strategy = FeatureStrategy.model_validate_json(strategy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("读取工作流特征策略失败 task_id=%s", task_id)
        raise _workflow_detail(422, "workflow_strategy_invalid", "feature_strategy", "工作流特征策略无效") from exc

    eda_summary: Dict[str, Any] = {}
    eda_path = task_dir / "eda_summary.json"
    if eda_path.is_file():
        try:
            loaded = json.loads(eda_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                eda_summary = loaded
        except (OSError, ValueError) as exc:
            logger.exception("读取工作流 EDA 摘要失败 task_id=%s", task_id)
            raise _workflow_detail(422, "workflow_eda_invalid", "task_id", "工作流 EDA 摘要无效") from exc
    target_col = eda_summary.get("target", "is_fraud")
    if not isinstance(target_col, str) or target_col not in frame.columns:
        raise _workflow_detail(422, "target_not_found", "target_col", "目标列不存在")

    clean_frame = frame.loc[frame[target_col].notna()].copy()
    try:
        spec = resolve_problem_spec(clean_frame[target_col], target_col, problem_type)
        validate_train_test_split(spec, clean_frame[target_col], WORKFLOW_HOLDOUT_SIZE)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc

    X = clean_frame.drop(columns=[target_col])
    if X.shape[1] == 0:
        raise _workflow_detail(422, "features_missing", "target_col", "未找到可训练特征列")
    y = clean_frame[target_col]
    stratify = y if spec.resolved_type != ProblemType.REGRESSION else None
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=WORKFLOW_HOLDOUT_SIZE,
        random_state=WORKFLOW_RANDOM_STATE,
        stratify=stratify,
    )
    try:
        validate_cv(spec, y_train, cv_folds)
    except ProblemSpecError as exc:
        raise HTTPException(status_code=422, detail=exc.as_detail()) from exc

    execution_strategy = _with_residual_categorical_encoding(clean_frame, target_col, strategy)
    return WorkflowTrainingContext(
        task_id=task_id,
        task_dir=task_dir,
        clean_frame=clean_frame,
        target_col=target_col,
        strategy=strategy,
        execution_strategy=execution_strategy,
        eda_summary=eda_summary,
        spec=spec,
        X_train=X_train,
        X_holdout=X_holdout,
        y_train=y_train,
        y_holdout=y_holdout,
        cv_folds=cv_folds,
    )


def _safe_workflow_error(exc: Exception) -> str:
    logger.error("工作流训练失败 error_type=%s", type(exc).__name__, exc_info=True)
    return "模型训练失败，请检查配置后重试"

@router.get("/step3_train_stream")
async def train_stream(
    task_id: str = Query(..., description="任务 ID"),
    problem_type: str = Query("auto", description="任务类型：auto/分类/回归"),
    n_trials: int = Query(30, description="Optuna Trial 次数", ge=10, le=100),
    cv_folds: int = Query(5, description="交叉验证折数", ge=3, le=10)
):
    """预检并启动一个 task 独占、holdout-safe 的训练运行。"""
    context = _prepare_workflow_training(task_id, problem_type, cv_folds)
    control = _claim_workflow_run(task_id)
    run_id = control.run_id
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    completion_payload: Dict[str, Any] = {}
    problem_spec = context.spec.to_dict()
    state_metadata: Dict[str, Any] = {
        "requested_type": problem_type,
        "problem_spec": problem_spec,
        "cv_folds": cv_folds,
    }

    try:
        _write_run_state(task_id, run_id, {"status": "running", **state_metadata})
    except Exception:
        _release_workflow_run(task_id, run_id)
        raise
    
    def background_training_task():
        """Run model selection on training rows and evaluate once on holdout."""
        try:
            logger.info(
                "任务 %s: 开始训练 run_id=%s requested_type=%s cv_folds=%s",
                task_id,
                run_id,
                problem_type,
                cv_folds,
            )
            _require_current_run(task_id, run_id)
            spec = context.spec
            data_features = extract_data_features(context.clean_frame, context.target_col)
            data_features["task_type"] = spec.resolved_type.value
            scenario = context.eda_summary.get("user_query", "通用机器学习建模")
            suggested_config = recommend_automl_config(data_features, scenario)
            optuna_config, suggested_type, agent_corrections = _coordinate_automl_config(
                suggested_config, spec, n_trials
            )
            state_metadata.update(
                {
                    "problem_spec": problem_spec,
                    "suggested_type": suggested_type,
                    "agent_corrections": agent_corrections,
                    "best_metric": optuna_config["optimization_metric"],
                }
            )
            _write_run_state(task_id, run_id, {"status": "running", **state_metadata})
            logger.info(
                "任务 %s: resolved_type=%s suggested_type=%s metric=%s candidates=%s corrections=%s",
                task_id,
                spec.resolved_type.value,
                suggested_type,
                optuna_config["optimization_metric"],
                optuna_config["candidate_models"],
                agent_corrections,
            )

            # 标签编码只 fit train；holdout 仅 transform。
            if spec.resolved_type == ProblemType.REGRESSION:
                label_encoder = None
                y_train = pd.Series(
                    pd.to_numeric(context.y_train, errors="raise").astype(float),
                    name=context.target_col,
                )
                y_holdout = pd.Series(
                    pd.to_numeric(context.y_holdout, errors="raise").astype(float),
                    name=context.target_col,
                )
            else:
                from sklearn.preprocessing import LabelEncoder

                label_encoder = LabelEncoder()
                y_train = pd.Series(
                    label_encoder.fit_transform(context.y_train),
                    index=context.y_train.index,
                    name=context.target_col,
                )
                y_holdout = pd.Series(
                    label_encoder.transform(context.y_holdout),
                    index=context.y_holdout.index,
                    name=context.target_col,
                )

            # 4. SSE 回调函数 (适配 AutoMLOptimizer.optimize() 的返回值)
            import optuna

            def sse_callback(study: optuna.Study, trial: optuna.Trial):
                """Optuna Trial 结束时的回调，推送 SSE"""
                if trial.state.is_finished():
                    _require_current_run(task_id, run_id)
                    try:
                        current_best = float(study.best_value)
                    except ValueError:
                        current_best = None
                    payload = {
                        "status": "RUNNING",
                        "trial": trial.number,
                        "metric": optuna_config["optimization_metric"],
                        "value": float(trial.value) if trial.value is not None else None,
                        "best_value": current_best,
                        "best_metric": optuna_config["optimization_metric"],
                        "params": trial.params,
                        "problem_spec": problem_spec,
                        "suggested_type": suggested_type,
                        "agent_corrections": agent_corrections,
                        "run_id": run_id,
                        "cv_folds": cv_folds,
                    }
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
                    except Exception as e:
                        logger.error(f"推送 SSE 数据失败：{e}")

            feature_transformer = FeatureEngineeringTransformer(
                strategy=context.execution_strategy,
                target_col=context.target_col,
                allow_target_encoding=False,
            )
            optimizer = AutoMLOptimizer(
                context.X_train,
                y_train,
                optuna_config,
                cv_splits=cv_folds,
                n_jobs=1,
                feature_transformer=feature_transformer,
                cancel_event=control.cancel_event,
            )

            # 运行优化并获取最佳模型
            try:
                # sse_callback 实时推送每个 Trial（不再结束后回放）
                study, fitted_pipeline = optimizer.optimize(
                    n_trials=n_trials, show_progress=False, callbacks=[sse_callback]
                )
                best_value = float(study.best_value)
                selected_algorithm = study.best_params.get("model_type") or optuna_config[
                    "candidate_models"
                ][0]
            except Exception as e:
                logger.error(f"Optuna 优化失败：{e}")
                raise

            _require_current_run(task_id, run_id)
            fitted_transformer = fitted_pipeline.named_steps["features"]
            feature_executor = fitted_transformer.executor_
            best_model = fitted_pipeline.named_steps["model"]

            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            artifact_path = get_saved_path(f"model_{validate_task_id(task_id)}.pkl")

            best_metric = optuna_config["optimization_metric"]
            if spec.resolved_type == ProblemType.REGRESSION:
                metrics = compute_regression_metrics(fitted_pipeline, context.X_holdout, y_holdout)
            else:
                metrics = compute_classification_metrics(fitted_pipeline, context.X_holdout, y_holdout)
            metrics_evaluation = {
                "scope": "holdout",
                "rows": int(len(context.X_holdout)),
                "train_rows": int(len(context.X_train)),
                "random_state": WORKFLOW_RANDOM_STATE,
            }
            artifact = {
                "artifact_schema_version": 3,
                "problem_type": spec.resolved_type.value,
                "problem_spec": problem_spec,
                "metric_directions": spec.metric_directions,
                "run_id": run_id,
                "cv_folds": cv_folds,
                "feature_executor": feature_executor,
                "model": best_model,
                "label_encoder": label_encoder,
                "config": optuna_config,
                "best_metric": best_metric,
                "best_value": best_value,
                "metrics": metrics,
                "metrics_evaluation": metrics_evaluation,
                "selected_algorithm": selected_algorithm,
                "suggested_type": suggested_type,
                "agent_corrections": agent_corrections,
                "feature_columns": list(feature_executor.state["feature_columns"]),
            }
            _persist_run_artifact(task_id, run_id, artifact, artifact_path)

            logger.info("模型已保存 task_id=%s run_id=%s", task_id, run_id)
            completed_state = {
                "status": "completed",
                **state_metadata,
                "best_metric": best_metric,
                "best_value": best_value,
                "metrics": metrics,
                "metrics_evaluation": metrics_evaluation,
                "selected_algorithm": selected_algorithm,
                "cv_folds": cv_folds,
            }
            _write_run_state(task_id, run_id, completed_state)
            logger.info(
                "workflow_training_completed task_id=%s resolved_type=%s "
                "selected_algorithm=%s metric=%s",
                task_id,
                spec.resolved_type.value,
                selected_algorithm,
                best_metric,
            )
            completion_payload.update(
                {
                    "status": "DONE",
                    "metric": best_metric,
                    "value": best_value,
                    "problem_spec": problem_spec,
                    "suggested_type": suggested_type,
                    "agent_corrections": agent_corrections,
                    "best_metric": best_metric,
                    "best_value": best_value,
                    "metrics": metrics,
                    "metrics_evaluation": metrics_evaluation,
                    "selected_algorithm": selected_algorithm,
                    "run_id": run_id,
                    "cv_folds": cv_folds,
                }
            )

            # 基线只使用生产模型实际拟合的 train 行；holdout 不进入拟合状态。
            try:
                from model_code.mlops_monitor import DataDriftMonitor

                monitor = DataDriftMonitor()
                transformed_train = feature_executor.transform(context.X_train)
                baseline_data = monitor.extract_baseline(
                    transformed_train, features=list(transformed_train.columns)
                )

                baseline_path = get_saved_path(f"baseline_{validate_task_id(task_id)}.json")
                temporary_baseline = baseline_path.with_name(f".{baseline_path.name}.{run_id}.tmp")
                try:
                    monitor.save_baseline(baseline_data, temporary_baseline)
                    with workflow_run_lock:
                        _require_current_run(task_id, run_id)
                        os.replace(temporary_baseline, baseline_path)
                finally:
                    temporary_baseline.unlink(missing_ok=True)
                logger.info("基线数据已保存 task_id=%s run_id=%s", task_id, run_id)
            except Exception as e:
                logger.warning("保存基线数据失败 task_id=%s error_type=%s", task_id, type(e).__name__)

        except StaleWorkflowRun:
            logger.warning("忽略失效工作流运行 task_id=%s run_id=%s", task_id, run_id)
        except OptimizationCancelled:
            canonical_error = "训练已取消"
            try:
                _write_run_state(
                    task_id,
                    run_id,
                    {"status": "cancelled", **state_metadata, "error": canonical_error},
                    allow_cancelled=True,
                )
                asyncio.run_coroutine_threadsafe(
                    queue.put({"status": "CANCELLED", "error": canonical_error, "run_id": run_id}), loop
                )
            except StaleWorkflowRun:
                pass
        except Exception as e:
            canonical_error = _safe_workflow_error(e)
            try:
                _write_run_state(
                    task_id,
                    run_id,
                    {"status": "failed", **state_metadata, "error": canonical_error},
                    allow_cancelled=True,
                )
                asyncio.run_coroutine_threadsafe(
                    queue.put({"status": "ERROR", "error": canonical_error, "run_id": run_id}), loop
                )
            except StaleWorkflowRun:
                logger.warning("失效运行未写失败状态 task_id=%s run_id=%s", task_id, run_id)
        finally:
            _release_workflow_run(task_id, run_id)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
    
    # 将阻塞的训练任务扔进线程池
    loop.run_in_executor(executor_pool, background_training_task)
    
    # SSE 事件生成器
    async def event_generator():
        """
        SSE 事件流生成器

        【优化】：超时时间从 300 秒改为 60 秒，更快响应连接断开
        """
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=60)

                if data is None:
                    if completion_payload:
                        yield f"data: {json.dumps(completion_payload, ensure_ascii=False)}\n\n"
                    break

                yield f"data: {json.dumps(data)}\n\n"

            except asyncio.TimeoutError:
                # 超时，发送心跳 (60 秒无数据)
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


@router.post("/task/{task_id}/run/{run_id}/cancel")
async def cancel_workflow_run(task_id: str, run_id: str):
    validate_task_id(task_id)
    with workflow_run_lock:
        control = active_workflow_runs.get(task_id)
        if control is None or control.run_id != run_id:
            raise HTTPException(status_code=404, detail="未找到活动训练运行")
        control.cancel_event.set()
    return {"task_id": task_id, "run_id": run_id, "status": "cancelling"}


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
        model_version = validate_task_id(req.model_version)
        artifact_path = get_saved_path(f"model_{model_version}.pkl")
        if not os.path.exists(artifact_path):
            raise HTTPException(
                status_code=404,
                detail=f"模型未找到：{model_version}，请先训练模型"
            )

        artifact = joblib.load(artifact_path)
        model = artifact.get("model")  # mock 模式下为 None

        logger.info(f"加载模型：{model_version}")

        # 2. mock 模式保留模拟预测
        if model is None:
            logger.warning("模型未加载，使用模拟预测结果")
            return {
                "status": "success",
                "model_version": model_version,
                "prediction": 0.85,
                "probability": 0.85,
                "input_features": req.features
            }

        # 3. 共享推理：executor 仅 transform（绝不重新 fit）、无二次 get_dummies、
        #    标签反变换为原始标签、概率为预测类别的概率
        result = predict_with_artifact(artifact, [req.features])[0]

        return {
            "status": "success",
            "model_version": model_version,
            "prediction": result["prediction"],
            "probability": result["probability"],
            "encoded_prediction": result["encoded_prediction"],
            "diagnostics": result["diagnostics"],
            "input_features": req.features
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 辅助接口 ====================

@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_dir = get_task_dir(task_id)

    if not os.path.exists(task_dir):
        return {"status": "not_found", "task_id": task_id}

    files = os.listdir(task_dir)

    status = {
        "task_id": task_id,
        "status": "in_progress",
        "files": files
    }
    persisted_state = _read_workflow_state(task_id)
    if persisted_state:
        status.update(
            {
                key: value
                for key, value in persisted_state.items()
                if key not in {"task_id", "files"}
            }
        )

    if "raw_data.parquet" in files:
        status["step1"] = "done"
    if "feature_strategy.json" in files:
        status["step2"] = "done"
    if status["status"] == "completed":
        status["step3"] = "done"

    model_path = get_saved_path(f"model_{validate_task_id(task_id)}.pkl")
    can_restore_from_artifact = persisted_state is None or status["status"] == "completed"
    if model_path.exists() and can_restore_from_artifact:
        status["step3"] = "done"
        status["status"] = "completed"

        # 加载模型获取通用指标和持久化任务语义。
        try:
            import joblib
            artifact = joblib.load(model_path)
            status["best_metric"] = artifact.get("best_metric") or artifact.get("config", {}).get(
                "optimization_metric", "roc_auc"
            )
            status["best_value"] = artifact.get("best_value", 0)
            status["feature_columns"] = artifact.get("feature_columns", [])
            for field in (
                "problem_spec",
                "suggested_type",
                "agent_corrections",
                "metrics",
                "metrics_evaluation",
            ):
                if field in artifact:
                    status[field] = artifact[field]

            # 只有没有 schema-v3 任务元数据的旧分类产物保留 best_auc。
            schema_version = artifact.get("artifact_schema_version", 0)
            if int(schema_version) < 3 and artifact.get("problem_type") != "regression":
                status["best_auc"] = artifact.get("best_value", 0)
        except Exception as e:
            logger.warning(f"读取模型状态失败：{e}")

    return status


@router.get("/task/{task_id}/importance")
async def get_feature_importance(task_id: str):
    """
    获取特征重要性

    Args:
        task_id: 任务 ID

    Returns:
        List[Dict]: 特征重要性列表
    """
    model_path = get_saved_path(f"model_{validate_task_id(task_id)}.pkl")

    if not os.path.exists(model_path):
        return []

    try:
        import joblib
        artifact = joblib.load(model_path)
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns", [])

        if model is None:
            return []

        if not hasattr(model, 'feature_importances_'):
            return []

        importances = model.feature_importances_

        # 构建特征重要性列表
        importance_list = []
        for i, col in enumerate(feature_columns):
            if i < len(importances):
                importance_list.append({
                    "feature": col,
                    "importance": float(importances[i])
                })

        # 按重要性排序
        importance_list.sort(key=lambda x: x["importance"], reverse=True)

        return importance_list

    except Exception as e:
        logger.error(f"获取特征重要性失败：{e}")
        return []


@router.get("/models")
async def list_models():
    """列出所有已训练的模型"""
    if not SAVED_DIR.exists():
        return {"models": []}

    models = []
    for f in os.listdir(SAVED_DIR):
        if f.startswith("model_task_") and f.endswith(".pkl"):
            task_id = f.replace("model_", "").replace(".pkl", "")
            try:
                validate_task_id(task_id)
                models.append({
                    "task_id": task_id,
                    "model_path": str(get_saved_path(f))
                })
            except HTTPException:
                logger.warning(f"跳过无效模型文件名：{f}")

    return {"models": models}
