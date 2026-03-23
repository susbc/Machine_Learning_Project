"""
Agent Output Schemas - Pydantic 严格校验规则

防崩核心：用 Pydantic 严格类型校验 + Literal 枚举约束，防止 LLM 格式幻觉。
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict, Any


# ==========================================
# 1. DataRetrieval_Agent 的输出协议
# ==========================================

class SQLGenerationResult(BaseModel):
    """
    NL2SQL 生成结果
    
    用于 DataRetrieval_Agent 的严格输出校验
    """
    thought_process: str = Field(
        ...,
        description="思考过程，说明选择了哪些表和字段，以及为什么这么选。"
    )
    is_executable: bool = Field(
        ...,
        description="布尔值。如果数据充足可以生成完整 SQL 则为 true；否则为 false。"
    )
    sql: Optional[str] = Field(
        None,
        description="具体的 SQL 语句。如果 is_executable 为 false，此字段为空。"
    )
    
    @validator('sql')
    def validate_sql(cls, v, values):
        """验证 SQL 字段与 is_executable 的一致性"""
        if values.get('is_executable') and not v:
            raise ValueError("is_executable 为 true 时，sql 字段不能为空")
        if not values.get('is_executable') and v:
            raise ValueError("is_executable 为 false 时，sql 字段应为空")
        return v


# ==========================================
# 2. Feature_Agent 的输出协议 (特征工程策略)
# ==========================================

class FeatureAction(BaseModel):
    """
    单个特征处理动作
    
    【核心防崩点】：action 严格限制为枚举值，LLM 自创方法立即报错！
    """
    column: str = Field(
        ...,
        description="目标特征列名"
    )
    
    # 严格枚举约束 - LLM 只能从这些选项中选择
    action: Literal[
        "drop",                    # 删除字段
        "fillna_mean",             # 均值填充
        "fillna_median",           # 中位数填充
        "fillna_mode",             # 众数填充
        "fillna_constant",         # 常数填充
        "log_transform",           # 对数变换
        "sqrt_transform",          # 平方根变换
        "boxcox_transform",        # Box-Cox 变换
        "onehot_encode",           # One-Hot 编码
        "label_encode",            # Label 编码
        "target_encode",           # Target Encoding
        "frequency_encode",        # 频率编码
        "extract_hour",            # 提取小时
        "extract_day",             # 提取星期
        "extract_month",           # 提取月份
        "extract_year",            # 提取年份
        "binning",                 # 分箱处理
        "interaction",             # 特征交叉
        "scale_standard",          # 标准化
        "scale_minmax",            # Min-Max 归一化
        "scale_robust"             # Robust 标准化
    ] = Field(
        ...,
        description="对该特征执行的具体处理动作"
    )
    
    reason: str = Field(
        ...,
        description="执行该操作的业务或统计学理由（20 字以内）"
    )
    
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="可选参数，如 {'value': 0} 用于 fillna_constant"
    )
    
    @validator('reason')
    def validate_reason_length(cls, v):
        """验证理由长度"""
        if len(v) > 50:
            raise ValueError("理由描述过长，请精简到 50 字以内")
        return v


class FeatureStrategy(BaseModel):
    """
    特征工程完整策略
    
    包含所有特征的处理动作列表
    """
    actions: List[FeatureAction] = Field(
        ...,
        description="特征处理动作列表"
    )
    
    @validator('actions')
    def validate_actions_not_empty(cls, v):
        """验证动作列表不为空"""
        if not v:
            raise ValueError("特征处理动作列表不能为空")
        return v


# ==========================================
# 3. AutoML_Agent 的输出协议
# ==========================================

class SearchSpaceParam(BaseModel):
    """单个超参数的搜索空间定义"""
    type: Literal["int", "float", "loguniform", "categorical"] = Field(
        ...,
        description="参数类型"
    )
    low: Optional[float] = Field(None, description="下界 (int/float/loguniform)")
    high: Optional[float] = Field(None, description="上界 (int/float/loguniform)")
    choices: Optional[List[Any]] = Field(None, description="可选值列表 (categorical)")


class ModelSearchSpace(BaseModel):
    """单个模型的搜索空间"""
    learning_rate: Optional[SearchSpaceParam] = None
    num_leaves: Optional[SearchSpaceParam] = None
    max_depth: Optional[SearchSpaceParam] = None
    n_estimators: Optional[SearchSpaceParam] = None
    scale_pos_weight: Optional[SearchSpaceParam] = None
    l2_leaf_reg: Optional[SearchSpaceParam] = None
    auto_class_weights: Optional[SearchSpaceParam] = None
    # 可以扩展更多参数...


class AutoMLConfig(BaseModel):
    """
    AutoML 完整配置
    
    用于 AutoML_Agent 的严格输出校验
    """
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression"
    ] = Field(..., description="任务类型")
    
    optimization_metric: Literal[
        "roc_auc", "accuracy", "f1", "precision", "recall",
        "rmse", "mse", "mae", "r2"
    ] = Field(..., description="优化指标")
    
    candidate_models: List[Literal[
        "lightgbm", "xgboost", "catboost",
        "random_forest", "logistic_regression"
    ]] = Field(..., description="候选模型列表")
    
    n_trials: int = Field(..., description="Optuna Trial 次数", ge=10, le=200)
    
    rationale: str = Field(..., description="推荐理由说明")
    
    search_spaces: Dict[str, ModelSearchSpace] = Field(
        ...,
        description="各模型的搜索空间配置"
    )
    
    @validator('candidate_models')
    def validate_models_not_empty(cls, v):
        """验证候选模型列表不为空"""
        if not v:
            raise ValueError("候选模型列表不能为空")
        if len(v) > 5:
            raise ValueError("候选模型最多 5 个，避免搜索空间过大")
        return v
    
    @validator('n_trials')
    def validate_n_trials(cls, v):
        """验证 Trial 次数在合理范围"""
        if v < 10:
            raise ValueError("Trial 次数至少 10 次，否则无法有效搜索")
        if v > 200:
            raise ValueError("Trial 次数最多 200 次，避免耗时过长")
        return v


# ==========================================
# 4. EDA_Agent 的输出协议
# ==========================================

class RiskFlag(BaseModel):
    """单个风险标记"""
    type: Literal[
        "high_missing_rate",
        "severe_skewness",
        "single_value",
        "business_conflict",
        "potential_leakage",
        "high_correlation",
        "high_cardinality"
    ] = Field(..., description="风险类型")
    
    column: str = Field(..., description="相关字段名")
    
    value: Any = Field(..., description="风险值 (如 null_rate=0.15)")
    
    severity: Literal["high", "medium", "low"] = Field(..., description="风险等级")


class EDAReport(BaseModel):
    """
    EDA 分析报告
    
    用于 EDA_Agent 的严格输出校验
    """
    overview: str = Field(..., description="数据概览")
    
    quality_alerts: List[str] = Field(
        default_factory=list,
        description="数据质量告警列表"
    )
    
    risk_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="风险标记列表"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="处理建议列表"
    )


# ==========================================
# 导出所有 Schema
# ==========================================

__all__ = [
    "SQLGenerationResult",
    "FeatureAction",
    "FeatureStrategy",
    "AutoMLConfig",
    "ModelSearchSpace",
    "SearchSpaceParam",
    "EDAReport",
    "RiskFlag",
]
