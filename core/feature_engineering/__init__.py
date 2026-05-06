# Feature Engineering - 特征工程模块
"""
特征工程模块 - Phase 2 核心组件

包含:
- EDAAgent: 探索性数据分析智能体
- FeatureAgent: 特征工程智能体

注: FeatureEngineeringExecutor 请使用 model_code.feature_executor 中的有状态版本
"""

from .eda_agent import EDAAgent, compute_eda_summary
from .feature_agent import FeatureAgent

__all__ = [
    "EDAAgent",
    "FeatureAgent",
    "compute_eda_summary",
]
