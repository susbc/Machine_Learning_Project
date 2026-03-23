# AutoML - 自动机器学习模块
"""
AutoML 模块 - Phase 3 核心组件

包含:
- AutoMLAgent: AutoML 智能体 (模型推荐 + 搜索空间生成)
- AutoMLOptimizer: Optuna 优化器 (贝叶斯寻优)
- extract_data_features: 数据特征提取函数
- plot_optimization_results: 优化结果可视化
"""

from .automl_agent import AutoMLAgent, extract_data_features
from .automl_optimizer import AutoMLOptimizer, plot_optimization_results

__all__ = [
    "AutoMLAgent",
    "AutoMLOptimizer",
    "extract_data_features",
    "plot_optimization_results",
]
