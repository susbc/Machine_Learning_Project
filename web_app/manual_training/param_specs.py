"""
param_specs.py — 每算法的 UI 参数元数据。

定义 ParamSpec dataclass + get_specs(algorithm) 返回某算法在 UI 中暴露的参数列表。
供 tab_train.py 动态渲染 widgets，供 llm_tuning.py 做越界截断。
"""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    widget: Literal["slider", "number_input", "select", "checkbox"]
    default: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step: Optional[Any] = None
    fmt: Optional[str] = None
    options: Optional[List[Any]] = None
    help: Optional[str] = None


_ALGO_SPECS: dict[str, list[ParamSpec]] = {
    "lightgbm": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量，越多精度越高但越慢"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率，越小越稳定但需要更多树"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 6, 3, 15, 1, help="限制树深度，防止过拟合"),
        ParamSpec("num_leaves", "num_leaves (叶子节点数)", "slider", 50, 10, 200, 10, help="LightGBM 核心参数，控制模型复杂度"),
        ParamSpec("min_child_samples", "min_child_samples (叶子最小样本)", "slider", 20, 5, 100, 5, help="叶子节点最少样本数"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树使用的样本比例"),
        ParamSpec("colsample_bytree", "colsample_bytree (特征采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树使用的特征比例"),
        ParamSpec("reg_alpha", "reg_alpha (L1 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L1 正则化系数"),
        ParamSpec("reg_lambda", "reg_lambda (L2 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L2 正则化系数"),
    ],
    "xgboost": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 6, 3, 15, 1, help="树的最大深度"),
        ParamSpec("min_child_weight", "min_child_weight (叶子最小权重)", "slider", 1, 1, 20, 1, help="叶子节点最小权重和"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="样本采样比例"),
        ParamSpec("colsample_bytree", "colsample_bytree (特征采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="特征采样比例"),
        ParamSpec("reg_alpha", "reg_alpha (L1 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L1 正则化"),
        ParamSpec("reg_lambda", "reg_lambda (L2 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L2 正则化"),
    ],
    "catboost": [
        ParamSpec("iterations", "iterations (迭代次数)", "slider", 200, 50, 1000, 50, help=" boosting 迭代次数"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率"),
        ParamSpec("depth", "depth (树深度)", "slider", 6, 3, 15, 1, help="树深度"),
        ParamSpec("l2_leaf_reg", "l2_leaf_reg (L2 正则)", "slider", 3.0, 0.1, 20.0, 0.5, "%.1f", help="L2 正则化系数"),
        ParamSpec("border_count", "border_count (分箱数)", "slider", 256, 32, 512, 32, help="数值特征分箱数量"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树样本采样比例"),
    ],
    "random_forest": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 10, 3, 30, 1, help="树最大深度，None 表示不限制"),
        ParamSpec("min_samples_split", "min_samples_split (最小分割样本)", "slider", 2, 2, 20, 1, help="内部节点再分割所需最小样本数"),
        ParamSpec("min_samples_leaf", "min_samples_leaf (叶子最小样本)", "slider", 1, 1, 10, 1, help="叶子节点最少样本数"),
        ParamSpec("max_features", "max_features (最大特征数)", "select", "sqrt", options=["sqrt", "log2", None], help="每次分割考虑的最大特征数"),
    ],
    "logistic_regression": [
        ParamSpec("C", "C (正则化强度倒数)", "slider", 1.0, 0.001, 100.0, 0.1, "%.3f", help="正则化强度的倒数，越小正则化越强"),
        ParamSpec("penalty", "penalty (正则化类型)", "select", "l2", options=["l1", "l2"], help="正则化类型"),
        ParamSpec("solver", "solver (优化算法)", "select", "liblinear", options=["liblinear", "saga"], help="优化算法，l1 正则需 liblinear 或 saga"),
        ParamSpec("max_iter", "max_iter (最大迭代次数)", "slider", 1000, 100, 5000, 100, help="优化器最大迭代次数"),
    ],
}


def get_specs(algorithm: str) -> list[ParamSpec]:
    """返回某算法在 UI 中暴露的参数列表。"""
    specs = _ALGO_SPECS.get(algorithm)
    if specs is None:
        raise ValueError(f"不支持的算法：{algorithm}。支持的算法：{list(_ALGO_SPECS.keys())}")
    return specs


def get_supported_algorithms() -> list[str]:
    """返回所有支持的算法标识。"""
    return list(_ALGO_SPECS.keys())
