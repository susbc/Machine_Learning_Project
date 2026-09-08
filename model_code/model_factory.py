"""
模型工厂 - 统一的算法实例化入口

支持 6 种算法:
- lightgbm (LightGBM)
- xgboost (XGBoost)
- catboost (CatBoost)
- random_forest (随机森林)
- logistic_regression (逻辑回归)
- linear_regression (线性回归)

使用方式:
    from model_code.model_factory import ModelFactory
    model = ModelFactory.build("lightgbm", task_type="binary")
"""

from typing import Dict, List, Any, Optional

import numpy as np

from core.automl.problem_spec import ALGORITHMS, ProblemType, normalize_problem_type


# 算法显示名映射
DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "随机森林",
    "logistic_regression": "逻辑回归",
    "linear_regression": "线性回归",
}

# CatBoost 中与文件系统写入相关的训练参数。训练参数来自 API 调用方，
# 若允许这些键，调用方可让 fit() 在任意目录创建文件（train_dir 会被
# 自动创建并写入训练日志），构成目录污染/受限文件写入原语。
FORBIDDEN_FILESYSTEM_TRAINING_PARAMS = {
    "train_dir",
    "allow_writing_files",
    "snapshot_file",
}


class ModelFactory:
    """静态方法算法工厂"""

    @staticmethod
    def get_supported_algorithms() -> List[str]:
        """返回所有支持的算法标识"""
        return [
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "logistic_regression",
            "linear_regression",
        ]

    @staticmethod
    def get_display_name(algorithm: str) -> str:
        """返回算法的人类可读名称"""
        return DISPLAY_NAMES.get(algorithm, algorithm)

    @staticmethod
    def build(algorithm: str, task_type: str = "binary", params: Optional[Dict] = None) -> Any:
        """
        构建与任务类型匹配的 estimator 实例。

        Args:
            algorithm: 算法标识
            task_type: 规范任务类型或兼容别名 "binary" / "multiclass"
            params: 覆盖默认参数的自定义参数

        Returns:
            sklearn 兼容的 estimator 实例
        """
        params = dict(params or {})
        resolved = normalize_problem_type(task_type)
        if resolved == ProblemType.AUTO:
            raise ValueError("模型构建前必须解析明确的任务类型，不能使用 auto")

        builders = {
            "lightgbm": ModelFactory._build_lightgbm,
            "xgboost": ModelFactory._build_xgboost,
            "catboost": ModelFactory._build_catboost,
            "random_forest": ModelFactory._build_random_forest,
            "logistic_regression": ModelFactory._build_logistic_regression,
            "linear_regression": ModelFactory._build_linear_regression,
        }

        if algorithm not in builders:
            raise ValueError(
                f"不支持的算法：{algorithm}。"
                f"支持的算法: {list(builders.keys())}"
            )

        if algorithm not in ALGORITHMS[resolved]:
            raise ValueError(f"算法 {algorithm} 不适用于 {resolved.value}")

        return builders[algorithm](resolved.value, params)

    @staticmethod
    def get_param_grid(algorithm: str) -> Dict[str, Any]:
        """
        返回各算法的 RandomizedSearchCV 参数空间

        Args:
            algorithm: 算法标识

        Returns:
            参数字典 (值为 numpy array)
        """
        grids = {
            "lightgbm": {
                "n_estimators": np.arange(100, 501, 100),
                "learning_rate": np.linspace(0.01, 0.3, 10),
                "max_depth": np.arange(3, 10),
                "num_leaves": np.arange(20, 101, 20),
                "min_child_samples": np.arange(10, 51, 10),
                "subsample": np.linspace(0.7, 1.0, 4),
                "colsample_bytree": np.linspace(0.7, 1.0, 4),
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0, 0.01, 0.1, 1],
            },
            "xgboost": {
                "n_estimators": np.arange(100, 501, 100),
                "learning_rate": np.linspace(0.01, 0.3, 10),
                "max_depth": np.arange(3, 10),
                "min_child_weight": np.arange(1, 11, 2),
                "subsample": np.linspace(0.7, 1.0, 4),
                "colsample_bytree": np.linspace(0.7, 1.0, 4),
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0, 0.01, 0.1, 1],
            },
            "catboost": {
                "iterations": [100, 200, 300, 500],
                "learning_rate": np.linspace(0.01, 0.3, 10),
                "depth": np.arange(3, 10),
                "l2_leaf_reg": [1, 3, 5, 7, 10],
                "border_count": [32, 64, 128, 256],
                "subsample": np.linspace(0.7, 1.0, 4),
            },
            "random_forest": {
                "n_estimators": np.arange(100, 501, 100),
                "max_depth": np.arange(3, 20, 2),
                "min_samples_split": np.arange(2, 21, 2),
                "min_samples_leaf": np.arange(1, 11),
                "max_features": ["sqrt", "log2", None],
            },
            "logistic_regression": {
                "C": np.logspace(-3, 3, 7),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            },
            "linear_regression": {},
        }

        if algorithm not in grids:
            raise ValueError(f"不支持的算法：{algorithm}")

        return grids[algorithm]

    @staticmethod
    def get_default_params(algorithm: str, task_type: str = "binary") -> Dict[str, Any]:
        """
        返回各算法的默认参数

        Args:
            algorithm: 算法标识
            task_type: "binary" 或 "multiclass"

        Returns:
            默认参数字典
        """
        resolved = normalize_problem_type(task_type)
        if resolved == ProblemType.AUTO:
            raise ValueError("默认参数需要明确的任务类型，不能使用 auto")

        if resolved == ProblemType.REGRESSION:
            defaults = {
                "lightgbm": {
                    "objective": "regression",
                    "n_estimators": 200,
                    "verbose": -1,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "xgboost": {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "n_estimators": 200,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "catboost": {
                    "iterations": 200,
                    "verbose": False,
                    "random_state": 42,
                    "task_type": "CPU",
                },
                "random_forest": {
                    "n_estimators": 200,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "linear_regression": {},
            }
        else:
            is_binary = resolved == ProblemType.BINARY
            defaults = {
                "lightgbm": {
                    "objective": "binary" if is_binary else "multiclass",
                    "n_estimators": 200,
                    "verbose": -1,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "xgboost": {
                    "objective": "binary:logistic" if is_binary else "multi:softprob",
                    "eval_metric": "auc" if is_binary else "mlogloss",
                    "n_estimators": 200,
                    "n_jobs": -1,
                    "random_state": 42,
                    "use_label_encoder": False,
                },
                "catboost": {
                    "iterations": 200,
                    "verbose": False,
                    "random_state": 42,
                    "task_type": "CPU",
                },
                "random_forest": {
                    "n_estimators": 200,
                    "n_jobs": -1,
                    "random_state": 42,
                    "class_weight": "balanced",
                },
                "logistic_regression": {
                    "max_iter": 1000,
                    "random_state": 42,
                    "class_weight": "balanced",
                },
            }

        if algorithm not in defaults:
            raise ValueError(f"不支持的算法：{algorithm}")

        return defaults[algorithm]

    # ---- 内部构建方法 ----

    @staticmethod
    def _build_lightgbm(task_type: str, params: Dict) -> Any:
        import lightgbm as lgb
        defaults = ModelFactory.get_default_params("lightgbm", task_type)
        if task_type == ProblemType.MULTICLASS.value:
            # LightGBM 参数名为 num_class（num_classes 仅为别名）；仅在显式提供时注入，
            # 否则由 LGBMClassifier.fit 根据 y 自动推断，避免传入 None。
            num_class = params.pop("num_class", None) or params.pop("num_classes", None)
            if num_class is not None:
                defaults["num_class"] = num_class
        defaults.update(params)
        if task_type == ProblemType.REGRESSION.value:
            return lgb.LGBMRegressor(**defaults)
        return lgb.LGBMClassifier(**defaults)

    @staticmethod
    def _build_xgboost(task_type: str, params: Dict) -> Any:
        import xgboost as xgb
        defaults = ModelFactory.get_default_params("xgboost", task_type)
        if task_type == ProblemType.MULTICLASS.value:
            num_class = params.pop("num_class", None)
            if num_class is not None:
                defaults["num_class"] = num_class
        defaults.update(params)
        if task_type == ProblemType.REGRESSION.value:
            return xgb.XGBRegressor(**defaults)
        return xgb.XGBClassifier(**defaults)

    @staticmethod
    def _build_catboost(task_type: str, params: Dict) -> Any:
        forbidden = FORBIDDEN_FILESYSTEM_TRAINING_PARAMS.intersection(params)
        if forbidden:
            raise ValueError(f"禁止使用文件系统相关的训练参数：{', '.join(sorted(forbidden))}")
        defaults = ModelFactory.get_default_params("catboost", task_type)
        if task_type == ProblemType.MULTICLASS.value:
            classes_count = params.pop("classes_count", None)
            if classes_count is not None:
                defaults["classes_count"] = classes_count
        if "auto_class_weights" in params:
            defaults["auto_class_weights"] = params.pop("auto_class_weights")
        defaults.update(params)
        if task_type == ProblemType.REGRESSION.value:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(**defaults)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**defaults)

    @staticmethod
    def _build_random_forest(task_type: str, params: Dict) -> Any:
        defaults = ModelFactory.get_default_params("random_forest", task_type)
        defaults.update(params)
        if task_type == ProblemType.REGRESSION.value:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**defaults)
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**defaults)

    @staticmethod
    def _build_logistic_regression(task_type: str, params: Dict) -> Any:
        from sklearn.linear_model import LogisticRegression
        defaults = ModelFactory.get_default_params("logistic_regression", task_type)
        defaults.update(params)
        return LogisticRegression(**defaults)

    @staticmethod
    def _build_linear_regression(task_type: str, params: Dict) -> Any:
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**params)
