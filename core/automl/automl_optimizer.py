"""
AutoML Optimizer - 自动机器学习优化器

Phase 3 核心组件：基于 Optuna 的贝叶斯优化框架，执行 AutoML Agent 生成的配置。

核心设计理念:
- Agent 负责"智能决策" (选模型 + 定搜索空间)
- Python 负责"执行寻优" (Optuna 贝叶斯优化)
- 工厂模式动态组装模型
- 交叉验证保证稳定性
"""

from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np

from core.automl.problem_spec import METRIC_DIRECTION

logger = logging.getLogger(__name__)


class AutoMLOptimizer:
    """
    AutoML 优化器
    
    基于 Optuna 的贝叶斯优化框架，支持多模型自动选择和超参数寻优。
    
    核心功能:
    1. 动态解析 Agent 生成的搜索空间配置
    2. 工厂模式实例化模型 (LightGBM/XGBoost/CatBoost/RF)
    3. 交叉验证评估 (StratifiedKFold)
    4. 贝叶斯优化 (TPE 算法)
    5. 最佳模型训练与保存
    """
    
    def __init__(self, X, y, agent_config: Dict[str, Any],
                 cv_splits: int = 5, n_jobs: int = -1):
        """
        初始化优化器

        Args:
            X: 特征矩阵 (pandas DataFrame 或 numpy array)
            y: 目标变量 (pandas Series 或 numpy array)
            agent_config: AutoML Agent 生成的配置
                {
                    "task_type": "binary_classification",
                    "optimization_metric": "roc_auc",
                    "candidate_models": ["lightgbm", "catboost"],
                    "n_trials": 30,
                    "search_spaces": {...}
                }
            cv_splits: 交叉验证折数
            n_jobs: 并行作业数 (-1 表示使用所有 CPU)
        """
        import pandas as pd

        # 保存特征名称 (用于特征重要性输出)
        if hasattr(X, 'columns') and hasattr(X, 'values'):
            self.feature_names = list(X.columns)
            self.X = X.values
        else:
            self.X = X
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.y = y if not hasattr(y, 'values') else y.values
        self.config = agent_config
        self.cv_splits = cv_splits
        self.n_jobs = n_jobs

        # 优化指标
        self.metric = agent_config.get("optimization_metric", "roc_auc")
        self.task_type = agent_config.get("task_type", "binary_classification")

        # 最佳结果
        self.best_model = None
        self.best_study = None
        self.best_params = None
        self.best_score = None

        logger.info(f"AutoMLOptimizer 初始化完成：{len(self.X):,} 样本 × {len(self.feature_names)} 特征")
        logger.info(f"优化目标：{self.metric}, 候选模型：{agent_config.get('candidate_models', [])}")
    
    def _build_model(self, trial, model_name: str):
        """
        根据 Agent 给定的搜索空间，动态组装模型参数
        
        Args:
            trial: Optuna Trial 对象
            model_name: 模型名称
            
        Returns:
            模型实例
        """
        space = self.config["search_spaces"].get(model_name, {})
        params = {}
        
        # 动态解析 Optuna 的 suggest 方法
        for param_name, rules in space.items():
            param_type = rules.get("type", "int")
            
            try:
                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        rules.get("low", 1), 
                        rules.get("high", 100)
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        rules.get("low", 0.0),
                        rules.get("high", 1.0)
                    )
                elif param_type == "loguniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        rules.get("low", 0.01),
                        rules.get("high", 1.0),
                        log=True
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        rules.get("choices", [])
                    )
            except Exception as e:
                logger.warning(f"参数 {param_name} 解析失败：{e}")
        
        # 委托统一模型工厂，确保分类/回归能力校验和默认参数只有一个来源。
        try:
            from model_code.model_factory import ModelFactory

            safe_params = dict(params)
            if model_name in {"lightgbm", "xgboost", "random_forest"}:
                safe_params["n_jobs"] = 1
            if model_name == "catboost":
                safe_params.update({"thread_count": 1, "verbose": False})
            return ModelFactory.build(model_name, task_type=self.task_type, params=safe_params)
        except ImportError as e:
            logger.error(f"模型 {model_name} 依赖缺失：{e}")
            raise
    
    def objective(self, trial):
        """
        Optuna 的核心优化目标函数
        
        Args:
            trial: Optuna Trial 对象
            
        Returns:
            float: 交叉验证得分
        """
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        
        try:
            # 1. 动态选择本次 Trial 要尝试的模型类型
            candidate_models = self.config.get("candidate_models", ["lightgbm"])
            
            # 如果只有一个候选模型，直接使用该模型
            if len(candidate_models) == 1:
                model_name = candidate_models[0]
            else:
                model_name = trial.suggest_categorical("model_type", candidate_models)
            
            # 2. 构建该模型及其超参数
            model = self._build_model(trial, model_name)
            
            # 3. 交叉验证 (防止过拟合，保证稳定性)
            if self.task_type in ["binary_classification", "multiclass_classification"]:
                min_class_count = int(np.min(np.unique(self.y, return_counts=True)[1]))
                safe_splits = min(self.cv_splits, min_class_count)
                if safe_splits < 2:
                    raise ValueError("每个类别至少需要 2 条记录才能执行交叉验证")
                cv = StratifiedKFold(
                    n_splits=safe_splits,
                    shuffle=True,
                    random_state=42
                )
            else:
                cv = KFold(
                    n_splits=self.cv_splits,
                    shuffle=True,
                    random_state=42
                )
            
            # 4. 选择评分指标
            scoring = self._get_scoring_metric()
            
            # 5. 执行交叉验证
            scores = cross_val_score(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                error_score="raise",
            )

            mean_score = float(np.mean(scores))
            return -mean_score if self.metric in {"rmse", "mae"} else mean_score
            
        except Exception as e:
            logger.warning(f"Trial 失败：{e}")
            trial.set_user_attr("failure", str(e))
            import optuna
            raise optuna.TrialPruned(str(e)) from e
    
    def _get_scoring_metric(self) -> str:
        """获取 sklearn 评分指标名称（多分类 roc_auc 自动切换为 OVR 加权）"""
        if self.metric == "roc_auc" and self.task_type == "multiclass_classification":
            return "roc_auc_ovr_weighted"
        if self.task_type == "multiclass_classification" and self.metric in {"f1", "precision", "recall"}:
            return f"{self.metric}_weighted"
        metric_mapping = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "rmse": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        }
        return metric_mapping.get(self.metric, "roc_auc")
    
    def optimize(self, n_trials: Optional[int] = None,
                 timeout: Optional[int] = None,
                 show_progress: bool = True,
                 callbacks: Optional[List] = None) -> Tuple:
        """
        启动贝叶斯搜索

        Args:
            n_trials: Trial 次数 (默认使用配置中的值)
            timeout: 超时时间 (秒)
            show_progress: 是否显示进度条
            callbacks: 额外的 Optuna 回调（如 SSE 实时进度推送），逐个 Trial 触发

        Returns:
            Tuple: (study, best_model)
        """
        import optuna

        # 设置 Trial 次数
        if n_trials is None:
            n_trials = self.config.get("n_trials", 30)
        
        # 使用可读分数：误差为正数并最小化，收益指标最大化。
        direction = METRIC_DIRECTION[self.metric]
        
        # 创建 Study
        self.best_study = optuna.create_study(
            direction=direction,
            study_name="automl_agent_study",
            sampler=optuna.samplers.TPESampler(seed=42)  # TPE 贝叶斯优化
        )
        
        # 显示进度
        if show_progress:
            from tqdm import tqdm
            import sys
            
            class TQDMProgressCallback:
                def __init__(self, total):
                    self.pbar = tqdm(total=total, file=sys.stdout, desc="Optuna 优化中")
                
                def __call__(self, study, trial):
                    self.pbar.update(1)
            
            callback = TQDMProgressCallback(n_trials)
        else:
            callback = None

        # 合并进度条回调与调用方回调（如 SSE 实时推送）
        optuna_callbacks = ([callback] if callback else []) + list(callbacks or [])

        # 启动寻优
        logger.info(f"开始贝叶斯优化，{n_trials} 次 Trial...")

        self.best_study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=optuna_callbacks,
            show_progress_bar=False
        )

        successful_trials = [
            trial for trial in self.best_study.trials
            if trial.value is not None and not trial.user_attrs.get("failure")
        ]
        if not successful_trials:
            failures = [trial.user_attrs.get("failure") for trial in self.best_study.trials if trial.user_attrs.get("failure")]
            detail = failures[0] if failures else "没有完成的 Trial"
            raise RuntimeError(f"所有 AutoML Trial 均失败：{detail}")
        
        # 输出最佳结果
        logger.info(f"最佳 Trial 分数：{self.best_study.best_value:.4f}")
        logger.info(f"最佳模型与参数：{self.best_study.best_params}")
        
        # 保存最佳参数
        self.best_params = dict(self.best_study.best_params)

        # 用全量训练集重新训练最佳模型
        best_model_name = self.best_params.get("model_type")

        # 如果没有 model_type，尝试从候选模型中获取
        if best_model_name is None:
            candidate_models = self.config.get("candidate_models", [])
            if len(candidate_models) == 1:
                best_model_name = candidate_models[0]
            elif len(candidate_models) > 1:
                # 从 trial 中获取最佳模型类型
                if hasattr(self.best_study.best_trial, 'params') and 'model_type' in self.best_study.best_trial.params:
                    best_model_name = self.best_study.best_trial.params['model_type']
                else:
                    # 默认使用第一个模型
                    best_model_name = candidate_models[0]

        if best_model_name:
            logger.info(f"使用最佳参数重新训练 {best_model_name}...")
            try:
                self.best_model = self._build_model(self.best_study.best_trial, best_model_name)
                self.best_model.fit(self.X, self.y)
                self.best_score = self.best_study.best_value
                logger.info(f"模型训练完成，最佳得分：{self.best_score:.4f}")
            except Exception as e:
                logger.error(f"模型训练失败：{e}")
                self.best_model = None
                raise RuntimeError(f"最佳模型全量训练失败：{e}") from e
        else:
            logger.warning("无法确定最佳模型类型")

        return self.best_study, self.best_model
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        获取特征重要性

        Args:
            top_n: 返回 Top-N 个特征

        Returns:
            Dict: {特征名：重要性}
        """
        if self.best_model is None:
            logger.warning("模型尚未训练，无法获取特征重要性")
            return {}

        try:
            # 获取重要性
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            else:
                logger.warning("模型不支持特征重要性")
                return {}

            # 排序
            indices = np.argsort(importances)[::-1][:top_n]

            return {
                self.feature_names[i]: float(importances[i])
                for i in indices
            }

        except Exception as e:
            logger.error(f"获取特征重要性失败：{e}")
            return {}
    
    def save_model(self, path: str):
        """
        保存最佳模型
        
        Args:
            path: 保存路径
        """
        import joblib
        
        if self.best_model is None:
            logger.warning("没有可保存的模型")
            return
        
        joblib.dump({
            "model": self.best_model,
            "params": self.best_params,
            "config": self.config,
            "best_score": self.best_score
        }, path)
        
        logger.info(f"模型已保存：{path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        import joblib
        
        data = joblib.load(path)
        self.best_model = data["model"]
        self.best_params = data["params"]
        self.config = data["config"]
        self.best_score = data["best_score"]
        
        logger.info(f"模型已加载：{path}")


def plot_optimization_results(study, save_path: Optional[str] = None):
    """
    绘制 Optuna 优化结果可视化
    
    Args:
        study: Optuna Study 对象
        save_path: 保存路径 (可选)
        
    Returns:
        dict: Plotly 图表字典
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
            plot_parallel_coordinate
        )
        
        charts = {
            "history": plot_optimization_history(study),
            "importances": plot_param_importances(study),
            "slice": plot_slice(study),
            "parallel": plot_parallel_coordinate(study)
        }
        
        if save_path:
            import json
            for name, fig in charts.items():
                fig.write_html(f"{save_path}_{name}.html")
        
        return charts
        
    except ImportError:
        logger.warning("optuna.visualization 未安装，无法生成可视化图表")
        return {}
