"""
自动迭代训练模块

提供自动化模型迭代优化功能，包括:
- 自动超参数调整
- 自动特征选择
- 自动采样比例调整
- 迭代进度跟踪
- 最佳模型自动保存
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.feature_selection import SelectFromModel
from model_code.model_factory import ModelFactory
from model_code.utils import ConfigDrivenPreprocessor, smart_sampling

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    自动迭代训练器
    
    实现自动化模型训练和优化流程
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        max_iterations: int = 10,
        early_stopping_rounds: int = 3,
        min_improvement: float = 0.001,
        model_dir: Optional[Path] = None,
        algorithm: str = "auto",
        preprocessor: Optional[ConfigDrivenPreprocessor] = None,
        enable_sampling: bool = False,
        sampling_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化自动训练器

        Args:
            base_config: 基础配置字典，包含:
                - X_train: 训练集特征 DataFrame
                - y_train: 训练集标签 Series
                - X_test: 测试集特征 DataFrame (可选)
                - y_test: 测试集标签 Series (可选)
                - model_params: 模型参数
                - test_size: 测试集比例 (仅当未提供 X_test 时用于回退划分)
                - random_state: 随机种子
            max_iterations: 最大迭代次数
            early_stopping_rounds: 提前停止轮数
            min_improvement: 最小改进阈值
            model_dir: 模型保存目录
            algorithm: 算法标识 (默认 "auto" 回退到 lightgbm)
            preprocessor: 预处理管道 (可选)
            enable_sampling: 是否启用智能抽样
            sampling_config: 抽样配置字典
        """
        self.base_config = base_config
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.min_improvement = min_improvement
        self.model_dir = Path(model_dir) if model_dir else Path("model_code/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm  # 算法标识，默认 "auto" 回退到 lightgbm

        # 预处理管道
        self.preprocessor = preprocessor
        self.enable_sampling = enable_sampling
        self.sampling_config = sampling_config or {"negative_ratio": 0.1, "random_state": 42}

        # 迭代历史
        self.history: List[Dict[str, Any]] = []
        self.best_iteration: int = 0
        self.best_score: float = 0
        self.best_model = None
        self.best_params: Dict[str, Any] = {}

        # 中断标志
        self.interrupted = False

    def _get_data_split(self) -> Tuple[Any, Any, Any, Any]:
        """
        获取训练/测试数据 - 优先使用预划分数据，防止数据穿越
        支持预处理管道和智能抽样

        Returns:
            X_train, X_test, y_train, y_test
        """
        random_state = self.base_config.get('random_state', 42)
        test_size = self.base_config.get('test_size', 0.3)

        # 优先使用预划分的数据 (防止数据穿越)
        if 'X_train' in self.base_config and 'y_train' in self.base_config:
            X_train = self.base_config['X_train']
            y_train = self.base_config['y_train']

            # 如果也提供了预划分测试集，使用它
            if 'X_test' in self.base_config and 'y_test' in self.base_config:
                X_test = self.base_config['X_test']
                y_test = self.base_config['y_test']
            else:
                # 没有预划分测试集，从训练集再分一份 (仅用于内部验证)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
                )
        else:
            # 回退：从原始数据划分 (存在数据穿越风险，仅用于独立使用场景)
            X = self.base_config['X']
            y = self.base_config['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        # ---- 数据预处理 (仅在训练集上 fit，防止数据穿越) ----
        if self.preprocessor is not None:
            if self.preprocessor.feature_names_in_ is None:
                self.preprocessor.fit(X_train)
            X_train = self.preprocessor.transform(X_train)
            X_test = self.preprocessor.transform(X_test)
            logger.info(f"数据预处理完成 (缺失值={self.preprocessor.enable_missing}, "
                        f"异常值={self.preprocessor.enable_outlier}, "
                        f"编码={self.preprocessor.enable_encoding}, "
                        f"标准化={self.preprocessor.enable_standardization})")

        # ---- 智能抽样 (可选) ----
        if self.enable_sampling:
            train_df = X_train.copy()
            train_df['_label'] = y_train
            neg_ratio = self.sampling_config.get("negative_ratio", 0.1)
            rs = self.sampling_config.get("random_state", random_state)
            logger.info(f"执行负样本欠采样 (负样本比例={neg_ratio:.2f})...")
            sampled_df = smart_sampling(train_df, label_col='_label',
                                        negative_ratio=neg_ratio, random_state=rs)
            y_train = sampled_df['_label']
            X_train = sampled_df.drop(columns=['_label'])

        return X_train, X_test, y_train, y_test
    
    def train_iteration(self, iteration: int, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        单次迭代训练
        
        Args:
            iteration: 迭代次数
            params: 本次迭代的参数 (可选)
        
        Returns:
            迭代结果字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"迭代 {iteration}/{self.max_iterations}")
        logger.info(f"{'='*60}")
        
        # 获取数据 (优先使用预划分数据，防止数据穿越)
        X_train, X_test, y_train, y_test = self._get_data_split()

        # 使用当前参数或默认参数
        model_params = self.base_config.get('model_params', {})
        if params is None:
            params = model_params

        # 训练模型
        algo = self.algorithm if self.algorithm != "auto" else "lightgbm"
        model = ModelFactory.build(
            algorithm=algo,
            task_type="binary",
            params=params,
        )
        
        model.fit(X_train, y_train)
        
        # 评估
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 记录结果
        iteration_result = {
            'iteration': iteration,
            'params': params.copy(),
            'auc': auc,
            'f1': f1,
            'model': model,
            'feature_importances': model.feature_importances_,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(iteration_result)
        
        logger.info(f"迭代 {iteration} 完成:")
        logger.info(f"  AUC: {auc:.6f}")
        logger.info(f"  F1:  {f1:.6f}")
        
        # 更新最佳模型
        if auc > self.best_score:
            self.best_score = auc
            self.best_iteration = iteration
            self.best_model = model
            self.best_params = params.copy()
            
            # 保存最佳模型
            model_path = self.model_dir / f"auto_best_model_iter{iteration}.joblib"
            joblib.dump(model, model_path)
            
            logger.info(f"✨ 发现更好的模型！AUC: {auc:.6f}")
            logger.info(f"   已保存：{model_path}")
        
        return iteration_result
    
    def optimize_hyperparameters(self, n_iter: int = 10) -> Dict[str, Any]:
        """
        自动超参数优化
        
        Args:
            n_iter: 随机搜索迭代次数
        
        Returns:
            最佳参数
        """
        logger.info("\n" + "="*60)
        logger.info("开始超参数优化")
        logger.info("="*60)
        
        X_train, _, y_train, _ = self._get_data_split()
        random_state = self.base_config.get('random_state', 42)
        
        algo = self.algorithm if self.algorithm != "auto" else "lightgbm"
        param_distributions = ModelFactory.get_param_grid(algo)
        
        # 基础模型
        model = ModelFactory.build(algorithm=algo, task_type="binary")
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0,
            random_state=random_state
        )
        
        random_search.fit(X_train, y_train)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        logger.info(f"超参数优化完成!")
        logger.info(f"最佳 AUC: {best_score:.6f}")
        logger.info(f"最佳参数:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        self.best_params = best_params
        
        return best_params
    
    def feature_selection(
        self,
        threshold: str = 'median',
        max_features: Optional[int] = None
    ) -> List[str]:
        """
        自动特征选择
        
        Args:
            threshold: 选择阈值 ('mean', 'median', 或数值)
            max_features: 最大特征数
        
        Returns:
            选中的特征名列表
        """
        logger.info("\n" + "="*60)
        logger.info("开始特征选择")
        logger.info("="*60)
        
        X_train, _, y_train, _ = self._get_data_split()
        feature_names = X_train.columns.tolist()

        # 使用 ModelFactory 进行特征选择
        algo = self.algorithm if self.algorithm != "auto" else "lightgbm"
        model = ModelFactory.build(
            algorithm=algo,
            task_type="binary",
            params=self.base_config.get('model_params', {}),
        )

        model.fit(X_train, y_train)
        
        # 基于重要性选择特征
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        selected_mask = selector.get_support()
        
        selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
        
        if max_features and len(selected_features) > max_features:
            # 按重要性排序，取前 N 个
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'selected': selected_mask
            })
            importance_df = importance_df[importance_df['selected']]
            importance_df = importance_df.sort_values('importance', ascending=False)
            selected_features = importance_df.head(max_features)['feature'].tolist()
        
        logger.info(f"原始特征数：{len(feature_names)}")
        logger.info(f"选中特征数：{len(selected_features)}")
        logger.info(f"选中的特征：{selected_features}")
        
        return selected_features
    
    def adjust_sampling_ratio(
        self,
        current_ratio: float,
        direction: str = 'auto',
        step: float = 0.05
    ) -> float:
        """
        自动调整采样比例
        
        Args:
            current_ratio: 当前采样比例
            direction: 调整方向 ('increase', 'decrease', 'auto')
            step: 调整步长
        
        Returns:
            新的采样比例
        """
        if direction == 'auto':
            # 根据历史表现自动决定
            if len(self.history) >= 2:
                recent_auc = [h['auc'] for h in self.history[-2:]]
                if recent_auc[1] > recent_auc[0]:
                    direction = 'increase'  # 继续当前方向
                else:
                    direction = 'decrease'
            else:
                direction = 'increase'
        
        if direction == 'increase':
            new_ratio = min(1.0, current_ratio + step)
        else:
            new_ratio = max(0.1, current_ratio - step)
        
        logger.info(f"调整采样比例：{current_ratio:.2f} -> {new_ratio:.2f}")
        
        return new_ratio
    
    def check_early_stopping(self) -> bool:
        """
        检查是否应该提前停止
        
        Returns:
            是否应该停止
        """
        if len(self.history) < self.early_stopping_rounds:
            return False
        
        # 检查最近几轮是否有改进
        recent_scores = [h['auc'] for h in self.history[-self.early_stopping_rounds:]]
        max_recent = max(recent_scores)
        
        if max_recent - self.best_score < self.min_improvement:
            logger.info(f"提前停止：最近 {self.early_stopping_rounds} 轮无显著改进")
            return True
        
        return False
    
    def interrupt(self):
        """中断训练"""
        self.interrupted = True
        logger.info("训练已中断")

    @classmethod
    def from_file(
        cls,
        file_path: str,
        label_col: str,
        field_config: Optional[pd.DataFrame] = None,
        key_col: Optional[str] = None,
        algorithm: str = "auto",
        enable_preprocessing: bool = True,
        enable_sampling: bool = False,
        sampling_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "AutoTrainer":
        """
        从数据文件创建 AutoTrainer (独立使用模式)

        支持自动读取 CSV/Excel 文件，自动生成字段配置。

        Args:
            file_path: CSV 或 Excel 文件路径
            label_col: 目标变量列名
            field_config: 可选，字段配置 DataFrame。如果为 None 则自动生成。
            key_col: 可选，ID/键列名
            algorithm: 算法标识
            enable_preprocessing: 是否启用预处理管道
            enable_sampling: 是否启用负样本欠采样
            sampling_config: 采样配置
            **kwargs: 额外的 AutoTrainer 构造函数参数

        Returns:
            AutoTrainer 实例
        """
        from web_app.utils.file_handler import read_file_from_path

        df = read_file_from_path(file_path)
        if df is None:
            raise ValueError(f"无法读取文件：{file_path}")

        X = df.drop(columns=[label_col])
        y = df[label_col]

        preprocessor = None
        if enable_preprocessing:
            if field_config is None:
                field_config = auto_generate_field_config(df, label_col, key_col)
            preprocessor = ConfigDrivenPreprocessor(field_config)

        config: Dict[str, Any] = {
            'X': X,
            'y': y,
            'test_size': 0.3,
            'random_state': 42,
        }

        return cls(
            base_config=config,
            algorithm=algorithm,
            preprocessor=preprocessor,
            enable_sampling=enable_sampling,
            sampling_config=sampling_config,
            **kwargs,
        )
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整迭代流程
        
        Returns:
            训练总结
        """
        logger.info("\n" + "="*60)
        logger.info("开始自动迭代训练")
        logger.info(f"最大迭代次数：{self.max_iterations}")
        logger.info(f"{'='*60}")
        
        # 第 1 步：超参数优化
        logger.info("\n[步骤 1/4] 超参数优化")
        if not self.base_config.get('skip_hyperopt', False):
            self.optimize_hyperparameters(n_iter=10)
        else:
            logger.info("跳过超参数优化")
        
        # 第 2 步：特征选择
        logger.info("\n[步骤 2/4] 特征选择")
        if not self.base_config.get('skip_feature_selection', False):
            selected_features = self.feature_selection(max_features=50)
            # 更新所有数据集的特征 (支持预划分数据)
            if 'X_train' in self.base_config:
                self.base_config['X_train'] = self.base_config['X_train'][selected_features]
                if 'X_test' in self.base_config:
                    self.base_config['X_test'] = self.base_config['X_test'][selected_features]
            else:
                self.base_config['X'] = self.base_config['X'][selected_features]
        else:
            logger.info("跳过特征选择")
            if 'X_train' in self.base_config:
                selected_features = self.base_config['X_train'].columns.tolist()
            else:
                selected_features = self.base_config['X'].columns.tolist()
        
        # 第 3 步：迭代训练
        logger.info("\n[步骤 3/4] 迭代训练")
        
        for iteration in range(1, self.max_iterations + 1):
            if self.interrupted:
                logger.info("训练被用户中断")
                break
            
            # 训练
            self.train_iteration(iteration, self.best_params.copy())
            
            # 检查提前停止
            if self.check_early_stopping():
                logger.info("触发提前停止条件")
                break
        
        # 第 4 步：保存结果
        logger.info("\n[步骤 4/4] 保存结果")
        
        # 保存训练历史
        history_path = self.model_dir / f"auto_train_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        train_summary = {
            'history': self.history,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'best_params': self.best_params,
            'selected_features': selected_features,
            'total_iterations': len(self.history),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(train_summary, history_path)
        
        logger.info(f"\n自动训练完成!")
        logger.info(f"最佳迭代：{self.best_iteration}")
        logger.info(f"最佳 AUC: {self.best_score:.6f}")
        logger.info(f"训练历史已保存：{history_path}")
        
        return train_summary
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        获取迭代历史 DataFrame
        
        Returns:
            历史 DataFrame
        """
        if not self.history:
            return pd.DataFrame()
        
        data = []
        for h in self.history:
            data.append({
                'iteration': h['iteration'],
                'auc': h['auc'],
                'f1': h['f1'],
                'timestamp': h['timestamp'],
                **{f"param_{k}": v for k, v in h['params'].items()}
            })
        
        return pd.DataFrame(data)
    
    def plot_history(self):
        """
        绘制训练历史图表
        
        Returns:
            Plotly Figure
        """
        import plotly.graph_objects as go
        
        df = self.get_history_dataframe()
        
        if df.empty:
            return go.Figure().add_annotation(text="无训练历史数据", showarrow=False)
        
        fig = go.Figure()
        
        # AUC 曲线
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['auc'],
            mode='lines+markers',
            name='ROC-AUC',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # F1 曲线
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['f1'],
            mode='lines+markers',
            name='F1 分数',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        # 最佳线
        fig.add_hline(
            y=self.best_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"最佳 AUC: {self.best_score:.4f}"
        )
        
        fig.update_layout(
            title='自动迭代训练历史',
            xaxis_title='迭代次数',
            yaxis_title='分数',
            height=500,
            showlegend=True
        )
        
        return fig


def auto_generate_field_config(
    df: pd.DataFrame,
    label_col: str,
    key_col: Optional[str] = None
) -> pd.DataFrame:
    """
    根据 DataFrame 自动生成字段配置

    使用 infer_field_type 自动推断字段类型，并设置合理的默认处理规则。

    Args:
        df: 输入 DataFrame
        label_col: 目标变量列名
        key_col: 可选，ID/键列名

    Returns:
        配置 DataFrame，包含 field_name, field_type, outlier_value,
        missing_value, ecode_type, standardize_type 列
    """
    from web_app.utils.file_handler import infer_field_type

    config_data = []

    for col in df.columns:
        if col == label_col:
            field_type = "label"
        elif col == key_col:
            field_type = "key"
        else:
            field_type = infer_field_type(df[col])

        # 缺失值处理默认规则
        missing_value = np.nan
        if df[col].isna().sum() > 0:
            if field_type == 'num':
                missing_value = 'mean'
            else:
                missing_value = 'mode'

        config_data.append({
            "field_name": col,
            "field_type": field_type,
            "outlier_value": np.nan,
            "missing_value": missing_value,
            "ecode_type": "one-hot" if field_type == "str" else np.nan,
            "standardize_type": "z-score" if field_type == "num" else np.nan,
        })

    return pd.DataFrame(config_data)


def run_auto_training(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: Dict[str, Any],
    max_iterations: int = 10,
    model_dir: Optional[Path] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    algorithm: str = "auto",
    preprocessor: Optional[ConfigDrivenPreprocessor] = None,
    enable_sampling: bool = False,
    sampling_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    便捷函数：运行自动训练

    Args:
        X: 特征 DataFrame (回退用，优先使用 X_train)
        y: 标签 Series (回退用，优先使用 y_train)
        model_params: 模型参数
        max_iterations: 最大迭代次数
        model_dir: 模型保存目录
        X_train: 训练集特征 (可选，推荐提供以防止数据穿越)
        y_train: 训练集标签 (可选，推荐提供以防止数据穿越)
        X_test: 测试集特征 (可选)
        y_test: 测试集标签 (可选)
        algorithm: 算法标识
        preprocessor: 预处理管道
        enable_sampling: 是否启用智能抽样
        sampling_config: 采样配置

    Returns:
        训练总结
    """
    config: Dict[str, Any] = {
        'model_params': model_params,
        'test_size': 0.3,
        'random_state': 42
    }

    if X_train is not None and y_train is not None:
        config['X_train'] = X_train
        config['y_train'] = y_train
        if X_test is not None and y_test is not None:
            config['X_test'] = X_test
            config['y_test'] = y_test
    else:
        config['X'] = X
        config['y'] = y
    
    trainer = AutoTrainer(
        base_config=config,
        max_iterations=max_iterations,
        model_dir=model_dir,
        algorithm=algorithm,
        preprocessor=preprocessor,
        enable_sampling=enable_sampling,
        sampling_config=sampling_config,
    )
    
    return trainer.run()
