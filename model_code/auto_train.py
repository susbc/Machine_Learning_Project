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
import lightgbm as lgb

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
        model_dir: Optional[Path] = None
    ):
        """
        初始化自动训练器
        
        Args:
            base_config: 基础配置字典，包含:
                - X: 特征 DataFrame
                - y: 标签 Series
                - model_params: 模型参数
                - test_size: 测试集比例
                - random_state: 随机种子
            max_iterations: 最大迭代次数
            early_stopping_rounds: 提前停止轮数
            min_improvement: 最小改进阈值
            model_dir: 模型保存目录
        """
        self.base_config = base_config
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.min_improvement = min_improvement
        self.model_dir = Path(model_dir) if model_dir else Path("model_code/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 迭代历史
        self.history: List[Dict[str, Any]] = []
        self.best_iteration: int = 0
        self.best_score: float = 0
        self.best_model = None
        self.best_params: Dict[str, Any] = {}
        
        # 中断标志
        self.interrupted = False
    
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
        
        # 获取数据
        X = self.base_config['X']
        y = self.base_config['y']
        test_size = self.base_config.get('test_size', 0.3)
        random_state = self.base_config.get('random_state', 42) + iteration
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 使用当前参数或默认参数
        if params is None:
            params = self.base_config.get('model_params', {})
        
        # 训练模型
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            verbose=-1,
            random_state=random_state,
            n_jobs=-1,
            **params
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
        
        X = self.base_config['X']
        y = self.base_config['y']
        random_state = self.base_config.get('random_state', 42)
        
        # 参数空间
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'num_leaves': [20, 30, 40, 50, 60, 70],
            'min_child_samples': [10, 20, 30, 40, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1]
        }
        
        # 基础模型
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            verbose=-1,
            random_state=random_state,
            n_jobs=-1
        )
        
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
        
        random_search.fit(X, y)
        
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
        
        X = self.base_config['X']
        y = self.base_config['y']
        feature_names = X.columns.tolist()
        
        # 使用 LightGBM 进行特征选择
        model = lgb.LGBMClassifier(
            objective='binary',
            verbose=-1,
            random_state=42,
            n_jobs=-1,
            **self.base_config.get('model_params', {})
        )
        
        model.fit(X, y)
        
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
            self.base_config['X'] = self.base_config['X'][selected_features]
        else:
            logger.info("跳过特征选择")
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


def run_auto_training(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: Dict[str, Any],
    max_iterations: int = 10,
    model_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    便捷函数：运行自动训练
    
    Args:
        X: 特征 DataFrame
        y: 标签 Series
        model_params: 模型参数
        max_iterations: 最大迭代次数
        model_dir: 模型保存目录
    
    Returns:
        训练总结
    """
    config = {
        'X': X,
        'y': y,
        'model_params': model_params,
        'test_size': 0.3,
        'random_state': 42
    }
    
    trainer = AutoTrainer(
        base_config=config,
        max_iterations=max_iterations,
        model_dir=model_dir
    )
    
    return trainer.run()
