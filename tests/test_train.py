"""
测试训练函数 (model_code/train.py)

测试覆盖：
- LightGBM 训练函数
- 最优阈值查找函数
- 模型保存和加载
- 训练结果验证
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# 导入被测试模块
from model_code.train import (
    lgbm_train_with_random_search,
    find_optimal_threshold,
)


class TestLGBMTrainWithRandomSearch:
    """测试 LightGBM 训练函数"""
    
    def test_train_basic(self, sample_data, temp_model_dir, temp_output_dir):
        """测试基本训练功能"""
        data_df, field_config = sample_data
        
        # 准备特征和标签
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        # 执行训练 (使用较少的迭代次数以加快测试)
        result = lgbm_train_with_random_search(
            X=X,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            test_size=0.3,
            random_state=42,
            n_iter=2,  # 减少迭代次数
            cv=2  # 减少交叉验证折数
        )
        
        # 验证返回结果
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'test_auc' in result
        assert 'test_ap' in result
        assert 'best_threshold' in result
        assert 'best_params' in result
        
        # 验证 AUC 在合理范围内
        # fixture 标签与特征独立，AUC 应只校验合法范围，不能强行要求随机结果 >= 0.5
        assert 0.0 <= result['test_auc'] <= 1.0
        assert 0.0 <= result['test_ap'] <= 1.0
        
        # 验证文件保存
        model_path = temp_model_dir / "best_lgbm_model.joblib"
        assert model_path.exists()
        
        summary_path = temp_model_dir / "train_summary.joblib"
        assert summary_path.exists()
        
        metric_path = temp_output_dir / "metric_thresholds.csv"
        assert metric_path.exists()
    
    def test_train_with_custom_params(self, sample_data, temp_model_dir, temp_output_dir):
        """测试自定义参数训练"""
        data_df, field_config = sample_data
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        result = lgbm_train_with_random_search(
            X=X,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            test_size=0.2,  # 自定义测试集比例
            random_state=123,  # 自定义随机种子
            n_iter=1,
            cv=2
        )
        
        # 验证训练集和测试集大小
        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size
        
        assert result['train_size'] == train_size
        assert result['test_size'] == test_size
    
    def test_train_model_loading(self, sample_data, temp_model_dir, temp_output_dir):
        """测试训练后模型加载"""
        data_df, field_config = sample_data
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        # 训练模型
        lgbm_train_with_random_search(
            X=X,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 加载模型
        model_path = temp_model_dir / "best_lgbm_model.joblib"
        loaded_model = joblib.load(model_path)
        
        # 验证模型可以预测
        predictions = loaded_model.predict(X.head(10))
        predictions_proba = loaded_model.predict_proba(X.head(10))
        
        assert len(predictions) == 10
        assert predictions_proba.shape == (10, 2)
        assert np.all(np.isin(predictions, [0, 1]))
        assert np.all((predictions_proba >= 0).all(axis=1))
        assert np.all((predictions_proba <= 1).all(axis=1))
    
    def test_train_metric_thresholds(self, sample_data, temp_model_dir, temp_output_dir):
        """测试阈值指标生成"""
        data_df, field_config = sample_data
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        lgbm_train_with_random_search(
            X=X,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 加载阈值指标
        metric_path = temp_output_dir / "metric_thresholds.csv"
        metric_df = pd.read_csv(metric_path)
        
        # 验证指标 DataFrame
        assert 'threshold' in metric_df.columns
        assert 'precision' in metric_df.columns
        assert 'recall' in metric_df.columns
        assert 'f1' in metric_df.columns
        
        # 验证阈值范围
        assert metric_df['threshold'].min() >= 0.0
        assert metric_df['threshold'].max() <= 1.0
        
        # 验证指标范围
        assert metric_df['precision'].between(0, 1).all()
        assert metric_df['recall'].between(0, 1).all()
        assert metric_df['f1'].between(0, 1).all()


class TestFindOptimalThreshold:
    """测试最优阈值查找函数"""
    
    def test_find_threshold_f1(self, sample_data):
        """测试基于 F1 的最优阈值查找"""
        data_df, field_config = sample_data
        
        # 创建简单的预测概率
        np.random.seed(42)
        y_true = data_df['label'].values
        y_proba = np.random.uniform(0, 1, len(y_true))
        
        # 为了测试，让预测概率与真实标签有一定相关性
        y_proba = np.where(y_true == 1, 
                          np.clip(y_proba + 0.3, 0, 1),
                          np.clip(y_proba - 0.3, 0, 1))
        
        best_threshold, metrics = find_optimal_threshold(
            y_true=y_true,
            y_proba=y_proba,
            metric='f1',
            threshold_range=(0.1, 0.9, 0.1)
        )
        
        # 验证返回结果
        assert isinstance(best_threshold, float)
        assert 0.0 <= best_threshold <= 1.0
        
        assert isinstance(metrics, dict)
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # 验证指标范围
        assert 0.0 <= metrics['f1'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
    
    def test_find_threshold_precision(self, sample_data):
        """测试基于 Precision 的最优阈值查找"""
        data_df, field_config = sample_data
        
        np.random.seed(42)
        y_true = data_df['label'].values
        y_proba = np.random.uniform(0, 1, len(y_true))
        y_proba = np.where(y_true == 1, 
                          np.clip(y_proba + 0.3, 0, 1),
                          np.clip(y_proba - 0.3, 0, 1))
        
        best_threshold, metrics = find_optimal_threshold(
            y_true=y_true,
            y_proba=y_proba,
            metric='precision',
            threshold_range=(0.1, 0.9, 0.1)
        )
        
        assert isinstance(best_threshold, float)
        assert 'precision' in metrics
        assert 0.0 <= metrics['precision'] <= 1.0
    
    def test_find_threshold_recall(self, sample_data):
        """测试基于 Recall 的最优阈值查找"""
        data_df, field_config = sample_data
        
        np.random.seed(42)
        y_true = data_df['label'].values
        y_proba = np.random.uniform(0, 1, len(y_true))
        y_proba = np.where(y_true == 1, 
                          np.clip(y_proba + 0.3, 0, 1),
                          np.clip(y_proba - 0.3, 0, 1))
        
        best_threshold, metrics = find_optimal_threshold(
            y_true=y_true,
            y_proba=y_proba,
            metric='recall',
            threshold_range=(0.1, 0.9, 0.1)
        )
        
        assert isinstance(best_threshold, float)
        assert 'recall' in metrics
        assert 0.0 <= metrics['recall'] <= 1.0
    
    def test_find_threshold_invalid_metric(self, sample_data):
        """测试无效指标参数"""
        data_df, field_config = sample_data
        
        np.random.seed(42)
        y_true = data_df['label'].values
        y_proba = np.random.uniform(0, 1, len(y_true))
        
        # 验证抛出异常
        with pytest.raises(ValueError, match="不支持的指标"):
            find_optimal_threshold(
                y_true=y_true,
                y_proba=y_proba,
                metric='invalid_metric',
                threshold_range=(0.1, 0.9, 0.1)
            )


class TestTrainIntegration:
    """训练集成测试"""
    
    def test_full_training_pipeline(self, sample_data, temp_model_dir, temp_output_dir):
        """测试完整训练流程"""
        data_df, field_config = sample_data
        
        # 1. 使用预处理器处理数据
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'city', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        # 2. 训练模型
        result = lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=2,
            cv=2
        )
        
        # 3. 加载模型进行预测
        model_path = temp_model_dir / "best_lgbm_model.joblib"
        model = joblib.load(model_path)
        
        predictions = model.predict(X_processed)
        predictions_proba = model.predict_proba(X_processed)
        
        # 4. 验证结果
        assert len(predictions) == len(y)
        assert predictions_proba.shape[0] == len(y)
        assert predictions_proba.shape[1] == 2
        
        # 5. 计算验证 AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y, predictions_proba[:, 1])
        
        assert 0.5 <= val_auc <= 1.0
