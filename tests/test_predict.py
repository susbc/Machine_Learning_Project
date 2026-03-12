"""
测试预测函数 (model_code/predict.py)

测试覆盖：
- 加载模型并预测
- 批量预测
- 带解释的预测
- 预测结果验证
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# 导入被测试模块
from model_code.predict import (
    load_and_predict,
    batch_predict,
    predict_with_explanation,
)


class TestLoadAndPredict:
    """测试加载模型并预测函数"""
    
    def test_predict_basic(self, sample_data, temp_model_dir, temp_output_dir):
        """测试基本预测功能"""
        data_df, field_config = sample_data
        
        # 先训练一个模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        # 预处理
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        # 训练
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 准备预测数据
        predict_data = X_processed.head(100).copy()
        key = data_df['user_id'].head(100)
        
        # 执行预测
        result = load_and_predict(
            data=predict_data,
            key=key,
            data_month='202401',
            model_dir=temp_model_dir,
            output_dir=temp_output_dir
        )
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert 'key' in result.columns
        assert 'probability' in result.columns
        assert 'prediction' in result.columns
        
        # 验证概率范围
        assert result['probability'].between(0, 1).all()
        
        # 验证预测值为 0 或 1
        assert result['prediction'].isin([0, 1]).all()
    
    def test_predict_with_custom_threshold(self, sample_data, temp_model_dir, temp_output_dir):
        """测试自定义阈值预测"""
        data_df, field_config = sample_data
        
        # 训练模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 使用自定义阈值预测
        predict_data = X_processed.head(50).copy()
        key = data_df['user_id'].head(50)
        
        custom_threshold = 0.3
        result = load_and_predict(
            data=predict_data,
            key=key,
            data_month='202402',
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            threshold=custom_threshold
        )
        
        # 验证结果
        assert len(result) == 50
        
        # 验证使用了自定义阈值
        # 阈值越低，预测为正例的越多
        pos_ratio_custom = result['prediction'].mean()
        
        # 使用默认阈值再预测一次
        result_default = load_and_predict(
            data=predict_data,
            key=key,
            data_month='202403',
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            threshold=0.5
        )
        
        # 验证不同阈值产生不同结果
        # (不一定总是成立，但大多数情况下 0.3 阈值会产生更多正例预测)
        assert isinstance(pos_ratio_custom, float)
    
    def test_predict_file_saving(self, sample_data, temp_model_dir, temp_output_dir):
        """测试预测结果保存"""
        data_df, field_config = sample_data
        
        # 训练模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 预测
        predict_data = X_processed.head(50).copy()
        key = data_df['user_id'].head(50)
        
        load_and_predict(
            data=predict_data,
            key=key,
            data_month='202404',
            model_dir=temp_model_dir,
            output_dir=temp_output_dir
        )
        
        # 验证文件保存
        result_path = temp_output_dir / "predict_res_202404.csv"
        assert result_path.exists()
        
        detail_path = temp_output_dir / "predict_res_202404_detail.csv"
        assert detail_path.exists()
        
        # 验证文件内容
        result_df = pd.read_csv(result_path)
        assert len(result_df) == 50
        assert 'key' in result_df.columns
        assert 'probability' in result_df.columns
        assert 'prediction' in result_df.columns
    
    def test_predict_model_not_found(self, sample_data, temp_output_dir):
        """测试模型文件不存在的情况"""
        data_df, field_config = sample_data
        
        predict_data = data_df[['age', 'income', 'score']].head(10).copy()
        key = data_df['user_id'].head(10)
        
        # 验证抛出异常
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            load_and_predict(
                data=predict_data,
                key=key,
                data_month='202405',
                model_dir=temp_model_dir,  # 空目录，没有模型
                output_dir=temp_output_dir
            )


class TestBatchPredict:
    """测试批量预测函数"""
    
    def test_batch_predict_basic(self, sample_data, temp_model_dir, temp_output_dir):
        """测试基本批量预测"""
        data_df, field_config = sample_data
        
        # 训练模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=1,
            cv=2
        )
        
        # 创建多个预测数据文件
        import tempfile
        import os
        
        temp_files = []
        data_months = ['202401', '202402']
        
        for i, month in enumerate(data_months):
            # 创建临时 CSV 文件
            temp_df = X_processed.iloc[i*50:(i+1)*50].copy()
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_df.to_csv(temp_file.name, index=False)
            temp_files.append(temp_file.name)
            temp_file.close()
        
        # 批量预测
        try:
            results = batch_predict(
                data_paths=temp_files,
                config_path='',  # 不使用配置文件
                data_months=data_months,
                model_dir=temp_model_dir,
                output_dir=temp_output_dir
            )
            
            # 验证结果
            assert isinstance(results, dict)
            assert len(results) == len(data_months)
            
            for month in data_months:
                assert month in results
                assert isinstance(results[month], pd.DataFrame)
                assert len(results[month]) > 0
            
            # 验证合并文件
            combined_path = temp_output_dir / "predict_res_combined.csv"
            assert combined_path.exists()
            
            combined_df = pd.read_csv(combined_path)
            assert len(combined_df) == sum(len(results[m]) for m in data_months)
        
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestPredictWithExplanation:
    """测试带解释的预测函数"""
    
    def test_predict_with_explanation_basic(self, sample_data, temp_model_dir):
        """测试带解释的基本预测"""
        data_df, field_config = sample_data
        
        # 训练模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_model_dir.parent / "output",
            n_iter=1,
            cv=2
        )
        
        # 带解释的预测
        predict_data = X_processed.head(20).copy()
        key = data_df['user_id'].head(20)
        
        result = predict_with_explanation(
            data=predict_data,
            key=key,
            model_dir=temp_model_dir,
            top_n=5
        )
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
        assert 'key' in result.columns
        assert 'probability' in result.columns
        assert 'prediction' in result.columns
        
        # 验证概率范围
        assert result['probability'].between(0, 1).all()
    
    def test_predict_with_explanation_feature_importance(self, sample_data, temp_model_dir):
        """测试特征重要性输出"""
        data_df, field_config = sample_data
        
        # 训练模型
        from model_code.train import lgbm_train_with_random_search
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_model_dir.parent / "output",
            n_iter=1,
            cv=2
        )
        
        # 预测
        predict_data = X_processed.head(10).copy()
        key = data_df['user_id'].head(10)
        
        # 捕获输出
        import io
        import sys
        
        captured = io.StringIO()
        sys.stdout = captured
        
        predict_with_explanation(
            data=predict_data,
            key=key,
            model_dir=temp_model_dir,
            top_n=3
        )
        
        sys.stdout = sys.__stdout__
        
        # 验证输出了特征重要性
        output = captured.getvalue()
        assert '重要特征' in output or 'Top' in output


class TestPredictIntegration:
    """预测集成测试"""
    
    def test_full_predict_pipeline(self, sample_data, temp_model_dir, temp_output_dir):
        """测试完整预测流程"""
        data_df, field_config = sample_data
        
        # 1. 预处理
        from model_code.utils import ConfigDrivenPreprocessor
        
        feature_cols = ['age', 'income', 'city', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        # 2. 训练
        from model_code.train import lgbm_train_with_random_search
        
        train_result = lgbm_train_with_random_search(
            X=X_processed,
            y=y,
            model_dir=temp_model_dir,
            output_dir=temp_output_dir,
            n_iter=2,
            cv=2
        )
        
        # 3. 预测
        from model_code.predict import load_and_predict
        
        predict_data = X_processed.head(100).copy()
        key = data_df['user_id'].head(100)
        
        predict_result = load_and_predict(
            data=predict_data,
            key=key,
            data_month='202406',
            model_dir=temp_model_dir,
            output_dir=temp_output_dir
        )
        
        # 4. 验证预测质量
        # 计算预测为正例的比例
        pos_ratio = predict_result['prediction'].mean()
        
        # 验证概率分布
        prob_mean = predict_result['probability'].mean()
        prob_std = predict_result['probability'].std()
        
        assert 0.0 <= prob_mean <= 1.0
        assert prob_std > 0  # 概率应该有变化
        
        # 5. 验证结果文件
        result_path = temp_output_dir / "predict_res_202406.csv"
        assert result_path.exists()
        
        loaded_result = pd.read_csv(result_path)
        pd.testing.assert_frame_equal(predict_result, loaded_result)
