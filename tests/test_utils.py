"""
测试预处理函数 (model_code/utils.py)

测试覆盖：
- 智能采样函数
- 类型转换函数
- 异常值处理函数
- 缺失值处理函数
- 编码函数
- 标准化函数
- ConfigDrivenPreprocessor 类
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 导入被测试模块
from model_code.utils import (
    smart_sampling,
    neg_data_undersampling,
    convert_dataframe_types_vectorized,
    handle_dataframe_outliers,
    handle_dataframe_missing_values,
    encode_dataframe,
    standardize_dataframe,
    ConfigDrivenPreprocessor,
)


class TestSmartSampling:
    """测试智能采样函数"""
    
    def test_sampling_basic(self, sample_data):
        """测试基本采样功能"""
        data_df, _ = sample_data
        
        result = smart_sampling(data_df, label_col='label', negative_ratio=0.1, random_state=42)
        
        # 验证结果类型
        assert isinstance(result, pd.DataFrame)
        
        # 验证正样本全部保留
        original_positive = len(data_df[data_df['label'] == 1])
        result_positive = len(result[result['label'] == 1])
        assert result_positive == original_positive
        
        # 验证负样本按比例抽样
        original_negative = len(data_df[data_df['label'] == 0])
        result_negative = len(result[result['label'] == 0])
        expected_negative = int(original_negative * 0.1)
        # 允许少量误差
        assert abs(result_negative - expected_negative) <= 2
        
        # 验证数据被打乱
        assert not result.equals(data_df.head(len(result)))
    
    def test_sampling_edge_cases(self, sample_data):
        """测试边界情况"""
        data_df, _ = sample_data
        
        # 测试 negative_ratio=1.0 (全部保留)
        result_full = smart_sampling(data_df, label_col='label', negative_ratio=1.0, random_state=42)
        assert len(result_full) == len(data_df)
        
        # 测试 negative_ratio=0.0 (只保留正样本)
        result_only_pos = smart_sampling(data_df, label_col='label', negative_ratio=0.0, random_state=42)
        assert len(result_only_pos) == len(data_df[data_df['label'] == 1])


class TestNegDataUndersampling:
    """测试负样本欠采样"""
    
    def test_undersampling_basic(self, sample_data):
        """测试基本欠采样功能"""
        data_df, field_config = sample_data
        
        # 分离正负样本
        positive_df = data_df[data_df['label'] == 1]
        negative_df = data_df[data_df['label'] == 0]
        
        # 只保留用于比较的数值列，确保形状匹配
        numeric_cols = ['age', 'income']
        positive_numeric = positive_df[numeric_cols]
        negative_numeric = negative_df[numeric_cols]
        
        # 计算正样本统计量
        target_stats = positive_numeric.describe().T[["mean", "std"]].values
        
        # 执行欠采样 (使用更小的步长和更多迭代)
        result_index = neg_data_undersampling(
            negative_numeric,  # 只传入数值列
            target_stats, 
            step_size=50, 
            max_iter=20, 
            error_rate=0.5,  # 放宽误差范围
            max_samples=500
        )
        
        # 验证返回结果
        assert result_index is not None
        assert len(result_index) > 0
        assert len(result_index) <= len(negative_numeric)


class TestConvertDataFrameTypes:
    """测试类型转换函数"""
    
    def test_convert_basic(self, sample_data):
        """测试基本类型转换"""
        data_df, field_config = sample_data
        
        # 创建包含字符串数值的数据
        data_df['age_str'] = data_df['age'].astype(str)
        field_config = pd.concat([
            field_config,
            pd.DataFrame([{'field_name': 'age_str', 'field_type': 'num', 'outlier_value': np.nan,
                          'missing_value': np.nan, 'ecode_type': np.nan, 'standardize_type': np.nan}])
        ], ignore_index=True)
        
        result = convert_dataframe_types_vectorized(data_df, field_config)
        
        # 验证转换后的类型
        assert pd.api.types.is_numeric_dtype(result['age_str'])
    
    def test_convert_no_num_fields(self, sample_data):
        """测试没有数值字段的情况"""
        data_df, field_config = sample_data
        
        # 创建没有 num 类型的配置
        field_config_no_num = field_config.copy()
        field_config_no_num.loc[field_config_no_num['field_type'] == 'num', 'field_type'] = 'str'
        
        result = convert_dataframe_types_vectorized(data_df, field_config_no_num)
        
        # 验证数据不变
        pd.testing.assert_frame_equal(result, data_df)


class TestHandleOutliers:
    """测试异常值处理"""
    
    def test_handle_3std(self, small_sample_data):
        """测试 3σ 异常值处理"""
        data_df, field_config = small_sample_data
        
        # 记录原始异常值
        original_outlier = data_df.loc[3, 'income']  # -100000
        
        result = handle_dataframe_outliers(data_df, field_config)
        
        # 验证异常值被裁剪到范围内
        assert result.loc[3, 'income'] >= 0
        assert result.loc[3, 'income'] <= 100000
        assert result.loc[3, 'income'] != original_outlier
    
    def test_handle_range(self, small_sample_data):
        """测试固定范围异常值处理"""
        data_df, field_config = small_sample_data
        
        result = handle_dataframe_outliers(data_df, field_config)
        
        # 验证所有值都在指定范围内
        assert result['income'].min() >= 0
        assert result['income'].max() <= 100000
    
    def test_handle_no_outliers(self, sample_data):
        """测试没有异常值的情况"""
        data_df, field_config = sample_data
        
        # 创建不包含异常值处理配置的数据
        field_config_no_outlier = field_config.copy()
        field_config_no_outlier['outlier_value'] = np.nan
        
        result = handle_dataframe_outliers(data_df, field_config_no_outlier)
        
        # 验证数据不变
        pd.testing.assert_frame_equal(result, data_df)


class TestHandleMissingValues:
    """测试缺失值处理"""
    
    def test_handle_mean(self, small_sample_data):
        """测试均值填充"""
        data_df, field_config = small_sample_data
        
        # 记录缺失值位置
        missing_mask = data_df['age'].isna()
        original_mean = data_df['age'].mean()
        
        result = handle_dataframe_missing_values(data_df, field_config)
        
        # 验证缺失值被填充
        assert not result['age'].isna().any()
        
        # 验证填充值为均值
        filled_values = result.loc[missing_mask, 'age']
        expected_mean = original_mean  # 使用原始均值
        assert np.allclose(filled_values, expected_mean, rtol=0.1)
    
    def test_handle_median(self, small_sample_data):
        """测试中位数填充"""
        data_df, field_config = small_sample_data
        
        # 修改配置为中位数填充
        field_config.loc[field_config['field_name'] == 'age', 'missing_value'] = 'median'
        
        result = handle_dataframe_missing_values(data_df, field_config)
        
        # 验证缺失值被填充
        assert not result['age'].isna().any()
    
    def test_handle_mode(self, sample_data):
        """测试众数填充"""
        data_df, field_config = sample_data
        
        # 创建分类字段并添加缺失值
        data_df['category'] = np.random.choice(['A', 'B', 'C'], len(data_df))
        data_df.loc[0:10, 'category'] = np.nan
        
        field_config = pd.concat([
            field_config,
            pd.DataFrame([{'field_name': 'category', 'field_type': 'str', 'outlier_value': np.nan,
                          'missing_value': 'mode', 'ecode_type': np.nan, 'standardize_type': np.nan}])
        ], ignore_index=True)
        
        result = handle_dataframe_missing_values(data_df, field_config)
        
        # 验证缺失值被填充
        assert not result['category'].isna().any()


class TestEncodeDataframe:
    """测试编码函数"""
    
    def test_one_hot_encoding(self, small_sample_data):
        """测试独热编码"""
        data_df, field_config = small_sample_data
        
        result = encode_dataframe(data_df, field_config)
        
        # 验证原始列被移除
        assert 'city' not in result.columns
        
        # 验证生成了独热编码列
        city_cols = [col for col in result.columns if col.startswith('city_')]
        assert len(city_cols) > 0
        
        # 验证编码正确
        assert result[city_cols].sum().sum() == len(data_df)
    
    def test_ordinal_encoding(self, sample_data):
        """测试顺序编码"""
        data_df, field_config = sample_data
        
        # 修改配置为顺序编码
        field_config.loc[field_config['field_name'] == 'city', 'ecode_type'] = ''
        
        result = encode_dataframe(data_df, field_config)
        
        # 验证列仍然存在
        assert 'city' in result.columns
        
        # 验证类型为数值
        assert pd.api.types.is_numeric_dtype(result['city'])


class TestStandardizeDataframe:
    """测试标准化函数"""
    
    def test_zscore_standardization(self, sample_data):
        """测试 Z-score 标准化"""
        data_df, field_config = sample_data
        
        result = standardize_dataframe(data_df, field_config)
        
        # 验证标准化后均值为 0，标准差为 1 (近似)
        age_mean = result['age'].mean()
        age_std = result['age'].std()
        
        assert np.abs(age_mean) < 0.001  # 接近 0
        assert np.abs(age_std - 1.0) < 0.001  # 接近 1
    
    def test_minmax_standardization(self, sample_data):
        """测试 Min-Max 标准化"""
        data_df, field_config = sample_data
        
        result = standardize_dataframe(data_df, field_config)
        
        # 验证标准化后范围在 [0, 1]
        income_min = result['income'].min()
        income_max = result['income'].max()
        
        assert np.abs(income_min) < 0.001  # 接近 0
        assert np.abs(income_max - 1.0) < 0.001  # 接近 1
    
    def test_robust_standardization(self, sample_data):
        """测试鲁棒标准化"""
        data_df, field_config = sample_data
        
        result = standardize_dataframe(data_df, field_config)
        
        # 验证列存在
        assert 'score' in result.columns


class TestConfigDrivenPreprocessor:
    """测试配置驱动的预处理器类"""
    
    def test_preprocessor_fit_transform(self, sample_data):
        """测试预处理器的 fit 和 transform"""
        data_df, field_config = sample_data
        
        # 移除 label 和 key 列用于预处理
        feature_cols = ['age', 'income', 'city', 'score']
        X = data_df[feature_cols].copy()
        
        # 创建预处理器
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        
        # 拟合
        preprocessor.fit(X)
        
        # 验证拟合后的属性
        assert len(preprocessor.fill_values_) > 0
        assert len(preprocessor.outlier_bounds_) > 0
        assert len(preprocessor.scalers_) > 0
        
        # 转换
        X_transformed = preprocessor.transform(X)
        
        # 验证输出类型
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
    
    def test_preprocessor_prevent_data_leakage(self, sample_data):
        """测试预处理器防止数据穿越"""
        data_df, field_config = sample_data
        
        feature_cols = ['age', 'income', 'city', 'score']
        X = data_df[feature_cols].copy()
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.7)
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        
        # 创建并拟合预处理器
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        preprocessor.fit(X_train)
        
        # 记录训练集统计量
        train_mean = X_train['age'].mean()
        
        # 转换测试集
        X_test_transformed = preprocessor.transform(X_test)
        
        # 验证测试集使用训练集的统计量
        # (这里通过验证预处理器确实使用了 fit 时的统计量来间接验证)
        assert hasattr(preprocessor, 'fill_values_')
        assert 'age' in preprocessor.fill_values_
        assert preprocessor.fill_values_['age'] == train_mean
    
    def test_preprocessor_with_pipeline(self, sample_data):
        """测试预处理器与 sklearn Pipeline 兼容"""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        data_df, field_config = sample_data
        
        # 准备数据
        feature_cols = ['age', 'income', 'score']
        X = data_df[feature_cols].copy()
        y = data_df['label'].copy()
        
        # 创建预处理器
        preprocessor = ConfigDrivenPreprocessor(df_field=field_config)
        
        # 创建 Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        
        # 验证 Pipeline 可以正常工作
        try:
            pipeline.fit(X, y)
            predictions = pipeline.predict(X)
            assert len(predictions) == len(y)
        except Exception as e:
            pytest.fail(f"Pipeline 执行失败：{e}")
