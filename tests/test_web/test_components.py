"""
测试 Web 组件 (web_app/components/)

测试覆盖：
- 图表组件
- 数据上传组件
- 配置编辑器组件
- 进度展示组件
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 导入被测试模块
from web_app.components.charts import (
    render_distribution_chart,
    render_correlation_heatmap,
    render_missing_value_heatmap,
    render_outlier_boxplot,
    render_roc_curve,
    render_precision_recall_curve,
    render_confusion_matrix,
    render_feature_importance,
    render_prediction_distribution,
    render_threshold_analysis,
)


class TestDistributionChart:
    """测试分布图组件"""
    
    def test_render_distribution_numeric(self, sample_data):
        """测试数值型分布图"""
        data_df, _ = sample_data
        
        fig = render_distribution_chart(data_df, 'age')
        
        # 验证返回 Plotly Figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # 验证标题
        assert 'age' in fig.layout.title.text.lower()
    
    def test_render_distribution_categorical(self, sample_data):
        """测试分类型分布图"""
        data_df, _ = sample_data
        
        fig = render_distribution_chart(data_df, 'city')
        
        # 验证返回 Figure
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # 验证标题
        assert 'city' in fig.layout.title.text.lower()


class TestCorrelationHeatmap:
    """测试相关性热力图"""
    
    def test_render_correlation_basic(self, sample_data):
        """测试基本相关性热力图"""
        data_df, _ = sample_data
        
        fig = render_correlation_heatmap(data_df)
        
        # 验证返回 Figure
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # 验证标题
        assert '相关性' in fig.layout.title.text or 'correlation' in fig.layout.title.text.lower()
    
    def test_render_correlation_no_numeric(self):
        """测试没有数值列的情况"""
        # 创建只有字符串的数据
        data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'city': ['北京', '上海', '广州']
        })
        
        fig = render_correlation_heatmap(data)
        
        # 验证返回 None
        assert fig is None


class TestMissingValueHeatmap:
    """测试缺失值热力图"""
    
    def test_render_missing_basic(self, sample_data):
        """测试基本缺失值热力图"""
        data_df, _ = sample_data
        
        fig = render_missing_value_heatmap(data_df)
        
        # 验证返回 Figure
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # 验证标题
        assert '缺失值' in fig.layout.title.text or 'missing' in fig.layout.title.text.lower()


class TestOutlierBoxplot:
    """测试箱线图"""
    
    def test_render_boxplot_basic(self, sample_data):
        """测试基本箱线图"""
        data_df, _ = sample_data
        
        numeric_cols = ['age', 'income', 'score']
        fig = render_outlier_boxplot(data_df, numeric_cols)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) > 0
        
        # 验证显示了多个列
        assert len(fig.data) <= len(numeric_cols)


class TestROCCurve:
    """测试 ROC 曲线"""
    
    def test_render_roc_basic(self, sample_data):
        """测试基本 ROC 曲线"""
        data_df, _ = sample_data
        
        # 创建预测概率
        np.random.seed(42)
        y_true = data_df['label'].values
        y_scores = np.random.uniform(0, 1, len(y_true))
        
        fig = render_roc_curve(y_true, y_scores)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) >= 2  # ROC 曲线 + 随机线
        
        # 验证标题
        assert 'ROC' in fig.layout.title.text


class TestPrecisionRecallCurve:
    """测试 Precision-Recall 曲线"""
    
    def test_render_pr_basic(self, sample_data):
        """测试基本 PR 曲线"""
        data_df, _ = sample_data
        
        np.random.seed(42)
        y_true = data_df['label'].values
        y_scores = np.random.uniform(0, 1, len(y_true))
        
        fig = render_precision_recall_curve(y_true, y_scores)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) > 0
        
        # 验证标题
        assert 'PR' in fig.layout.title.text or 'Precision-Recall' in fig.layout.title.text


class TestConfusionMatrix:
    """测试混淆矩阵"""
    
    def test_render_cm_basic(self, sample_data):
        """测试基本混淆矩阵"""
        data_df, _ = sample_data
        
        # 创建预测标签
        np.random.seed(42)
        y_true = data_df['label'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        
        fig = render_confusion_matrix(y_true, y_pred)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) > 0
        
        # 验证标题
        assert '混淆' in fig.layout.title.text or 'confusion' in fig.layout.title.text.lower()


class TestFeatureImportance:
    """测试特征重要性图"""
    
    def test_render_feature_importance_basic(self, sample_data):
        """测试基本特征重要性图"""
        feature_names = ['age', 'income', 'score', 'city', 'region']
        importances = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        fig = render_feature_importance(feature_names, importances, top_n=5)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) > 0
        
        # 验证标题
        assert '特征重要性' in fig.layout.title.text or 'feature' in fig.layout.title.text.lower()
        
        # 验证显示的特征数量
        assert len(fig.data[0].y) <= 5
    
    def test_render_feature_importance_top_n(self, sample_data):
        """测试 top_n 参数"""
        feature_names = [f'feature_{i}' for i in range(20)]
        importances = np.random.uniform(0, 1, 20)
        
        fig = render_feature_importance(feature_names, importances, top_n=10)
        
        # 验证只显示前 10 个特征
        assert len(fig.data[0].y) == 10


class TestPredictionDistribution:
    """测试预测分布图"""
    
    def test_render_prediction_dist_basic(self, sample_data):
        """测试基本预测分布图"""
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, 1000)
        
        fig = render_prediction_distribution(predictions, threshold=0.5)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) >= 2  # 正例和负例分布
        
        # 验证标题
        assert '预测' in fig.layout.title.text or 'prediction' in fig.layout.title.text.lower()


class TestThresholdAnalysis:
    """测试阈值分析图"""
    
    def test_render_threshold_analysis_basic(self, sample_data):
        """测试基本阈值分析图"""
        data_df, _ = sample_data
        
        np.random.seed(42)
        y_true = data_df['label'].values
        y_scores = np.random.uniform(0, 1, len(y_true))
        
        fig = render_threshold_analysis(y_true, y_scores)
        
        # 验证返回 Figure
        assert fig is not None
        assert len(fig.data) >= 3  # Precision, Recall, F1
        
        # 验证标题
        assert '阈值' in fig.layout.title.text or 'threshold' in fig.layout.title.text.lower()


# ============================================================================
# 测试文件处理工具
# ============================================================================

class TestFileHandler:
    """测试文件处理工具"""
    
    def test_calculate_data_statistics(self, sample_data):
        """测试数据统计计算"""
        from web_app.utils.file_handler import calculate_data_statistics
        
        data_df, _ = sample_data
        stats = calculate_data_statistics(data_df)
        
        # 验证返回字典
        assert isinstance(stats, dict)
        
        # 验证包含必要字段
        assert 'total_rows' in stats
        assert 'total_columns' in stats
        assert 'memory_usage' in stats
        assert 'columns_info' in stats
        assert 'missing_summary' in stats
        
        # 验证值正确
        assert stats['total_rows'] == len(data_df)
        assert stats['total_columns'] == len(data_df.columns)
        assert len(stats['columns_info']) == len(data_df.columns)
    
    def test_create_default_config(self, sample_data):
        """测试创建默认配置"""
        from web_app.utils.file_handler import create_default_config
        
        data_df, _ = sample_data
        config = create_default_config(data_df.columns.tolist())
        
        # 验证返回 DataFrame
        assert isinstance(config, pd.DataFrame)
        
        # 验证列
        assert 'field_name' in config.columns
        assert 'field_type' in config.columns
        assert 'outlier_value' in config.columns
        assert 'missing_value' in config.columns
        
        # 验证行数
        assert len(config) == len(data_df.columns)
    
    def test_infer_field_type(self, sample_data):
        """测试字段类型推断"""
        from web_app.utils.file_handler import infer_field_type
        
        data_df, _ = sample_data
        
        # 测试数值型
        assert infer_field_type(data_df['age']) == 'num'
        assert infer_field_type(data_df['income']) == 'num'
        
        # 测试 ID 类型
        assert infer_field_type(data_df['user_id']) == 'key'
        
        # 测试字符串类型
        assert infer_field_type(data_df['city']) == 'str'


# ============================================================================
# 测试会话状态管理
# ============================================================================

class TestSessionState:
    """测试会话状态管理"""
    
    def test_init_session_state_structure(self):
        """测试会话状态初始化结构"""
        from web_app.utils.session_state import init_session_state
        
        # 注意：这个函数需要 streamlit 的 session_state
        # 这里只测试函数存在和可导入
        assert init_session_state is not None
        assert callable(init_session_state)
    
    def test_clear_functions_exist(self):
        """测试清除函数存在"""
        from web_app.utils.session_state import (
            clear_data_state,
            clear_training_state,
            clear_prediction_state,
        )
        
        assert clear_data_state is not None
        assert clear_training_state is not None
        assert clear_prediction_state is not None
