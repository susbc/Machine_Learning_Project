"""
Pytest 配置文件

提供全局的 fixture 和配置
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_data():
    """
    创建示例数据用于测试
    
    Returns:
        tuple: (data_df, field_config_df)
    """
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'label': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # 添加一些缺失值
    data_df = pd.DataFrame(data)
    data_df.loc[np.random.choice(n_samples, 50), 'age'] = np.nan
    data_df.loc[np.random.choice(n_samples, 30), 'income'] = np.nan
    
    # 添加一些异常值
    data_df.loc[0:5, 'income'] = [1000000, -50000, 2000000, -100000, 1500000, -80000]
    
    # 创建字段配置
    field_config = pd.DataFrame([
        {'field_name': 'user_id', 'field_type': 'key', 'outlier_value': np.nan, 
         'missing_value': np.nan, 'ecode_type': np.nan, 'standardize_type': np.nan},
        {'field_name': 'age', 'field_type': 'num', 'outlier_value': '3std', 
         'missing_value': 'mean', 'ecode_type': np.nan, 'standardize_type': 'z-score'},
        {'field_name': 'income', 'field_type': 'num', 'outlier_value': '(0, 1000000)', 
         'missing_value': 'median', 'ecode_type': np.nan, 'standardize_type': 'min-max'},
        {'field_name': 'city', 'field_type': 'str', 'outlier_value': np.nan, 
         'missing_value': 'mode', 'ecode_type': 'one-hot', 'standardize_type': np.nan},
        {'field_name': 'score', 'field_type': 'num', 'outlier_value': '5std', 
         'missing_value': 'mean', 'ecode_type': np.nan, 'standardize_type': 'robust'},
        {'field_name': 'label', 'field_type': 'label', 'outlier_value': np.nan, 
         'missing_value': np.nan, 'ecode_type': np.nan, 'standardize_type': np.nan},
    ])
    
    return data_df, field_config


@pytest.fixture
def small_sample_data():
    """
    创建小样本数据用于快速测试
    
    Returns:
        tuple: (data_df, field_config_df)
    """
    np.random.seed(42)
    
    data = {
        'user_id': [1, 2, 3, 4, 5],
        'age': [25, 30, np.nan, 45, 50],
        'income': [50000, 60000, 70000, -100000, 90000],  # 包含异常值
        'city': ['北京', '上海', '广州', '深圳', '北京'],
        'label': [0, 1, 0, 1, 0]
    }
    
    data_df = pd.DataFrame(data)
    
    field_config = pd.DataFrame([
        {'field_name': 'user_id', 'field_type': 'key', 'outlier_value': np.nan, 
         'missing_value': np.nan, 'ecode_type': np.nan, 'standardize_type': np.nan},
        {'field_name': 'age', 'field_type': 'num', 'outlier_value': '3std', 
         'missing_value': 'mean', 'ecode_type': np.nan, 'standardize_type': 'z-score'},
        {'field_name': 'income', 'field_type': 'num', 'outlier_value': '(0, 100000)', 
         'missing_value': 'median', 'ecode_type': np.nan, 'standardize_type': 'min-max'},
        {'field_name': 'city', 'field_type': 'str', 'outlier_value': np.nan, 
         'missing_value': 'mode', 'ecode_type': 'one-hot', 'standardize_type': np.nan},
        {'field_name': 'label', 'field_type': 'label', 'outlier_value': np.nan, 
         'missing_value': np.nan, 'ecode_type': np.nan, 'standardize_type': np.nan},
    ])
    
    return data_df, field_config


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    创建临时输出目录
    
    Args:
        tmp_path: pytest 临时路径 fixture
    
    Returns:
        Path: 临时输出目录路径
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """
    创建临时模型目录
    
    Args:
        tmp_path: pytest 临时路径 fixture
    
    Returns:
        Path: 临时模型目录路径
    """
    model_dir = tmp_path / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
