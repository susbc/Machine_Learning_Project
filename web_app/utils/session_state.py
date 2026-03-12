"""
Streamlit 会话状态管理模块

提供统一的 session_state 管理，确保跨页面状态共享
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional
from pathlib import Path


# ============================================================================
# 初始化函数
# ============================================================================

def init_session_state() -> None:
    """
    初始化所有会话状态变量
    应在每个页面加载时调用
    """
    
    # ==================== 项目路径配置 ====================
    if "project_root" not in st.session_state:
        st.session_state.project_root = Path(__file__).parent.parent.parent
    
    if "data_dir" not in st.session_state:
        st.session_state.data_dir = st.session_state.project_root / "input_data"
        st.session_state.data_dir.mkdir(parents=True, exist_ok=True)
    
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = st.session_state.project_root / "model_code" / "saved"
        st.session_state.model_dir.mkdir(parents=True, exist_ok=True)
    
    if "config_dir" not in st.session_state:
        st.session_state.config_dir = st.session_state.project_root / "model_code" / "config"
        st.session_state.config_dir.mkdir(parents=True, exist_ok=True)
    
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = st.session_state.project_root / "output"
        st.session_state.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 数据状态 ====================
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data: Optional[pd.DataFrame] = None
    
    if "data_file_path" not in st.session_state:
        st.session_state.data_file_path: Optional[str] = None
    
    if "data_preview_limit" not in st.session_state:
        st.session_state.data_preview_limit = 100
    
    if "data_statistics" not in st.session_state:
        st.session_state.data_statistics: Optional[Dict] = None
    
    # ==================== 配置状态 ====================
    if "field_config" not in st.session_state:
        st.session_state.field_config: Optional[pd.DataFrame] = None
    
    if "config_file_path" not in st.session_state:
        st.session_state.config_file_path: Optional[str] = None
    
    if "config_modified" not in st.session_state:
        st.session_state.config_modified = False
    
    # ==================== 训练状态 ====================
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False
    
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = 0
    
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    
    if "training_results" not in st.session_state:
        st.session_state.training_results: Optional[Dict] = None
    
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 50,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    
    if "cv_results" not in st.session_state:
        st.session_state.cv_results: Optional[Dict] = None
    
    # ==================== 模型管理状态 ====================
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model: Optional[str] = None
    
    if "model_details" not in st.session_state:
        st.session_state.model_details: Optional[Dict] = None
    
    # ==================== 预测状态 ====================
    if "predict_data" not in st.session_state:
        st.session_state.predict_data: Optional[pd.DataFrame] = None
    
    if "predict_results" not in st.session_state:
        st.session_state.predict_results: Optional[pd.DataFrame] = None
    
    if "prediction_threshold" not in st.session_state:
        st.session_state.prediction_threshold = 0.5
    
    # ==================== 分析状态 ====================
    if "eda_charts" not in st.session_state:
        st.session_state.eda_charts = {}
    
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    
    if "report_path" not in st.session_state:
        st.session_state.report_path: Optional[str] = None


# ============================================================================
# 状态管理工具函数
# ============================================================================

def clear_data_state() -> None:
    """清除数据相关状态"""
    st.session_state.uploaded_data = None
    st.session_state.data_file_path = None
    st.session_state.data_statistics = None
    st.session_state.eda_charts = {}


def clear_training_state() -> None:
    """清除训练相关状态"""
    st.session_state.training_in_progress = False
    st.session_state.training_progress = 0
    st.session_state.training_logs = []
    st.session_state.training_results = None
    st.session_state.cv_results = None


def clear_prediction_state() -> None:
    """清除预测相关状态"""
    st.session_state.predict_data = None
    st.session_state.predict_results = None


def add_training_log(message: str) -> None:
    """添加训练日志"""
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    st.session_state.training_logs.append(message)


def get_training_logs() -> list:
    """获取训练日志"""
    return st.session_state.get("training_logs", [])


# ============================================================================
# 页面导航状态
# ============================================================================

def set_current_page(page_name: str) -> None:
    """设置当前页面"""
    st.session_state.current_page = page_name


def get_current_page() -> str:
    """获取当前页面"""
    return st.session_state.get("current_page", "首页")
