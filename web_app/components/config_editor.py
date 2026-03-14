"""
配置编辑器组件

提供 field_config.csv 的在线编辑、保存、导入导出功能
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.file_handler import (
    save_config_to_excel,
    read_file_from_path,
    create_default_config,
    infer_field_type,
)


# ============================================================================
# 配置选项定义
# ============================================================================

FIELD_TYPE_OPTIONS = ["num", "str", "key", "label"]
OUTLIER_OPTIONS = [None, "3std", "5std", "(0,100)", "(0,1000)"]
MISSING_VALUE_OPTIONS = [None, "mean", "median", "mode", "0", "-1"]
ECODE_TYPE_OPTIONS = [None, "one-hot", "ordinal"]
STANDARDIZE_TYPE_OPTIONS = [None, "z-score", "min-max", "robust"]


def render_config_editor(config_df: pd.DataFrame) -> pd.DataFrame:
    """
    渲染配置编辑器
    
    Args:
        config_df: 配置 DataFrame
    
    Returns:
        编辑后的配置 DataFrame
    """
    st.header("⚙️ 配置编辑")
    
    # 清理 NaN 值，替换为 None（Streamlit data_editor 要求）
    clean_config = config_df.copy()
    for col in clean_config.columns:
        if clean_config[col].dtype == object:
            clean_config[col] = clean_config[col].where(
                pd.notna(clean_config[col]), None
            )
    
    # 使用 data_editor 进行在线编辑
    edited_config = st.data_editor(
        clean_config,
        num_rows="dynamic",
        use_container_width=True,
        height=400,
        column_config={
            "field_name": st.column_config.TextColumn(
                "字段名",
                help="字段名称",
                disabled=True,  # 字段名不可编辑
            ),
            "field_type": st.column_config.SelectboxColumn(
                "字段类型",
                help="字段的数据类型",
                options=FIELD_TYPE_OPTIONS,
                required=True,
            ),
            "outlier_value": st.column_config.SelectboxColumn(
                "异常值处理",
                help="异常值处理方法",
                options=OUTLIER_OPTIONS,
                required=False,
            ),
            "missing_value": st.column_config.SelectboxColumn(
                "缺失值填充",
                help="缺失值填充方法",
                options=MISSING_VALUE_OPTIONS,
                required=False,
            ),
            "ecode_type": st.column_config.SelectboxColumn(
                "编码方式",
                help="分类变量编码方式",
                options=ECODE_TYPE_OPTIONS,
                required=False,
            ),
            "standardize_type": st.column_config.SelectboxColumn(
                "标准化",
                help="数值标准化方法",
                options=STANDARDIZE_TYPE_OPTIONS,
                required=False,
            ),
        },
        hide_index=True,
    )
    
    return edited_config


def render_config_actions(config_df: pd.DataFrame, config_path: Optional[str] = None) -> None:
    """
    渲染配置操作按钮（保存、导入、导出）
    
    Args:
        config_df: 配置 DataFrame
        config_path: 配置文件路径
    """
    st.header("💾 配置操作")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 保存配置
        if st.button("💾 保存配置", use_container_width=True):
            if config_path:
                save_path = config_path
            else:
                save_path = st.session_state.config_dir / "field_config.xlsx"
            
            if save_config_to_excel(config_df, str(save_path)):
                st.success(f"配置已保存至：{save_path}")
                st.session_state.config_file_path = str(save_path)
            else:
                st.error("保存失败")
    
    with col2:
        # 导出配置
        from utils.file_handler import create_excel_download_button
        create_excel_download_button(
            config_df,
            "field_config_export.xlsx",
            "📥 导出配置"
        )
    
    with col3:
        # 导入配置
        uploaded_config = st.file_uploader(
            "📂 导入配置",
            type=["xlsx", "xls", "csv"],
            key="config_import"
        )
        
        if uploaded_config is not None:
            try:
                if uploaded_config.name.endswith('.csv'):
                    imported_config = pd.read_csv(uploaded_config)
                else:
                    imported_config = pd.read_excel(uploaded_config)
                
                st.session_state.field_config = imported_config
                st.session_state.config_modified = True
                st.success("配置导入成功！页面将刷新...")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败：{str(e)}")
    
    with col4:
        # 重置配置
        if st.button("🔄 重置配置", use_container_width=True):
            if st.session_state.get('uploaded_data') is not None:
                st.session_state.field_config = create_default_config(
                    st.session_state.uploaded_data.columns.tolist()
                )
                st.session_state.config_modified = True
                st.success("配置已重置")
                st.rerun()


def render_field_type_distribution(config_df: pd.DataFrame) -> None:
    """
    渲染字段类型分布图
    
    Args:
        config_df: 配置 DataFrame
    """
    st.header("📊 字段类型分布")
    
    if 'field_type' in config_df.columns:
        type_counts = config_df['field_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 饼图
            import plotly.express as px
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title='字段类型分布',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 统计表格
            type_df = pd.DataFrame({
                '字段类型': type_counts.index,
                '数量': type_counts.values
            })
            st.dataframe(type_df, use_container_width=True, hide_index=True)


def render_config_summary(config_df: pd.DataFrame) -> None:
    """
    渲染配置摘要信息
    
    Args:
        config_df: 配置 DataFrame
    """
    st.header("📋 配置摘要")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总字段数", len(config_df))
    
    with col2:
        if 'field_type' in config_df.columns:
            num_count = len(config_df[config_df['field_type'] == 'num'])
            st.metric("数值型字段", num_count)
    
    with col3:
        if 'field_type' in config_df.columns:
            str_count = len(config_df[config_df['field_type'] == 'str'])
            st.metric("分类型字段", str_count)
    
    with col4:
        if 'missing_value' in config_df.columns:
            fill_count = config_df['missing_value'].notna().sum()
            st.metric("配置缺失填充", fill_count)
    
    # 配置完整性检查
    st.subheader("✅ 配置完整性检查")
    
    issues = []
    
    # 检查必填字段
    if 'field_name' not in config_df.columns:
        issues.append("❌ 缺少 field_name 列")
    
    if 'field_type' not in config_df.columns:
        issues.append("❌ 缺少 field_type 列")
    
    # 检查 label 字段
    if 'field_type' in config_df.columns:
        if 'label' not in config_df['field_type'].values:
            issues.append("⚠️ 未指定 label 字段（目标变量）")
        else:
            issues.append("✅ 已指定 label 字段")
    
    # 检查 key 字段
    if 'field_type' in config_df.columns:
        if 'key' not in config_df['field_type'].values:
            issues.append("⚠️ 未指定 key 字段（ID 标识）")
        else:
            issues.append("✅ 已指定 key 字段")
    
    if issues:
        for issue in issues:
            st.write(issue)
    else:
        st.success("✅ 配置完整，无问题")


def auto_infer_config(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据数据自动推断配置
    
    Args:
        data_df: 数据 DataFrame
    
    Returns:
        推断的配置 DataFrame
    """
    config_data = []
    
    for col in data_df.columns:
        field_type = infer_field_type(data_df[col])
        
        # 智能推断缺失值填充策略
        missing_value = np.nan
        if data_df[col].isna().sum() > 0:
            if field_type == 'num':
                missing_value = 'mean'
            else:
                missing_value = 'mode'
        
        config_data.append({
            "field_name": col,
            "field_type": field_type,
            "outlier_value": np.nan,
            "missing_value": missing_value,
            "ecode_type": "one-hot" if field_type == 'str' else np.nan,
            "standardize_type": "z-score" if field_type == 'num' else np.nan,
        })
    
    return pd.DataFrame(config_data)


def render_config_management_page() -> None:
    """
    配置管理主页面渲染
    """
    st.set_page_config(
        page_title="配置管理",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ 配置管理")
    st.markdown("---")
    
    # 检查是否有数据
    if st.session_state.get('uploaded_data') is None:
        st.warning("⚠️ 请先在【数据管理】页面上传数据")
        st.info("💡 配置管理需要基于已上传的数据生成字段配置")
        return
    
    # 初始化或加载配置
    if st.session_state.get('field_config') is None:
        # 尝试加载现有配置
        config_path = st.session_state.config_dir / "field_config.xlsx"
        if config_path.exists():
            st.session_state.field_config = read_file_from_path(str(config_path))
            st.session_state.config_file_path = str(config_path)
            st.success(f"已加载现有配置：{config_path}")
        else:
            # 自动推断配置
            st.session_state.field_config = auto_infer_config(
                st.session_state.uploaded_data
            )
            st.info("✨ 已根据数据自动生成配置，您可以进行调整")
    
    config_df = st.session_state.field_config
    
    # 渲染编辑器
    edited_config = render_config_editor(config_df)
    
    # 检测配置变化
    if not edited_config.equals(config_df):
        st.session_state.field_config = edited_config
        st.session_state.config_modified = True
    
    st.markdown("---")
    
    # 渲染操作按钮
    render_config_actions(edited_config, st.session_state.get('config_file_path'))
    
    st.markdown("---")
    
    # 渲染分布和摘要
    col1, col2 = st.columns(2)
    
    with col1:
        render_field_type_distribution(edited_config)
    
    with col2:
        render_config_summary(edited_config)
