"""
数据上传组件

提供文件上传、预览、统计信息展示功能
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from pathlib import Path

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.file_handler import (
    read_uploaded_file,
    read_file_from_path,
    calculate_data_statistics,
    create_download_button,
)


def render_data_upload_section() -> Optional[pd.DataFrame]:
    """
    渲染数据上传区域
    
    Returns:
        上传的 DataFrame 或 None
    """
    st.header("📤 数据上传")
    
    # 文件上传器
    uploaded_file = st.file_uploader(
        "上传数据文件 (支持 CSV、Excel 格式)",
        type=["csv", "xlsx", "xls"],
        help="支持的文件格式：.csv, .xlsx, .xls"
    )
    
    if uploaded_file is not None:
        st.success(f"文件已上传：{uploaded_file.name}")
        return read_uploaded_file(uploaded_file)
    
    return None


def render_data_preview(df: pd.DataFrame, limit: int = 100) -> None:
    """
    渲染数据预览表格
    
    Args:
        df: 要预览的 DataFrame
        limit: 显示行数限制
    """
    st.header("📊 数据预览")
    
    # 行数选择器
    preview_limit = st.slider(
        "显示行数",
        min_value=10,
        max_value=min(len(df), 500),
        value=min(limit, len(df)),
        step=10
    )
    
    # 显示数据表格
    st.dataframe(
        df.head(preview_limit),
        use_container_width=True,
        height=400
    )
    
    st.caption(f"显示前 {preview_limit} 行，共 {len(df)} 行")


def render_data_statistics(df: pd.DataFrame) -> None:
    """
    渲染数据统计信息
    
    Args:
        df: 要分析的 DataFrame
    """
    st.header("📈 数据统计信息")
    
    # 计算统计信息
    stats = calculate_data_statistics(df)
    
    # 基础统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总行数", f"{stats['total_rows']:,}")
    
    with col2:
        st.metric("总列数", f"{stats['total_columns']}")
    
    with col3:
        st.metric("内存占用", f"{stats['memory_usage']:.2f} MB")
    
    with col4:
        st.metric("数据类型数", len(stats['dtype_summary']))
    
    # 数据类型分布
    st.subheader("数据类型分布")
    if stats['dtype_summary']:
        dtype_df = pd.DataFrame({
            '数据类型': list(stats['dtype_summary'].keys()),
            '数量': list(stats['dtype_summary'].values())
        })
        
        fig = px.bar(
            dtype_df,
            x='数据类型',
            y='数量',
            title='数据类型分布',
            color='数据类型',
            text='数量'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 缺失值分析
    st.subheader("缺失值分析")
    missing_df = pd.DataFrame({
        '字段名': list(stats['missing_summary'].keys()),
        '缺失率 (%)': list(stats['missing_summary'].values())
    })
    missing_df = missing_df[missing_df['缺失率 (%)'] > 0].sort_values('缺失率 (%)', ascending=False)
    
    if len(missing_df) > 0:
        fig = px.bar(
            missing_df.head(20),
            x='缺失率 (%)',
            y='字段名',
            orientation='h',
            title='缺失率 Top 20 字段',
            color='缺失率 (%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ 数据完整，无缺失值")
    
    # 唯一值分析
    st.subheader("唯一值分析")
    unique_df = pd.DataFrame({
        '字段名': list(stats['unique_summary'].keys()),
        '唯一值数量': list(stats['unique_summary'].values())
    })
    unique_df = unique_df.sort_values('唯一值数量', ascending=True)
    
    fig = px.bar(
        unique_df.head(20),
        x='唯一值数量',
        y='字段名',
        orientation='h',
        title='唯一值数量 Top 20 字段',
        color='唯一值数量',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 数值列统计
    if 'numeric_stats' in stats and stats['numeric_stats']:
        st.subheader("数值列统计")
        numeric_stats_df = pd.DataFrame(stats['numeric_stats'])
        st.dataframe(
            numeric_stats_df.round(4),
            use_container_width=True
        )


def render_field_names(df: pd.DataFrame) -> None:
    """
    渲染字段名列表
    
    Args:
        df: 数据 DataFrame
    """
    st.header("🏷️ 字段名列表")
    
    # 创建字段名表格
    field_info = []
    for i, col in enumerate(df.columns):
        field_info.append({
            "序号": i + 1,
            "字段名": col,
            "数据类型": str(df[col].dtype),
            "非空值": int(df[col].notna().sum()),
            "唯一值": int(df[col].nunique())
        })
    
    field_df = pd.DataFrame(field_info)
    st.dataframe(
        field_df,
        use_container_width=True,
        height=300
    )
    
    # 导出字段名
    st.download_button(
        label="📥 导出字段名列表",
        data="\n".join(df.columns),
        file_name="field_names.txt",
        mime="text/plain"
    )


def render_data_management_page() -> None:
    """
    数据管理主页面渲染
    """
    st.set_page_config(
        page_title="数据管理",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 数据管理")
    st.markdown("---")
    
    # 上传数据
    uploaded_df = render_data_upload_section()
    
    if uploaded_df is not None:
        # 保存到 session state
        st.session_state.uploaded_data = uploaded_df
        st.session_state.data_statistics = calculate_data_statistics(uploaded_df)
        
        st.markdown("---")
        
        # 创建选项卡
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 数据预览",
            "🏷️ 字段列表",
            "📈 统计信息",
            "💾 导出数据"
        ])
        
        with tab1:
            render_data_preview(uploaded_df)
        
        with tab2:
            render_field_names(uploaded_df)
        
        with tab3:
            render_data_statistics(uploaded_df)
        
        with tab4:
            st.header("导出数据")
            col1, col2 = st.columns(2)
            
            with col1:
                create_download_button(
                    uploaded_df,
                    "exported_data.csv",
                    "下载 CSV 格式"
                )
            
            with col2:
                from utils.file_handler import create_excel_download_button
                create_excel_download_button(
                    uploaded_df,
                    "exported_data.xlsx",
                    "下载 Excel 格式"
                )
