"""
数据管理页面

功能：
- 数据上传（CSV、Excel）
- 数据预览
- 字段名展示
- 数据统计信息
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state
from web_app.components.data_upload import (
    render_data_upload_section,
    render_data_preview,
    render_data_statistics,
    render_field_names,
)
from web_app.utils.file_handler import (
    calculate_data_statistics,
    create_download_button,
    create_excel_download_button,
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="数据管理",
    page_icon="📊",
    layout="wide"
)

# 标题
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
        st.header("💾 导出数据")
        col1, col2 = st.columns(2)
        
        with col1:
            create_download_button(
                uploaded_df,
                "exported_data.csv",
                "📥 下载 CSV 格式"
            )
        
        with col2:
            create_excel_download_button(
                uploaded_df,
                "exported_data.xlsx",
                "📥 下载 Excel 格式"
            )
        
        st.info("💡 上传的数据将自动保存到会话状态，可在其他页面使用")

else:
    # 显示已有数据
    if st.session_state.get('uploaded_data') is not None:
        st.success("✅ 已加载之前上传的数据")
        
        df = st.session_state.uploaded_data
        
        st.markdown("---")
        
        # 创建选项卡
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 数据预览",
            "🏷️ 字段列表",
            "📈 统计信息",
            "💾 导出数据"
        ])
        
        with tab1:
            render_data_preview(df)
        
        with tab2:
            render_field_names(df)
        
        with tab3:
            render_data_statistics(df)
        
        with tab4:
            st.header("💾 导出数据")
            col1, col2 = st.columns(2)
            
            with col1:
                create_download_button(
                    df,
                    "exported_data.csv",
                    "📥 下载 CSV 格式"
                )
            
            with col2:
                create_excel_download_button(
                    df,
                    "exported_data.xlsx",
                    "📥 下载 Excel 格式"
                )
        
        # 清除数据按钮
        st.markdown("---")
        if st.button("🗑️ 清除已加载数据"):
            st.session_state.uploaded_data = None
            st.session_state.data_statistics = None
            st.rerun()
    
    else:
        st.info("👆 请上传数据文件开始分析")
        
        st.markdown("""
        ### 📝 使用说明
        
        1. **支持格式**: CSV、Excel (.xlsx, .xls)
        2. **编码支持**: 自动检测 UTF-8、GBK 等编码
        3. **数据预览**: 上传后可编辑查看前 100 行
        4. **统计分析**: 自动计算缺失率、唯一值、分布等
        
        ### 💡 提示
        
        - 上传的数据会保存在当前会话中
        - 可在其他页面（配置管理、模型训练等）使用已上传的数据
        - 大文件上传可能需要一些时间，请耐心等待
        """)
