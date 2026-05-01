"""
配置管理页面

功能：
- field_config.csv 在线编辑
- 字段类型、异常值处理、缺失值填充配置
- 配置保存/加载
- 配置导出/导入
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state
from web_app.components.config_editor import (
    render_config_editor,
    render_config_actions,
    render_field_type_distribution,
    render_config_summary,
    auto_infer_config,
)
from web_app.utils.file_handler import (
    read_file_from_path,
    save_config_to_excel,
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="配置管理",
    page_icon="⚙️",
    layout="wide"
)

# 标题
st.title("⚙️ 配置管理")
st.markdown("---")

# 检查是否有数据
if st.session_state.get('uploaded_data') is None:
    st.warning("⚠️ 请先在【数据管理】页面上传数据")
    st.info("💡 配置管理需要基于已上传的数据生成字段配置")
    
    if st.button("📊 前往数据管理"):
        st.switch_page("pages/1_数据管理.py")
else:
    # 初始化或加载配置
    if st.session_state.get('field_config') is None:
        # 尝试加载现有配置
        config_path = st.session_state.config_dir / "field_config.xlsx"
        if config_path.exists():
            st.session_state.field_config = read_file_from_path(str(config_path))
            st.session_state.config_file_path = str(config_path)
            st.success(f"✅ 已加载现有配置：{config_path}")
        else:
            # 自动推断配置
            st.session_state.field_config = auto_infer_config(
                st.session_state.uploaded_data
            )
            st.info("✨ 已根据数据自动生成配置，您可以进行调整")
    
    config_df = st.session_state.field_config
    
    # 显示当前配置的文件路径
    if st.session_state.get('config_file_path'):
        st.caption(f"📁 当前配置文件：{st.session_state.config_file_path}")
    
    # 渲染编辑器
    st.header("📝 配置编辑")
    edited_config = render_config_editor(config_df)
    
    # 检测配置变化
    if not edited_config.equals(config_df):
        st.session_state.field_config = edited_config
        st.session_state.config_modified = True
        st.info("✏️ 配置已修改，记得保存！")
    
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
    
    # 配置说明
    with st.expander("📖 配置说明"):
        st.markdown("""
        ### 字段配置说明
        
        #### field_type (字段类型)
        - **num**: 数值型字段，用于模型特征
        - **str**: 分类型字段，需要编码处理
        - **key**: 标识字段（如用户 ID），不参与训练
        - **label**: 目标变量（标签），用于监督学习
        
        #### outlier_value (异常值处理)
        - **3std**: 使用 3 倍标准差裁剪
        - **5std**: 使用 5 倍标准差裁剪
        - **(min,max)**: 使用固定范围裁剪，如 (0,100)
        - **空**: 不处理异常值
        
        #### missing_value (缺失值填充)
        - **mean**: 使用均值填充
        - **median**: 使用中位数填充
        - **mode**: 使用众数填充
        - **0/-1**: 使用固定值填充
        - **空**: 不填充
        
        #### ecode_type (编码方式)
        - **one-hot**: 独热编码（适合无序分类）
        - **ordinal**: 顺序编码（适合有序分类）
        - **空**: 不编码（数值型字段）
        
        #### standardize_type (标准化)
        - **z-score**: 标准差标准化（均值 0，方差 1）
        - **min-max**: 最小 - 最大标准化（缩放到 0-1）
        - **robust**: 稳健标准化（使用四分位数）
        - **空**: 不标准化
        """)
