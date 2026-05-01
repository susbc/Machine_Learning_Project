"""
tab_config.py — 字段配置 Tab

合并自：web_app/pages/2_配置管理.py
"""

import streamlit as st


def render() -> None:
    """渲染字段配置 Tab。"""
    from web_app.components.config_editor import (
        render_config_editor,
        render_config_actions,
        render_field_type_distribution,
        render_config_summary,
        auto_infer_config,
    )
    from web_app.utils.file_handler import read_file_from_path

    st.header("⚙️ 字段配置")

    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        st.info("💡 字段配置需要基于已上传的数据生成")
        return

    if st.session_state.get("field_config") is None:
        config_path = st.session_state.config_dir / "field_config.xlsx"
        if config_path.exists():
            st.session_state.field_config = read_file_from_path(str(config_path))
            st.session_state.config_file_path = str(config_path)
            st.success(f"✅ 已加载现有配置：{config_path}")
        else:
            st.session_state.field_config = auto_infer_config(
                st.session_state.uploaded_data
            )
            st.info("✨ 已根据数据自动生成配置，您可以进行调整")

    config_df = st.session_state.field_config

    if st.session_state.get("config_file_path"):
        st.caption(f"📁 当前配置文件：{st.session_state.config_file_path}")

    st.subheader("📝 配置编辑")
    edited_config = render_config_editor(config_df)

    if not edited_config.equals(config_df):
        st.session_state.field_config = edited_config
        st.session_state.config_modified = True
        st.info("✏️ 配置已修改，记得保存！")

    st.markdown("---")
    render_config_actions(edited_config, st.session_state.get("config_file_path"))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        render_field_type_distribution(edited_config)
    with col2:
        render_config_summary(edited_config)

    with st.expander("📖 配置说明"):
        st.markdown(
            """
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
        """
        )
