"""
手动训练流程入口页。

合并自原 pages/1_数据管理.py、2_配置管理.py、3_数据分析.py、
4_模型训练.py、5_模型管理.py、6_模型预测.py、7_结果分析.py、
8_数据源管理.py、9_模型对比.py。

设计文档：docs/superpowers/specs/2026-05-01-manual-training-merge-design.md
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state

st.set_page_config(
    page_title="手动训练流程",
    page_icon="📚",
    layout="wide",
)

init_session_state()

st.title("📚 手动训练流程")
st.caption(
    "数据准备 → 字段配置 → 数据探索 → 模型训练 → 模型管理与对比 → 预测与结果"
)
st.markdown("---")

TAB_LABELS = [
    "1️⃣ 数据准备",
    "2️⃣ 字段配置",
    "3️⃣ 数据探索",
    "4️⃣ 模型训练",
    "5️⃣ 模型管理与对比",
    "6️⃣ 预测与结果",
]

tab_data_ui, tab_config_ui, tab_eda_ui, tab_train_ui, tab_model_ui, tab_predict_ui = st.tabs(TAB_LABELS)

with tab_data_ui:
    from web_app.manual_training import tab_data
    tab_data.render()
with tab_config_ui:
    from web_app.manual_training import tab_config
    tab_config.render()
with tab_eda_ui:
    from web_app.manual_training import tab_eda
    tab_eda.render()
with tab_train_ui:
    from web_app.manual_training import tab_train
    tab_train.render()
with tab_model_ui:
    from web_app.manual_training import tab_model_mgmt
    tab_model_mgmt.render()
with tab_predict_ui:
    from web_app.manual_training import tab_predict
    tab_predict.render()
