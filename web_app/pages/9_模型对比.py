"""
模型对比页面

功能:
- 选择多个已训练模型
- 对比指标 (AUC, Precision, Recall, F1, 准确率)
- ROC 曲线对比图
- PR 曲线对比图
- 特征重要性对比
- 预测结果对比
- 导出对比报告
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_state import init_session_state
from components.model_comparison import (
    render_model_comparison_section,
    compare_models,
    create_metrics_comparison_chart,
    create_roc_comparison_chart,
    create_pr_comparison_chart,
    create_model_summary_table,
    export_comparison_report
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="模型对比",
    page_icon="⚖️",
    layout="wide"
)

# 标题
st.title("⚖️ 模型对比")
st.markdown("---")

# ============================================================================
# 检查前置条件
# ============================================================================

if st.session_state.get('uploaded_data') is None:
    st.warning("⚠️ 请先在【数据管理】页面上传数据")
    if st.button("📊 前往数据管理"):
        st.switch_page("pages/1_数据管理.py")
    st.stop()

if st.session_state.get('field_config') is None:
    st.warning("⚠️ 请先在【配置管理】页面配置字段")
    if st.button("⚙️ 前往配置管理"):
        st.switch_page("pages/2_配置管理.py")
    st.stop()

# ============================================================================
# 获取已训练模型列表
# ============================================================================

st.header("📦 已训练模型")

model_dir = st.session_state.get('model_dir', Path('model_code/saved'))

# 扫描模型文件
model_files = list(model_dir.glob("*.joblib"))

# 过滤出模型文件 (排除 train_summary)
model_files = [f for f in model_files if 'train_summary' not in f.name]

if not model_files:
    st.warning("未找到已训练的模型，请先在【模型训练】页面训练模型")
    if st.button("🚀 前往模型训练"):
        st.switch_page("pages/4_模型训练.py")
    st.stop()

# 构建模型信息列表
available_models = []

for model_path in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
    try:
        # 尝试加载训练摘要
        summary_name = f"train_summary_{model_path.stem.split('_')[-1]}.joblib"
        summary_path = model_dir / summary_name
        
        model_info = {
            'name': model_path.stem,
            'path': str(model_path),
            'modified': datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        }
        
        if summary_path.exists():
            summary = joblib.load(summary_path)
            model_info['auc'] = summary.get('test_auc', 0)
            model_info['f1'] = summary.get('best_f1', 0)
            model_info['params'] = summary.get('best_params', {})
        else:
            model_info['auc'] = 0
            model_info['f1'] = 0
            model_info['params'] = {}
        
        available_models.append(model_info)
    
    except Exception as e:
        st.warning(f"加载模型信息失败 {model_path.name}: {str(e)}")

# 显示模型列表
st.subheader(f"找到 {len(available_models)} 个模型")

# 模型列表表格
model_df = pd.DataFrame(available_models)

if len(model_df) > 0:
    display_cols = ['name', 'modified']
    if 'auc' in model_df.columns:
        display_cols.append('auc')
    if 'f1' in model_df.columns:
        display_cols.append('f1')
    
    st.dataframe(
        model_df[display_cols].rename(columns={
            'name': '模型名称',
            'modified': '修改时间',
            'auc': 'ROC-AUC',
            'f1': 'F1 分数'
        }),
        use_container_width=True,
        height=200
    )

# ============================================================================
# 准备测试数据
# ============================================================================

st.markdown("---")
st.header("📊 测试数据准备")

# 检查是否有训练结果
if st.session_state.get('training_results') is not None:
    training_results = st.session_state.training_results
    
    # 使用训练时的测试集
    if 'X_test' in training_results and 'y_test' in training_results:
        st.info("✅ 使用当前训练会话的测试集")
        
        # 重建测试集
        from sklearn.model_selection import train_test_split
        
        df = st.session_state.uploaded_data
        config_df = st.session_state.field_config
        
        label_col = config_df[config_df['field_type'] == 'label']['field_name'].values[0]
        key_col = config_df[config_df['field_type'] == 'key']['field_name'].values
        
        if key_col:
            key_col = key_col[0]
            feature_cols = [c for c in df.columns if c not in [label_col, key_col]]
        else:
            feature_cols = [c for c in df.columns if c != label_col]
        
        X = df[feature_cols]
        y = df[label_col]
        
        # 使用相同的随机种子划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("测试集样本数", len(y_test))
        with col2:
            st.metric("正样本比例", f"{y_test.mean():.2%}")
    
    else:
        st.warning("训练结果中未找到测试集数据")
        st.info("将使用当前数据重新划分测试集")
        
        # 重新划分
        df = st.session_state.uploaded_data
        config_df = st.session_state.field_config
        
        label_col = config_df[config_df['field_type'] == 'label']['field_name'].values[0]
        key_col = config_df[config_df['field_type'] == 'key']['field_name'].values
        
        if key_col:
            key_col = key_col[0]
            feature_cols = [c for c in df.columns if c not in [label_col, key_col]]
        else:
            feature_cols = [c for c in df.columns if c != label_col]
        
        X = df[feature_cols]
        y = df[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
else:
    # 没有训练结果，使用当前数据
    st.info("使用当前上传的数据划分测试集")
    
    df = st.session_state.uploaded_data
    config_df = st.session_state.field_config
    
    label_col = config_df[config_df['field_type'] == 'label']['field_name'].values[0]
    key_col = config_df[config_df['field_type'] == 'key']['field_name'].values
    
    if key_col:
        key_col = key_col[0]
        feature_cols = [c for c in df.columns if c not in [label_col, key_col]]
    else:
        feature_cols = [c for c in df.columns if c != label_col]
    
    X = df[feature_cols]
    y = df[label_col]
    
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("随机种子", 0, 10000, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("测试集样本数", len(y_test))
    with col2:
        st.metric("正样本比例", f"{y_test.mean():.2%}")

# ============================================================================
# 模型对比
# ============================================================================

st.markdown("---")

if st.session_state.get('X_test') is not None and st.session_state.get('y_test') is not None:
    render_model_comparison_section(
        available_models,
        st.session_state.X_test,
        st.session_state.y_test
    )

# ============================================================================
# 显示已保存的对比结果
# ============================================================================

if st.session_state.get('comparison_results') is not None:
    st.markdown("---")
    st.header("📋 对比详情")
    
    results = st.session_state.comparison_results
    
    # 详细指标
    st.subheader("详细性能指标")
    
    for model_name, result in results.items():
        with st.expander(f"📊 {model_name}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ROC-AUC", f"{result['auc']:.6f}")
                st.metric("精确率", f"{result['precision']:.6f}")
            
            with col2:
                st.metric("F1 分数", f"{result['f1']:.6f}")
                st.metric("召回率", f"{result['recall']:.6f}")
            
            with col3:
                st.metric("准确率", f"{result['accuracy']:.6f}")
                st.metric("AUC-PR", f"{result['auc_pr']:.6f}")
            
            # 混淆矩阵
            st.write("**混淆矩阵:**")
            cm_df = pd.DataFrame(
                result['confusion_matrix'],
                index=['真实负', '真实正'],
                columns=['预测负', '预测正']
            )
            st.dataframe(cm_df, use_container_width=True)

# ============================================================================
# 快捷操作
# ============================================================================

st.markdown("---")
st.subheader("⚡ 快捷操作")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 刷新模型列表", use_container_width=True):
        if 'comparison_results' in st.session_state:
            del st.session_state.comparison_results
        st.rerun()

with col2:
    if st.button("📦 前往模型管理", use_container_width=True):
        st.switch_page("pages/5_模型管理.py")

with col3:
    if st.button("🚀 训练新模型", use_container_width=True):
        st.switch_page("pages/4_模型训练.py")
