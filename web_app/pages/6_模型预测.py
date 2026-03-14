"""
模型预测页面

功能：
- 上传预测数据
- 选择已训练模型
- 预测进度展示
- 预测结果下载
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state
from web_app.utils.file_handler import read_uploaded_file, create_download_button
from web_app.components.progress import render_prediction_progress

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="模型预测",
    page_icon="🔮",
    layout="wide"
)

# 标题
st.title("🔮 模型预测")
st.markdown("---")

# ============================================================================
# 选择模型
# ============================================================================

st.header("📦 选择模型")

def scan_saved_models() -> list:
    """扫描模型目录中的所有模型文件"""
    model_dir = st.session_state.model_dir
    models = []
    
    if model_dir.exists():
        for file_path in model_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(file_path)
                
                model_info = {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'timestamp': file_path.stat().st_mtime,
                }
                
                if isinstance(model_data, dict) and 'test_auc' in model_data:
                    model_info['auc'] = model_data['test_auc']
                
                models.append(model_info)
            except:
                pass
    
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models

models = scan_saved_models()

if not models:
    st.warning("⚠️ 暂无已训练的模型，请先在【模型训练】页面训练模型")
    
    if st.button("🚀 前往模型训练"):
        st.switch_page("pages/4_模型训练.py")
    st.stop()

# 模型选择器
col1, col2 = st.columns([2, 1])

with col1:
    selected_model_name = st.selectbox(
        "选择要使用的模型",
        [m['name'] for m in models],
        format_func=lambda x: f"{x}"
    )

with col2:
    # 显示模型信息
    if selected_model_name:
        selected_model = next(m for m in models if m['name'] == selected_model_name)
        if 'auc' in selected_model:
            st.info(f"AUC: {selected_model['auc']:.4f}")

# 加载模型
if selected_model_name:
    selected_model = next(m for m in models if m['name'] == selected_model_name)
    
    try:
        model_data = joblib.load(selected_model['path'])
        
        if isinstance(model_data, dict):
            model = model_data.get('model')
            threshold = model_data.get('best_threshold', 0.5)
            feature_names = model_data.get('feature_names', None)
        else:
            model = model_data
            threshold = 0.5
            feature_names = None
        
        st.success(f"✅ 模型已加载：{selected_model_name}")
        
    except Exception as e:
        st.error(f"加载模型失败：{str(e)}")
        st.stop()
else:
    st.stop()

st.markdown("---")

# ============================================================================
# 上传预测数据
# ============================================================================

st.header("📤 上传预测数据")

uploaded_file = st.file_uploader(
    "上传用于预测的数据文件 (CSV/Excel)",
    type=["csv", "xlsx", "xls"],
    help="数据格式应与训练数据一致"
)

if uploaded_file is not None:
    predict_df = read_uploaded_file(uploaded_file)
    
    if predict_df is not None:
        st.session_state.predict_data = predict_df
        
        st.success(f"✅ 数据已上传：{uploaded_file.name}")
        st.info(f"📊 数据大小：{predict_df.shape[0]:,} 行 × {predict_df.shape[1]} 列")
        
        # 数据预览
        with st.expander("📋 数据预览"):
            st.dataframe(predict_df.head(), use_container_width=True)

if st.session_state.get('predict_data') is None:
    st.info("👆 请上传预测数据")
    st.stop()

st.markdown("---")

# ============================================================================
# 预测配置
# ============================================================================

st.header("⚙️ 预测配置")

col1, col2, col3 = st.columns(3)

with col1:
    threshold = st.slider(
        "分类阈值",
        min_value=0.1,
        max_value=0.9,
        value=threshold,
        step=0.05,
        format="%.2f"
    )
    st.session_state.prediction_threshold = threshold

with col2:
    include_proba = st.checkbox(
        "包含预测概率",
        value=True,
        help="在结果中包含预测概率值"
    )

with col3:
    batch_size = st.number_input(
        "批处理大小",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="大数据集时分批预测"
    )

# ============================================================================
# 执行预测
# ============================================================================

st.markdown("---")
st.header("🔮 预测控制")

col1, col2 = st.columns([2, 1])

with col1:
    predict_df = st.session_state.predict_data
    st.info(f"""
    📊 预测数据：{len(predict_df):,} 行  
    🎯 分类阈值：{threshold:.2f}  
    📦 模型：{selected_model_name}
    """)

with col2:
    start_prediction = st.button(
        "🚀 开始预测",
        type="primary",
        use_container_width=True
    )

if start_prediction:
    try:
        # 创建进度占位符
        progress_placeholder = st.empty()
        
        render_prediction_progress(0.1, "数据预处理...")
        
        # 数据预处理（与训练时保持一致）
        from model_code.utils import (
            handle_dataframe_outliers,
            handle_dataframe_missing_values,
            encode_dataframe,
            standardize_dataframe,
        )
        
        # 加载配置
        config_df = st.session_state.get('field_config')
        
        if config_df is None:
            st.error("❌ 未找到字段配置，请先在【配置管理】页面配置字段")
            st.stop()
        
        # 复制数据
        pred_data = predict_df.copy()
        
        render_prediction_progress(0.3, "异常值处理...")
        pred_data = handle_dataframe_outliers(pred_data, config_df)
        
        render_prediction_progress(0.5, "缺失值处理...")
        pred_data = handle_dataframe_missing_values(pred_data, config_df)
        
        render_prediction_progress(0.7, "编码和标准化...")
        pred_data = encode_dataframe(pred_data, config_df)
        pred_data = standardize_dataframe(pred_data, config_df)
        
        render_prediction_progress(0.85, "模型预测...")
        
        # 准备特征
        # 先获取 key_col（在 try 块外使用）
        key_col = config_df[config_df['field_type'] == 'key']['field_name'].values
        
        if feature_names:
            # 确保特征列顺序一致
            X_pred = pred_data[feature_names]
        else:
            # 移除 label 和 key 列
            label_col = config_df[config_df['field_type'] == 'label']['field_name'].values
            
            exclude_cols = []
            if len(label_col) > 0:
                exclude_cols.append(label_col[0])
            if len(key_col) > 0:
                exclude_cols.append(key_col[0])
            
            X_pred = pred_data[[c for c in pred_data.columns if c not in exclude_cols]]
        
        # 执行预测
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_pred)[:, 1]
        else:
            y_pred_proba = model.predict(X_pred)
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        render_prediction_progress(1.0, "模型预测完成！")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            '预测概率': y_pred_proba,
            '预测标签': y_pred,
        })
        
        # 添加原始 ID 列（如果有）
        if len(key_col) > 0 and key_col[0] in predict_df.columns:
            results_df.insert(0, key_col[0], predict_df[key_col[0]].values)
        
        st.session_state.predict_results = results_df
        st.session_state.prediction_completed = True
        
        st.success("✅ 预测完成！")
        st.rerun()
        
    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# 显示预测结果
# ============================================================================

if st.session_state.get('prediction_completed'):
    st.markdown("---")
    st.header("📊 预测结果")
    
    results_df = st.session_state.predict_results
    
    # 结果统计
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总样本数", f"{len(results_df):,}")
    
    with col2:
        pred_positive = int((results_df['预测标签'] == 1).sum())
        st.metric("预测正例", f"{pred_positive:,}")
    
    with col3:
        pred_negative = int((results_df['预测标签'] == 0).sum())
        st.metric("预测负例", f"{pred_negative:,}")
    
    with col4:
        positive_rate = pred_positive / len(results_df) * 100
        st.metric("正例比例", f"{positive_rate:.2f}%")
    
    # 结果预览
    st.subheader("📋 结果预览")
    st.dataframe(
        results_df.head(100),
        use_container_width=True,
        height=300
    )
    
    # 预测分布图
    st.subheader("📊 预测概率分布")
    
    import plotly.express as px
    
    fig = px.histogram(
        results_df,
        x='预测概率',
        color='预测标签',
        nbins=50,
        title='预测概率分布',
        labels={'预测概率': '预测概率', 'count': '样本数'},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    
    # 添加阈值线
    fig.add_shape(
        type="line",
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=results_df['预测标签'].value_counts().max(),
        line=dict(color="green", width=3, dash="dash"),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 导出结果
    st.subheader("💾 导出结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 下载 CSV 格式",
            data=csv,
            file_name=f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        from web_app.utils.file_handler import create_excel_download_button
        create_excel_download_button(
            results_df,
            f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "📥 下载 Excel 格式"
        )
    
    # 清除结果按钮
    st.markdown("---")
    if st.button("🗑️ 清除预测结果"):
        st.session_state.predict_results = None
        st.session_state.prediction_completed = False
        st.rerun()

else:
    st.info("👆 点击【开始预测】按钮执行预测")
