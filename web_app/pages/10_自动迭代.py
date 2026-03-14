"""
自动迭代训练页面

功能:
- 设置迭代次数/停止条件
- 自动调整超参数
- 自动特征选择
- 自动采样比例调整
- 迭代进度跟踪
- 最佳模型自动保存
- 迭代历史可视化
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state
from model_code.auto_train import AutoTrainer

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="自动迭代训练",
    page_icon="🔄",
    layout="wide"
)

# 标题
st.title("🔄 自动迭代训练")
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
# 配置区域
# ============================================================================

st.header("⚙️ 迭代配置")

col1, col2, col3 = st.columns(3)

with col1:
    max_iterations = st.slider(
        "最大迭代次数",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    
    early_stopping_rounds = st.slider(
        "提前停止轮数",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

with col2:
    min_improvement = st.number_input(
        "最小改进阈值",
        min_value=0.0001,
        max_value=0.1,
        value=0.001,
        step=0.0001,
        format="%.4f"
    )
    
    test_size = st.slider(
        "测试集比例",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05
    )

with col3:
    random_state = st.number_input(
        "随机种子",
        min_value=0,
        max_value=10000,
        value=42,
        step=1
    )

# 高级选项
with st.expander("🔧 高级选项"):
    col1, col2 = st.columns(2)
    
    with col1:
        skip_hyperopt = st.checkbox(
            "跳过超参数优化",
            value=False,
            help="如果已调优过参数，可跳过此步骤"
        )
        
        skip_feature_selection = st.checkbox(
            "跳过特征选择",
            value=False,
            help="使用全部特征进行训练"
        )
    
    with col2:
        n_hyperopt_iter = st.number_input(
            "超参数优化迭代次数",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        max_features = st.number_input(
            "最大特征数",
            min_value=5,
            max_value=200,
            value=50,
            step=5
        )

# ============================================================================
# 模型参数配置
# ============================================================================

st.header("🎯 模型参数")

# 使用当前会话的模型参数或默认值
default_params = st.session_state.get('model_params', {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 50
})

col1, col2, col3 = st.columns(3)

with col1:
    n_estimators = st.slider(
        "n_estimators",
        min_value=50,
        max_value=1000,
        value=default_params.get('n_estimators', 200),
        step=50
    )
    
    learning_rate = st.slider(
        "learning_rate",
        min_value=0.001,
        max_value=0.3,
        value=default_params.get('learning_rate', 0.1),
        step=0.01,
        format="%.3f"
    )

with col2:
    max_depth = st.slider(
        "max_depth",
        min_value=3,
        max_value=15,
        value=default_params.get('max_depth', 6),
        step=1
    )
    
    num_leaves = st.slider(
        "num_leaves",
        min_value=10,
        max_value=200,
        value=default_params.get('num_leaves', 50),
        step=10
    )

with col3:
    subsample = st.slider(
        "subsample",
        min_value=0.5,
        max_value=1.0,
        value=default_params.get('subsample', 0.9),
        step=0.05,
        format="%.2f"
    )
    
    colsample_bytree = st.slider(
        "colsample_bytree",
        min_value=0.5,
        max_value=1.0,
        value=default_params.get('colsample_bytree', 0.9),
        step=0.05,
        format="%.2f"
    )

model_params = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'num_leaves': num_leaves,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree
}

# ============================================================================
# 训练控制
# ============================================================================

st.markdown("---")
st.header("🚀 训练控制")

# 准备数据
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

# 编码标签列为数字 (用于计算正样本比例)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 显示数据信息
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("样本数", f"{len(df):,}")

with col2:
    st.metric("特征数", len(feature_cols))

with col3:
    st.metric("正样本比例", f"{y_encoded.mean():.2%}")

# 训练按钮
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    start_training = st.button(
        "🚀 开始自动迭代训练",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get('auto_training_in_progress', False)
    )

with col2:
    if st.button(
        "⏸️ 中断训练",
        use_container_width=True,
        disabled=not st.session_state.get('auto_training_in_progress', False)
    ):
        if 'auto_trainer' in st.session_state:
            st.session_state.auto_trainer.interrupt()
        st.session_state.auto_training_in_progress = False
        st.warning("训练已中断")

with col3:
    if st.button(
        "🗑️ 清除历史",
        use_container_width=True,
        disabled=st.session_state.get('auto_train_results') is None
    ):
        st.session_state.auto_train_results = None
        st.session_state.auto_train_history = None
        st.rerun()

# ============================================================================
# 执行训练
# ============================================================================

if start_training:
    st.session_state.auto_training_in_progress = True
    st.session_state.auto_train_logs = []
    
    # 创建配置
    config = {
        'X': X,
        'y': y,
        'model_params': model_params,
        'test_size': test_size,
        'random_state': random_state,
        'skip_hyperopt': skip_hyperopt,
        'skip_feature_selection': skip_feature_selection
    }
    
    # 创建训练器
    trainer = AutoTrainer(
        base_config=config,
        max_iterations=max_iterations,
        early_stopping_rounds=early_stopping_rounds,
        min_improvement=min_improvement,
        model_dir=st.session_state.get('model_dir', Path('model_code/saved'))
    )
    
    st.session_state.auto_trainer = trainer
    
    # 创建进度显示区域
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    try:
        # 运行训练
        for iteration in range(1, max_iterations + 1):
            if trainer.interrupted:
                break
            
            # 更新进度
            progress = iteration / max_iterations
            progress_placeholder.progress(progress, text=f"迭代 {iteration}/{max_iterations}")
            
            # 执行单次迭代
            with log_placeholder.container():
                result = trainer.train_iteration(iteration, trainer.best_params.copy())
                
                # 显示日志
                st.code(f"""
迭代 {iteration}/{max_iterations}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROC-AUC: {result['auc']:.6f}
F1 分数：{result['f1']:.6f}
最佳 AUC: {trainer.best_score:.6f} (迭代 {trainer.best_iteration})
                """)
            
            # 实时更新图表
            with chart_placeholder.container():
                fig = trainer.plot_history()
                st.plotly_chart(fig, use_container_width=True)
            
            # 检查提前停止
            if trainer.check_early_stopping():
                st.info("触发提前停止条件")
                break
            
            # 超参数优化 (第一次迭代后)
            if iteration == 1 and not skip_hyperopt:
                st.info("正在进行超参数优化...")
                trainer.optimize_hyperparameters(n_iter=n_hyperopt_iter)
            
            # 特征选择 (第一次迭代后)
            if iteration == 1 and not skip_feature_selection:
                st.info("正在进行特征选择...")
                selected_features = trainer.feature_selection(max_features=max_features)
                trainer.base_config['X'] = trainer.base_config['X'][selected_features]
                st.success(f"选中 {len(selected_features)} 个特征")
        
        # 训练完成
        st.session_state.auto_training_in_progress = False
        
        # 保存结果
        results = {
            'history': trainer.history,
            'best_iteration': trainer.best_iteration,
            'best_score': trainer.best_score,
            'best_params': trainer.best_params,
            'total_iterations': len(trainer.history),
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.auto_train_results = results
        st.session_state.auto_train_history = trainer.get_history_dataframe()
        
        # 保存最佳模型
        if trainer.best_model:
            model_path = st.session_state.get('model_dir', Path('model_code/saved')) / f"auto_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(trainer.best_model, model_path)
            st.success(f"✅ 最佳模型已保存：{model_path.name}")
        
        st.success("🎉 自动迭代训练完成!")
        st.rerun()
        
    except Exception as e:
        st.session_state.auto_training_in_progress = False
        st.error(f"❌ 训练失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# 显示结果
# ============================================================================

if st.session_state.get('auto_train_results') is not None:
    st.markdown("---")
    st.header("📊 训练结果")
    
    results = st.session_state.auto_train_results
    history_df = st.session_state.auto_train_history
    
    # 核心指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总迭代次数", results['total_iterations'])
    
    with col2:
        st.metric("最佳迭代", results['best_iteration'])
    
    with col3:
        st.metric("最佳 AUC", f"{results['best_score']:.6f}")
    
    with col4:
        st.metric("完成时间", results['timestamp'].split('T')[0])
    
    # 训练历史图表
    st.subheader("📈 训练历史")
    
    if history_df is not None and not history_df.empty:
        # 指标趋势图
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['auc'],
            mode='lines+markers',
            name='ROC-AUC',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['f1'],
            mode='lines+markers',
            name='F1 分数',
            line=dict(color='green', width=3)
        ))
        
        fig.add_hline(
            y=results['best_score'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"最佳 AUC: {results['best_score']:.4f}"
        )
        
        fig.update_layout(
            title='迭代训练趋势',
            xaxis_title='迭代次数',
            yaxis_title='分数',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 历史数据表格
        st.subheader("📋 迭代历史详情")
        st.dataframe(
            history_df.round(6),
            use_container_width=True,
            height=300
        )
    
    # 最佳参数
    st.subheader("🎯 最佳模型参数")
    
    best_params = results.get('best_params', {})
    if best_params:
        params_df = pd.DataFrame({
            '参数': list(best_params.keys()),
            '值': [str(v) for v in best_params.values()]
        })
        st.dataframe(params_df, use_container_width=True)
    else:
        st.info("无参数信息")
    
    # 导出结果
    st.subheader("💾 导出结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if history_df is not None and not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 下载训练历史 (CSV)",
                data=csv,
                file_name=f"auto_train_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # 导出报告
        report = f"""
自动迭代训练报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

训练时间：{results['timestamp']}
总迭代次数：{results['total_iterations']}
最佳迭代：{results['best_iteration']}
最佳 AUC: {results['best_score']:.6f}

最佳参数:
{best_params}
        """
        st.download_button(
            label="📥 下载训练报告",
            data=report,
            file_name=f"auto_train_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================================
# 导入 Plotly
# ============================================================================

# 延迟导入以避免未使用警告
if st.session_state.get('auto_train_results') is not None:
    import plotly.graph_objects as go

# ============================================================================
# 快捷操作
# ============================================================================

st.markdown("---")
st.subheader("⚡ 快捷操作")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📦 前往模型管理", use_container_width=True):
        st.switch_page("pages/5_模型管理.py")

with col2:
    if st.button("⚖️ 模型对比", use_container_width=True):
        st.switch_page("pages/9_模型对比.py")

with col3:
    if st.button("🚀 标准训练", use_container_width=True):
        st.switch_page("pages/4_模型训练.py")
