"""
自动迭代训练页面 - 升级版

功能:
- 文件上传（支持独立使用，无需数据管理页面）
- 数据预览与统计
- 预处理开关（缺失值/异常值/编码/标准化）
- 算法选择（自动推荐 / 指定算法）
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
from model_code.auto_train import AutoTrainer, auto_generate_field_config
from model_code.model_factory import ModelFactory
from model_code.utils import ConfigDrivenPreprocessor

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
# 数据源选择：Session 模式 or 独立上传模式
# ============================================================================

has_session_data = (
    st.session_state.get('uploaded_data') is not None and
    st.session_state.get('field_config') is not None
)

if has_session_data:
    data_source = st.radio(
        "数据来源",
        options=["使用数据管理页面的数据", "上传新文件"],
        horizontal=True,
        label_visibility="collapsed",
        index=0,
    )
    use_session = (data_source == "使用数据管理页面的数据")
else:
    use_session = False

# ============================================================================
# 独立上传模式
# ============================================================================

standalone_df = None
standalone_info = None
standalone_label_col = None

if not use_session:
    st.header("1. 数据上传")

    uploaded_file = st.file_uploader(
        "上传数据文件（支持 CSV、Excel 格式）",
        type=["csv", "xlsx", "xls"],
        help="CSV 文件自动检测分隔符（逗号/制表符/分号/竖线）和编码（UTF-8/GBK）",
        key="auto_train_file_uploader"
    )

    if uploaded_file is not None:
        from web_app.utils.file_handler import read_uploaded_file, calculate_data_statistics

        standalone_df, standalone_info = read_uploaded_file(uploaded_file, return_info=True)

        if standalone_df is not None:
            # 显示文件识别信息
            with st.expander("🔍 文件识别信息", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("行数", f"{standalone_info.get('rows', 0):,}")
                c2.metric("列数", standalone_info.get('columns', 0))
                c3.metric("分隔符", standalone_info.get('delimiter_name', 'N/A'))
                c4.metric("编码", standalone_info.get('encoding', 'N/A'))

            # 目标变量列选择
            standalone_label_col = st.selectbox(
                "选择目标变量（Label）列",
                options=standalone_df.columns.tolist(),
                index=len(standalone_df.columns) - 1,
                help="选择用于监督学习的目标列",
                key="standalone_label_selector"
            )

            # 数据预览
            with st.expander("📊 数据预览", expanded=True):
                col_preview, col_stats = st.tabs(["数据预览", "列统计"])

                with col_preview:
                    st.dataframe(standalone_df.head(20), use_container_width=True)

                with col_stats:
                    stats = calculate_data_statistics(standalone_df)

                    # 缺失值分析
                    missing_data = {
                        k: v for k, v in stats.get('missing_summary', {}).items() if v > 0
                    }
                    if missing_data:
                        missing_df = pd.DataFrame({
                            '字段': list(missing_data.keys()),
                            '缺失率 (%)': [f"{v:.1f}" for v in missing_data.values()]
                        }).sort_values('缺失率 (%)', ascending=False, key=lambda x: x.astype(float))
                        st.dataframe(missing_df.head(20), use_container_width=True, height=300)
                    else:
                        st.success("数据完整，无缺失值")

                    # 数据类型分布
                    dtype_data = stats.get('dtype_summary', {})
                    if dtype_data:
                        dtype_df = pd.DataFrame({
                            '数据类型': list(dtype_data.keys()),
                            '数量': list(dtype_data.values())
                        })
                        st.dataframe(dtype_df, use_container_width=True, height=200)

            # 数据概览卡片
            st.info(
                f"**数据概览** | "
                f"样本数: {len(standalone_df):,} | "
                f"特征数: {len(standalone_df.columns) - 1} | "
                f"分隔符: {standalone_info.get('delimiter_name', 'N/A')} | "
                f"编码: {standalone_info.get('encoding', 'N/A')}"
            )

    else:
        st.info("👆 请上传数据文件开始训练，或前往【数据管理】页面使用已有数据")
        st.stop()

# ============================================================================
# 确定当前使用的 DataFrame 和配置
# ============================================================================

if use_session:
    active_df = st.session_state.uploaded_data
    active_config = st.session_state.field_config
    active_label_col = None  # Will be determined from config
else:
    active_df = standalone_df
    active_config = None  # Will be auto-generated
    active_label_col = standalone_label_col

# ============================================================================
# 算法选择
# ============================================================================

st.header("2. 算法选择")

algo_options = ["auto"] + ModelFactory.get_supported_algorithms()
algo_display = {
    "auto": "自动推荐（基于数据特征）",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "随机森林",
    "logistic_regression": "逻辑回归",
}

selected_algo = st.selectbox(
    "选择训练算法",
    options=algo_options,
    index=0,
    format_func=lambda x: algo_display[x],
    help="'自动推荐' 会根据数据特征自动选择最佳算法",
)

if selected_algo == "auto":
    from model_code.auto_selector import AutoModelSelector

    # 确定 label_col
    if use_session:
        label_col_val = active_config[active_config['field_type'] == 'label']['field_name'].values
        if len(label_col_val) > 0:
            label_col_val = label_col_val[0]
        else:
            label_col_val = active_df.columns[-1]
    else:
        label_col_val = standalone_label_col

    feature_cols_for_rec = [c for c in active_df.columns if c != label_col_val]
    X_rec = active_df[feature_cols_for_rec]
    y_rec = active_df[label_col_val]

    if y_rec.dtype == 'object':
        y_rec = pd.Series(LabelEncoder().fit_transform(y_rec), index=y_rec.index)

    rec = AutoModelSelector.recommend(X_rec, y_rec)
    st.success(f"**推荐算法**: {ModelFactory.get_display_name(rec['recommended'])} | **理由**: {rec['reason']}")
    st.info(f"备选算法: {[ModelFactory.get_display_name(a) for a in rec['alternatives'][:3]]}")

# ============================================================================
# 数据预处理选项
# ============================================================================

st.header("3. 数据预处理")

with st.expander("预处理开关", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        enable_missing = st.checkbox("缺失值处理", value=True, help="自动填充缺失值（数值型用均值/中位数，分类型用众数）")
        enable_outlier = st.checkbox("异常值处理", value=True, help="基于配置表的 outlier_value 裁剪异常值（3σ/5σ/固定范围）")

    with col2:
        enable_encoding = st.checkbox("编码处理", value=True, help="类别特征编码（独热编码/顺序编码）")
        enable_standardization = st.checkbox("标准化", value=True, help="数值特征标准化（z-score/min-max/robust）")

    # 智能抽样
    st.subheader("采样设置")
    enable_sampling = st.checkbox("启用负样本欠采样", value=False, help="通过迭代寻找符合统计特征的最优负样本子集")

    sampling_config = None
    if enable_sampling:
        neg_ratio = st.slider("负样本采样比例", 0.01, 1.0, 0.1, 0.01,
                             help="负样本保留比例，设为 0.1 表示保留 10% 的负样本")
        sampling_config = {"negative_ratio": neg_ratio, "random_state": 42}

# ============================================================================
# 迭代配置
# ============================================================================

st.header("4. 迭代配置")

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
with st.expander("高级选项"):
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

st.header("5. 模型参数")

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
# 确定 label_col 和 key_col
# ============================================================================

if use_session:
    label_col_val = active_config[active_config['field_type'] == 'label']['field_name'].values
    key_col_val = active_config[active_config['field_type'] == 'key']['field_name'].values

    label_col = label_col_val[0] if len(label_col_val) > 0 else active_df.columns[-1]
    key_col = key_col_val[0] if len(key_col_val) > 0 else None
else:
    label_col = standalone_label_col
    key_col = None

# ============================================================================
# 构建配置 + 预处理器
# ============================================================================

if not use_session:
    # 独立模式：自动生成字段配置
    working_config = auto_generate_field_config(active_df, label_col, key_col)
else:
    working_config = active_config.copy()

# 根据预处理开关更新配置
if not enable_missing:
    working_config['missing_value'] = np.nan
if not enable_outlier:
    working_config['outlier_value'] = np.nan
if not enable_encoding:
    working_config['ecode_type'] = np.nan
if not enable_standardization:
    working_config['standardize_type'] = np.nan

# 创建预处理器
preprocessor = ConfigDrivenPreprocessor(
    working_config,
    enable_missing=enable_missing,
    enable_outlier=enable_outlier,
    enable_encoding=enable_encoding,
    enable_standardization=enable_standardization,
)

# 准备 X / y
feature_cols = [c for c in active_df.columns if c not in ([label_col, key_col] if key_col else [label_col])]
X = active_df[feature_cols]
y = active_df[label_col]

# 编码标签列为数字
if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

# 显示数据信息
st.header("6. 训练控制")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("样本数", f"{len(active_df):,}")

with col2:
    st.metric("特征数", len(feature_cols))

with col3:
    st.metric("正样本比例", f"{y.mean():.2%}")

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
        'skip_feature_selection': skip_feature_selection,
    }

    # 创建训练器（带预处理）
    trainer = AutoTrainer(
        base_config=config,
        max_iterations=max_iterations,
        early_stopping_rounds=early_stopping_rounds,
        min_improvement=min_improvement,
        model_dir=st.session_state.get('model_dir', Path('model_code/saved')),
        algorithm=selected_algo,
        preprocessor=preprocessor,
        enable_sampling=enable_sampling,
        sampling_config=sampling_config,
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
最佳 AUC: {trainer.best_score:.6f}（迭代 {trainer.best_iteration}）
                """)

            # 实时更新图表
            with chart_placeholder.container():
                fig = trainer.plot_history()
                st.plotly_chart(fig, use_container_width=True)

            # 检查提前停止
            if trainer.check_early_stopping():
                st.info("触发提前停止条件")
                break

            # 超参数优化（第一次迭代后）
            if iteration == 1 and not skip_hyperopt:
                st.info("正在进行超参数优化...")
                trainer.optimize_hyperparameters(n_iter=n_hyperopt_iter)

            # 特征选择（第一次迭代后）
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
            'timestamp': datetime.now().isoformat(),
            'algorithm': selected_algo,
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
        algo_used = results.get('algorithm', 'unknown')
        st.metric("使用算法", ModelFactory.get_display_name(algo_used))

    # 训练历史图表
    st.subheader("📈 训练历史")

    if history_df is not None and not history_df.empty:
        import plotly.graph_objects as go

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
                label="📥 下载训练历史（CSV）",
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
算法：{ModelFactory.get_display_name(results.get('algorithm', 'unknown'))}
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
