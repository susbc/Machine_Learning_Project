"""
模型训练页面

功能：
- 模型参数调整（LightGBM 参数）
- 训练进度实时展示
- 交叉验证结果
- 训练完成后模型评估
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

# 导入现有模型代码
sys.path.insert(0, str(project_root))

from utils.session_state import init_session_state, add_training_log, get_training_logs
from components.progress import (
    render_training_progress,
    render_cv_results,
    render_training_summary,
)
from components.charts import (
    render_roc_curve,
    render_precision_recall_curve,
    render_confusion_matrix,
    render_feature_importance,
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="模型训练",
    page_icon="🚀",
    layout="wide"
)

# 标题
st.title("🚀 模型训练")
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
# 参数配置区域
# ============================================================================

st.header("🎯 模型参数配置")

with st.expander("⚙️ LightGBM 参数设置", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider(
            "n_estimators (树的数量)",
            min_value=50,
            max_value=1000,
            value=st.session_state.model_params.get('n_estimators', 200),
            step=50
        )
        
        learning_rate = st.slider(
            "learning_rate (学习率)",
            min_value=0.001,
            max_value=0.3,
            value=st.session_state.model_params.get('learning_rate', 0.1),
            step=0.01,
            format="%.3f"
        )
        
        max_depth = st.slider(
            "max_depth (树的最大深度)",
            min_value=3,
            max_value=15,
            value=st.session_state.model_params.get('max_depth', 6),
            step=1
        )
    
    with col2:
        num_leaves = st.slider(
            "num_leaves (叶子节点数)",
            min_value=10,
            max_value=200,
            value=st.session_state.model_params.get('num_leaves', 50),
            step=10
        )
        
        min_child_samples = st.slider(
            "min_child_samples (叶子最小样本数)",
            min_value=5,
            max_value=100,
            value=st.session_state.model_params.get('min_child_samples', 20),
            step=5
        )
        
        subsample = st.slider(
            "subsample (样本采样比例)",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.model_params.get('subsample', 0.9),
            step=0.05,
            format="%.2f"
        )
    
    with col3:
        colsample_bytree = st.slider(
            "colsample_bytree (特征采样比例)",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.model_params.get('colsample_bytree', 0.9),
            step=0.05,
            format="%.2f"
        )
        
        reg_alpha = st.slider(
            "reg_alpha (L1 正则化)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.model_params.get('reg_alpha', 0.1),
            step=0.1,
            format="%.1f"
        )
        
        reg_lambda = st.slider(
            "reg_lambda (L2 正则化)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.model_params.get('reg_lambda', 0.1),
            step=0.1,
            format="%.1f"
        )

# 训练配置
st.subheader("📊 训练配置")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider(
        "测试集比例",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        format="%.2f"
    )

with col2:
    cv_folds = st.slider(
        "交叉验证折数",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

with col3:
    random_state = st.number_input(
        "随机种子",
        min_value=0,
        max_value=10000,
        value=42,
        step=1
    )

# 是否启用网格搜索
use_grid_search = st.checkbox(
    "启用随机网格搜索调参",
    value=True,
    help="启用后会尝试不同的参数组合，寻找最优参数"
)

if use_grid_search:
    n_iter = st.slider(
        "随机搜索迭代次数",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
else:
    n_iter = 1

# 保存参数
st.session_state.model_params = {
    "n_estimators": n_estimators,
    "learning_rate": learning_rate,
    "max_depth": max_depth,
    "num_leaves": num_leaves,
    "min_child_samples": min_child_samples,
    "subsample": subsample,
    "colsample_bytree": colsample_bytree,
    "reg_alpha": reg_alpha,
    "reg_lambda": reg_lambda,
}

# ============================================================================
# 训练控制区域
# ============================================================================

st.markdown("---")
st.header("🚀 训练控制")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # 显示当前数据信息
    df = st.session_state.uploaded_data
    config_df = st.session_state.field_config
    
    st.info(f"""
    📊 训练数据：{len(df):,} 行 × {len(df.columns)} 列  
    ⚙️ 配置字段：{len(config_df)} 个  
    🎯 标签字段：{config_df[config_df['field_type']=='label']['field_name'].values[0] if 'label' in config_df['field_type'].values else '未指定'}
    """)

with col2:
    start_training = st.button(
        "🚀 开始训练",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get('training_in_progress', False)
    )

with col3:
    if st.button(
        "🗑️ 清除训练结果",
        use_container_width=True,
        disabled=st.session_state.get('training_results') is None
    ):
        st.session_state.training_results = None
        st.session_state.training_logs = []
        st.session_state.training_progress = 0
        st.rerun()

# ============================================================================
# 训练执行
# ============================================================================

if start_training:
    st.session_state.training_in_progress = True
    st.session_state.training_logs = []
    st.session_state.training_progress = 0
    
    # 创建进度占位符
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    
    try:
        add_training_log("=" * 60)
        add_training_log("开始模型训练流程")
        add_training_log("=" * 60)
        add_training_log(f"训练样本数：{len(df):,}")
        add_training_log(f"特征数：{len(df.columns) - 1}")
        add_training_log(f"参数配置：{st.session_state.model_params}")
        
        # 准备数据
        add_training_log("\n正在准备训练数据...")
        st.session_state.training_progress = 0.1
        
        # 识别 label 和 key 字段
        label_col = config_df[config_df['field_type'] == 'label']['field_name'].values
        key_col = config_df[config_df['field_type'] == 'key']['field_name'].values
        
        if len(label_col) == 0:
            raise ValueError("配置表中未指定 label 字段")
        
        label_col = label_col[0]
        key_col = key_col[0] if len(key_col) > 0 else None
        
        add_training_log(f"标签字段：{label_col}")
        add_training_log(f"ID 字段：{key_col or '未指定'}")
        
        # 数据预处理
        add_training_log("\n正在执行数据预处理...")
        st.session_state.training_progress = 0.2
        
        # 导入预处理函数
        from model_code.utils import (
            handle_dataframe_outliers,
            handle_dataframe_missing_values,
            encode_dataframe,
            standardize_dataframe,
        )
        
        # 复制数据避免修改原数据
        train_data = df.copy()
        
        # 异常值处理
        train_data = handle_dataframe_outliers(train_data, config_df)
        st.session_state.training_progress = 0.3
        
        # 缺失值处理
        train_data = handle_dataframe_missing_values(train_data, config_df)
        st.session_state.training_progress = 0.4
        
        # 编码处理
        train_data = encode_dataframe(train_data, config_df)
        st.session_state.training_progress = 0.5
        
        # 标准化处理
        train_data = standardize_dataframe(train_data, config_df)
        st.session_state.training_progress = 0.6
        
        # 准备特征和标签
        feature_cols = [c for c in train_data.columns if c not in [label_col, key_col] if key_col]
        if key_col is None:
            feature_cols = [c for c in train_data.columns if c != label_col]
        
        X = train_data[feature_cols]
        y = train_data[label_col]
        
        add_training_log(f"特征矩阵：{X.shape}")
        add_training_log(f"标签向量：{y.shape}")
        add_training_log(f"正样本比例：{y.mean():.2%}")
        
        # 训练模型
        add_training_log("\n开始训练 LightGBM 模型...")
        st.session_state.training_progress = 0.7
        
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.metrics import (
            roc_auc_score, precision_recall_curve,
            classification_report, confusion_matrix
        )
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        add_training_log(f"训练集：{X_train.shape[0]:,} 样本")
        add_training_log(f"测试集：{X_test.shape[0]:,} 样本")
        
        # 定义模型
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            verbose=-1,
            random_state=random_state,
            n_jobs=-1,
            **st.session_state.model_params
        )
        
        if use_grid_search and n_iter > 1:
            add_training_log(f"\n执行随机网格搜索 (n_iter={n_iter}, cv={cv_folds})...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 6, 7, 8],
                'num_leaves': [30, 50, 70],
                'min_child_samples': [10, 20, 30],
            }
            
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0,
                random_state=random_state
            )
            
            random_search.fit(X_train, y_train)
            
            model = random_search.best_estimator_
            add_training_log(f"最佳参数：{random_search.best_params_}")
            st.session_state.cv_results = random_search.cv_results_
        else:
            add_training_log("\n使用固定参数训练...")
            model.fit(X_train, y_train)
        
        st.session_state.training_progress = 0.85
        add_training_log("\n模型训练完成！")
        
        # 模型评估
        add_training_log("\n正在评估模型...")
        st.session_state.training_progress = 0.9
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # 计算指标
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # 找到最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        add_training_log(f"\n测试集 ROC-AUC: {test_auc:.6f}")
        add_training_log(f"最佳阈值：{best_threshold:.4f}")
        add_training_log(f"最佳 F1 分数：{best_f1:.6f}")
        
        # 保存训练结果
        st.session_state.training_results = {
            'model': model,
            'model_params': st.session_state.model_params,
            'test_auc': test_auc,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'feature_names': feature_cols,
            'feature_importances': model.feature_importances_,
            'y_test': y_test.values,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'timestamp': datetime.now().isoformat(),
        }
        
        # 保存模型到文件
        model_path = st.session_state.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        add_training_log(f"\n✅ 模型已保存：{model_path}")
        
        # 保存训练摘要
        summary_path = st.session_state.model_dir / f"train_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(st.session_state.training_results, summary_path)
        add_training_log(f"✅ 训练摘要已保存：{summary_path}")
        
        st.session_state.training_progress = 1.0
        add_training_log("\n" + "=" * 60)
        add_training_log("训练流程完成！")
        add_training_log("=" * 60)
        
        # 更新模型列表
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = []
        
        st.session_state.trained_models.append({
            'name': model_path.stem,
            'path': str(model_path),
            'auc': test_auc,
            'f1': best_f1,
            'timestamp': datetime.now().isoformat(),
        })
        
        st.session_state.training_in_progress = False
        
        # 刷新页面显示结果
        st.rerun()
        
    except Exception as e:
        st.session_state.training_in_progress = False
        add_training_log(f"\n❌ 训练失败：{str(e)}")
        st.error(f"训练出错：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# 显示训练进度和结果
# ============================================================================

if st.session_state.get('training_in_progress', False):
    render_training_progress(
        st.session_state.get('training_progress', 0),
        get_training_logs()
    )

if st.session_state.get('training_results') is not None:
    st.markdown("---")
    st.header("📊 训练结果")
    
    results = st.session_state.training_results
    
    # 核心指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC", f"{results['test_auc']:.4f}")
    
    with col2:
        st.metric("最佳 F1", f"{results['best_f1']:.4f}")
    
    with col3:
        st.metric("最佳阈值", f"{results['best_threshold']:.4f}")
    
    with col4:
        st.metric("测试集大小", f"{results['test_size']:,}")
    
    # 图表展示
    st.subheader("📈 模型评估")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "ROC 曲线",
        "PR 曲线",
        "混淆矩阵",
        "特征重要性"
    ])
    
    with tab1:
        fig = render_roc_curve(results['y_test'], results['y_pred_proba'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = render_precision_recall_curve(results['y_test'], results['y_pred_proba'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = render_confusion_matrix(results['y_test'], results['y_pred'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = render_feature_importance(
            results['feature_names'],
            results['feature_importances']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 分类报告
    st.subheader("📋 分类报告")
    from sklearn.metrics import classification_report
    report = classification_report(
        results['y_test'],
        results['y_pred'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)
    
    # 导出结果
    st.subheader("💾 导出结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 导出预测结果
        pred_df = pd.DataFrame({
            '真实标签': results['y_test'],
            '预测概率': results['y_pred_proba'],
            '预测标签': results['y_pred']
        })
        
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 下载预测结果 (CSV)",
            data=csv,
            file_name=f"train_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # 导出特征重要性
        importance_df = pd.DataFrame({
            '特征名': results['feature_names'],
            '重要性': results['feature_importances']
        }).sort_values('重要性', ascending=False)
        
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="📥 下载特征重要性 (CSV)",
            data=csv,
            file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
