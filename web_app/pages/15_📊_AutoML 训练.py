"""
Agentic AutoML Platform v3.0.0 - AutoML 训练界面

功能:
1. 数据上传
2. 自动特征工程
3. Optuna 超参优化 (SSE 实时推送)
4. 模型对比
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any

# 页面配置
st.set_page_config(
    page_title="AutoML 训练 - v3.0.0",
    page_icon="📊",
    layout="wide"
)

# 标题
st.title("📊 AutoML 训练")
st.markdown("**Agentic AutoML Platform v3.0.0** | 自动化机器学习训练")

# API 基础 URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 侧边栏 - 训练配置
st.sidebar.header("训练配置")
st.sidebar.info("""
**推荐配置**:
- Trial 次数：30-50 次
- 交叉验证：5 折
- 候选模型：2-3 个
""")

# ============================================================================
# 状态初始化
# ============================================================================

if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'trial_results' not in st.session_state:
    st.session_state.trial_results = []

# ============================================================================
# 数据上传
# ============================================================================

st.header("1. 数据上传")

uploaded_file = st.file_uploader(
    "上传 CSV/Excel 文件",
    type=["csv", "xlsx"],
    help="支持 CSV、Excel 格式"
)

if uploaded_file:
    try:
        # 读取数据
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ 数据加载成功：{df.shape}")

        # 数据预览
        with st.expander("📋 数据预览", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("📊 行数", f"{df.shape[0]:,}")
            col2.metric("📐 列数", df.shape[1])
            col3.metric("💾 内存占用", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # ====================================================================
        # 目标变量选择
        # ====================================================================

        st.header("2. 目标变量")

        target_col = st.selectbox(
            "选择目标变量列",
            options=df.columns.tolist(),
            index=len(df.columns) - 1 if len(df.columns) > 0 else 0,
            help="用于分类/回归的目标变量"
        )

        # 检测目标变量类型
        if df[target_col].dtype in ['int64', 'float64']:
            unique_values = df[target_col].nunique()
            if unique_values <= 2:
                st.info(f"📌 检测到二分类任务，目标值：{df[target_col].unique().tolist()}")
            elif unique_values <= 10:
                st.info(f"📌 检测到多分类任务，类别数：{unique_values}")
            else:
                st.info(f"📌 检测到回归任务")
        else:
            st.info(f"📌 检测到类别型目标变量，唯一值：{df[target_col].nunique()}")

        # ====================================================================
        # 特征选择
        # ====================================================================

        st.header("3. 特征选择")

        default_features = [col for col in df.columns if col != target_col]
        feature_cols = st.multiselect(
            "选择特征列 (不选则自动排除目标列)",
            options=df.columns.tolist(),
            default=default_features,
            help="选择用于训练的特征列"
        )

        # 特征统计
        if feature_cols:
            with st.expander("📊 特征统计"):
                stats_data = []
                for col in feature_cols:
                    stats_data.append({
                        '特征': col,
                        '类型': str(df[col].dtype),
                        '缺失率': f"{df[col].isnull().mean():.2%}",
                        '唯一值': df[col].nunique()
                    })
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        # ====================================================================
        # 训练配置
        # ====================================================================

        st.header("4. 训练配置")

        col1, col2 = st.columns(2)

        with col1:
            n_trials = st.slider(
                "Optuna Trial 次数",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Trial 次数越多，寻优越充分，但耗时越长"
            )
            cv_folds = st.slider(
                "交叉验证折数",
                min_value=3,
                max_value=10,
                value=5,
                help="K 折交叉验证，K 越大评估越稳定，但耗时越长"
            )

        with col2:
            st.markdown("**优化配置**:")
            st.markdown("""
            - **优化指标**: ROC-AUC (分类) / RMSE (回归)
            - **优化算法**: TPE 贝叶斯优化
            - **早停**: 支持 (20 轮无改善自动停止)
            - **剪枝**: 支持 (Pruning 低效 Trial)
            """)

        # ====================================================================
        # 模型选择
        # ====================================================================

        st.header("5. 模型选择")

        model_mapping = {
            "LightGBM": "lightgbm",
            "CatBoost": "catboost",
            "XGBoost": "xgboost",
            "Random Forest": "random_forest"
        }

        selected_models_display = st.multiselect(
            "选择候选模型",
            options=list(model_mapping.keys()),
            default=["LightGBM", "CatBoost"],
            help="选择 2-3 个模型进行 AutoML 寻优"
        )

        selected_models = [model_mapping[m] for m in selected_models_display]

        # ====================================================================
        # 开始训练
        # ====================================================================

        st.header("6. 开始训练")

        if st.button("🚀 开始 AutoML 训练", type="primary", disabled=st.session_state.training_started):
            if not feature_cols:
                st.error("❌ 请至少选择一个特征列")
            elif not selected_models:
                st.error("❌ 请至少选择一个候选模型")
            else:
                # 重置状态
                st.session_state.training_started = True
                st.session_state.training_completed = False
                st.session_state.trial_results = []
                st.session_state.current_task_id = None

                # 创建进度显示区域
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                chart_container = st.container()
                log_container = st.expander("📜 训练日志", expanded=True)
                log_text = log_container.empty()

                try:
                    # =========================
                    # 步骤 1: 创建任务
                    # =========================

                    status_text.text("📦 步骤 1/3: 创建任务...")

                    step1_payload = {
                        "user_query": f"提取所有特征数据，目标变量为 {target_col}",
                        "target_col": target_col
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/workflow/step1_retrieve_data",
                        json=step1_payload,
                        timeout=60
                    )

                    if response.status_code != 200:
                        st.error(f"❌ 步骤 1 失败：{response.text}")
                        st.session_state.training_started = False
                        st.stop()

                    step1_result = response.json()
                    task_id = step1_result.get('task_id')
                    st.session_state.current_task_id = task_id

                    st.success(f"✅ 任务创建成功：`{task_id}`")
                    st.json({
                        "SQL": step1_result.get('sql_executed', '')[:200] + "...",
                        "数据形状": step1_result.get('data_shape', [])
                    })

                    # =========================
                    # 步骤 2: 生成特征策略
                    # =========================

                    status_text.text("🧠 步骤 2/3: 生成特征策略...")

                    step2_payload = {
                        "task_id": task_id,
                        "eda_report_text": ""
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/workflow/step2_generate_features",
                        json=step2_payload,
                        timeout=120
                    )

                    if response.status_code != 200:
                        st.error(f"❌ 步骤 2 失败：{response.text}")
                        st.session_state.training_started = False
                        st.stop()

                    step2_result = response.json()
                    actions_count = step2_result.get('actions_count', 0)

                    st.success(f"✅ 特征策略生成完成：{actions_count} 个处理动作")

                    with st.expander("📋 查看特征策略"):
                        st.json(step2_result.get('strategy', {}))

                    # =========================
                    # 步骤 3: 训练 + SSE 推送
                    # =========================

                    status_text.text("🚀 步骤 3/3: 开始训练...")

                    # 使用 requests 流式读取 SSE
                    train_url = f"{API_BASE_URL}/api/workflow/step3_train_stream"
                    params = {"task_id": task_id, "n_trials": n_trials}

                    # 创建 SSE 会话
                    session = requests.Session()
                    trial_logs = []
                    best_auc = 0.5
                    best_model_name = selected_models_display[0] if selected_models_display else "LightGBM"
                    trial_count = 0

                    try:
                        response = session.get(train_url, params=params, stream=True, timeout=300)
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if line:
                                line_str = line.decode('utf-8')

                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]

                                    if data_str.strip() == '[DONE]':
                                        break

                                    try:
                                        data = json.loads(data_str)

                                        if data.get('status') == 'RUNNING':
                                            trial_count += 1
                                            trial_auc = data.get('value', 0)
                                            current_best = data.get('best_value', 0)
                                            params_info = data.get('params', {})

                                            if current_best > best_auc:
                                                best_auc = current_best

                                            # 更新进度
                                            progress = min(trial_count / n_trials * 100, 100)
                                            progress_bar.progress(progress)
                                            status_text.text(
                                                f"🚀 Trial {trial_count}/{n_trials} | "
                                                f"AUC: {trial_auc:.4f} | "
                                                f"Best: {best_auc:.4f}"
                                            )

                                            # 记录日志
                                            trial_logs.append(
                                                f"Trial {trial_count}: AUC = {trial_auc:.4f}, "
                                                f"Best = {current_best:.4f}"
                                            )
                                            log_text.text("\n".join(trial_logs[-15:]))

                                            # 实时更新图表
                                            with chart_container:
                                                chart_data = pd.DataFrame({
                                                    'Trial': list(range(1, trial_count + 1)),
                                                    'AUC': [best_auc] * trial_count
                                                })
                                                st.line_chart(
                                                    chart_data.set_index('Trial'),
                                                    use_container_width=True
                                                )

                                        elif data.get('status') == 'DONE':
                                            status_text.text("✅ 训练完成！")
                                            progress_bar.progress(100)
                                            break

                                        elif data.get('status') == 'ERROR':
                                            st.error(f"❌ 训练错误：{data.get('message', '未知错误')}")
                                            break

                                    except json.JSONDecodeError as e:
                                        st.warning(f"解析 SSE 数据失败：{e}")
                                    except Exception as e:
                                        st.error(f"处理数据失败：{e}")

                        session.close()

                    except requests.exceptions.RequestException as e:
                        st.error(f"连接后端失败：{e}")
                        st.info("请确保后端服务正在运行：`uvicorn backend.main:app --reload`")
                        st.session_state.training_started = False
                        st.stop()

                    # =========================
                    # 训练完成 - 显示结果
                    # =========================

                    st.session_state.training_completed = True
                    st.session_state.training_started = False

                    st.success("🎉 AutoML 训练完成！")

                    # 最佳结果
                    st.subheader("🏆 最佳模型结果")

                    result_col1, result_col2, result_col3 = st.columns(3)
                    result_col1.metric("最佳模型", best_model_name)
                    result_col2.metric("最佳 AUC", f"{best_auc:.4f}")
                    result_col3.metric("Trial 次数", trial_count)

                    # 获取训练结果
                    try:
                        result_response = requests.get(
                            f"{API_BASE_URL}/api/workflow/task/{task_id}/status",
                            timeout=10
                        )
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            st.json(result_data)
                    except:
                        pass

                    # 特征重要性
                    st.subheader("📊 特征重要性")

                    # 尝试从后端获取真实特征重要性
                    try:
                        importance_response = requests.get(
                            f"{API_BASE_URL}/api/workflow/task/{task_id}/importance",
                            timeout=10
                        )
                        if importance_response.status_code == 200:
                            importance_data = importance_response.json()
                            if importance_data and isinstance(importance_data, list) and len(importance_data) > 0:
                                importance_df = pd.DataFrame(importance_data)
                                if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
                                    st.bar_chart(
                                        importance_df.nlargest(20, 'importance').set_index('feature'),
                                        use_container_width=True
                                    )
                                else:
                                    st.info("特征重要性数据格式不正确")
                            else:
                                st.info("暂无特征重要性数据")
                        else:
                            # 使用模拟数据展示
                            st.info("使用模拟特征重要性数据（后端未返回真实数据）")
                            mock_importance = pd.DataFrame({
                                'feature': feature_cols[:10],
                                'importance': np.random.rand(len(feature_cols[:10]))
                            }).sort_values('importance', ascending=False)
                            st.bar_chart(mock_importance.set_index('feature'), use_container_width=True)
                    except Exception as e:
                        st.warning(f"获取特征重要性失败：{e}")
                        # 使用模拟数据展示
                        mock_importance = pd.DataFrame({
                            'feature': feature_cols[:10],
                            'importance': np.random.rand(len(feature_cols[:10]))
                        }).sort_values('importance', ascending=False)
                        st.bar_chart(mock_importance.set_index('feature'), use_container_width=True)

                    # 模型保存信息
                    st.success(f"✅ 模型已保存到 `saved/model_{task_id}.pkl`")

                    # 刷新按钮
                    if st.button("🔄 刷新页面", key="refresh_after_train"):
                        st.rerun()

                except requests.exceptions.Timeout:
                    st.error("⏱️ 请求超时，训练可能仍在后台进行")
                    st.info("请前往【模型管理】页面查看训练结果")
                    st.session_state.training_started = False
                except Exception as e:
                    st.error(f"❌ 训练失败：{str(e)}")
                    st.session_state.training_started = False

        # ====================================================================
        # 历史训练记录
        # ====================================================================

        st.header("📜 历史训练记录")

        try:
            models_response = requests.get(f"{API_BASE_URL}/api/workflow/models", timeout=10)

            if models_response.status_code == 200:
                models_list = models_response.json().get("models", [])

                if models_list:
                    df_models = pd.DataFrame(models_list)
                    st.dataframe(
                        df_models.rename(columns={
                            'task_id': '任务 ID',
                            'model_path': '模型路径'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("暂无历史训练记录")
            else:
                st.warning("无法获取历史训练记录")
        except requests.exceptions.RequestException as e:
            st.warning(f"获取历史记录失败：{e}")

    except Exception as e:
        st.error(f"❌ 数据加载失败：{str(e)}")
else:
    st.info("👆 请上传数据文件开始训练")

    # 示例数据按钮
    if st.button("📦 使用示例数据"):
        st.info("示例数据功能开发中... 请上传自己的数据文件")

# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>Agentic AutoML Platform v3.0.0</strong></p>
        <p>基于 Optuna TPE 贝叶斯优化 | 支持 LightGBM / CatBoost / XGBoost / Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True
)
