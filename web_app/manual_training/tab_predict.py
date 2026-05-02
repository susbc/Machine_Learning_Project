"""
tab_predict.py — 预测与结果 Tab

合并自：
  - web_app/pages/6_模型预测.py（选模型 / 上传数据 / 跑预测 / 导出）
  - web_app/pages/7_结果分析.py（预测结果分布 / 阈值分析）

注意：原 7_结果分析.py 顶部 import 有路径 bug（缺 web_app. 前缀），
本模块使用完整路径修正。
"""

import streamlit as st


def render() -> None:
    """渲染预测与结果 Tab。"""
    sub_run, sub_res = st.tabs(["🔮 预测", "📋 结果分析"])
    with sub_run:
        _render_run_predict()
    with sub_res:
        _render_result_analysis()


# ===== 子分区 1：预测控制（来自原 6_模型预测.py） =====


def _scan_saved_models_for_predict() -> list:
    """扫描模型目录中的所有模型文件。"""
    import joblib

    model_dir = st.session_state.model_dir
    models = []

    if model_dir.exists():
        for file_path in model_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(file_path)
                model_info = {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "timestamp": file_path.stat().st_mtime,
                }
                if isinstance(model_data, dict) and "test_auc" in model_data:
                    model_info["auc"] = model_data["test_auc"]
                models.append(model_info)
            except Exception:
                pass

    models.sort(key=lambda x: x["timestamp"], reverse=True)
    return models


def _render_run_predict() -> None:
    import pandas as pd
    import joblib
    from datetime import datetime
    from web_app.utils.file_handler import read_uploaded_file, create_excel_download_button
    from web_app.components.progress import render_prediction_progress

    st.subheader("🔮 模型预测")
    st.markdown("**📦 选择模型**")

    models = _scan_saved_models_for_predict()

    if not models:
        st.warning("⚠️ 暂无已训练的模型，请先在「4️⃣ 模型训练」Tab 训练模型")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_model_name = st.selectbox(
            "选择要使用的模型",
            [m["name"] for m in models],
            format_func=lambda x: f"{x}",
            key="pred_run_model_select",
        )

    with col2:
        if selected_model_name:
            selected_model = next(m for m in models if m["name"] == selected_model_name)
            if "auc" in selected_model:
                st.info(f"AUC: {selected_model['auc']:.4f}")

    if not selected_model_name:
        return

    selected_model = next(m for m in models if m["name"] == selected_model_name)

    try:
        model_data = joblib.load(selected_model["path"])
        if isinstance(model_data, dict):
            model = model_data.get("model")
            threshold = model_data.get("best_threshold", 0.5)
            feature_names = model_data.get("feature_names", None)
        else:
            model = model_data
            threshold = 0.5
            feature_names = None
        st.success(f"✅ 模型已加载：{selected_model_name}")
    except Exception as e:
        st.error(f"加载模型失败：{str(e)}")
        return

    st.markdown("---")
    st.markdown("**📤 上传预测数据**")

    uploaded_file = st.file_uploader(
        "上传用于预测的数据文件 (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="数据格式应与训练数据一致",
        key="pred_run_uploader",
    )

    if uploaded_file is not None:
        predict_df = read_uploaded_file(uploaded_file)
        if predict_df is not None:
            st.session_state.predict_data = predict_df
            st.success(f"✅ 数据已上传：{uploaded_file.name}")
            st.info(f"📊 数据大小：{predict_df.shape[0]:,} 行 × {predict_df.shape[1]} 列")
            with st.expander("📋 数据预览"):
                st.dataframe(predict_df.head(), use_container_width=True)

    if st.session_state.get("predict_data") is None:
        st.info("👆 请上传预测数据")
        return

    st.markdown("---")
    st.markdown("**⚙️ 预测配置**")

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider(
            "分类阈值",
            min_value=0.1,
            max_value=0.9,
            value=float(threshold),
            step=0.05,
            format="%.2f",
            key="pred_run_threshold",
        )
        st.session_state.prediction_threshold = threshold
    with col2:
        st.checkbox(
            "包含预测概率",
            value=True,
            help="在结果中包含预测概率值",
            key="pred_run_include_proba",
        )
    with col3:
        st.number_input(
            "批处理大小",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="大数据集时分批预测",
            key="pred_run_batch_size",
        )

    st.markdown("---")
    st.markdown("**🔮 预测控制**")

    col1, col2 = st.columns([2, 1])
    predict_df = st.session_state.predict_data
    with col1:
        st.info(
            f"""
        📊 预测数据：{len(predict_df):,} 行
        🎯 分类阈值：{threshold:.2f}
        📦 模型：{selected_model_name}
        """
        )
    with col2:
        start_prediction = st.button(
            "🚀 开始预测",
            type="primary",
            use_container_width=True,
            key="pred_run_start_btn",
        )

    if start_prediction:
        try:
            render_prediction_progress(0.1, "数据预处理...")

            from model_code.utils import (
                handle_dataframe_outliers,
                handle_dataframe_missing_values,
                encode_dataframe,
                standardize_dataframe,
            )

            config_df = st.session_state.get("field_config")
            if config_df is None:
                st.error("❌ 未找到字段配置，请先在「2️⃣ 字段配置」Tab 配置字段")
                return

            pred_data = predict_df.copy()

            render_prediction_progress(0.3, "异常值处理...")
            pred_data = handle_dataframe_outliers(pred_data, config_df)

            render_prediction_progress(0.5, "缺失值处理...")
            pred_data = handle_dataframe_missing_values(pred_data, config_df)

            render_prediction_progress(0.7, "编码和标准化...")
            pred_data = encode_dataframe(pred_data, config_df)
            pred_data = standardize_dataframe(pred_data, config_df)

            render_prediction_progress(0.85, "模型预测...")

            key_col = config_df[config_df["field_type"] == "key"]["field_name"].values

            if feature_names:
                X_pred = pred_data[feature_names]
            else:
                label_col = config_df[config_df["field_type"] == "label"]["field_name"].values
                exclude_cols = []
                if len(label_col) > 0:
                    exclude_cols.append(label_col[0])
                if len(key_col) > 0:
                    exclude_cols.append(key_col[0])
                X_pred = pred_data[[c for c in pred_data.columns if c not in exclude_cols]]

            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_pred)[:, 1]
            else:
                y_pred_proba = model.predict(X_pred)

            y_pred = (y_pred_proba >= threshold).astype(int)

            render_prediction_progress(1.0, "模型预测完成！")

            results_df = pd.DataFrame(
                {"预测概率": y_pred_proba, "预测标签": y_pred}
            )

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

    if st.session_state.get("prediction_completed"):
        st.markdown("---")
        st.markdown("**📊 预测结果**")

        results_df = st.session_state.predict_results

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总样本数", f"{len(results_df):,}")
        with col2:
            pred_positive = int((results_df["预测标签"] == 1).sum())
            st.metric("预测正例", f"{pred_positive:,}")
        with col3:
            pred_negative = int((results_df["预测标签"] == 0).sum())
            st.metric("预测负例", f"{pred_negative:,}")
        with col4:
            positive_rate = pred_positive / len(results_df) * 100
            st.metric("正例比例", f"{positive_rate:.2f}%")

        st.markdown("**📋 结果预览**")
        st.dataframe(results_df.head(100), use_container_width=True, height=300)

        st.markdown("**📊 预测概率分布**")

        import plotly.express as px

        fig = px.histogram(
            results_df,
            x="预测概率",
            color="预测标签",
            nbins=50,
            title="预测概率分布",
            labels={"预测概率": "预测概率", "count": "样本数"},
            color_discrete_map={0: "blue", 1: "red"},
        )
        fig.add_shape(
            type="line",
            x0=threshold,
            y0=0,
            x1=threshold,
            y1=results_df["预测标签"].value_counts().max(),
            line=dict(color="green", width=3, dash="dash"),
        )
        st.plotly_chart(
            fig, use_container_width=True, key="pred_run_proba_hist_chart"
        )

        st.markdown("**💾 导出结果**")
        col1, col2 = st.columns(2)
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 下载 CSV 格式",
                data=csv,
                file_name=f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="pred_run_dl_csv_btn",
            )
        with col2:
            create_excel_download_button(
                results_df,
                f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "📥 下载 Excel 格式",
            )

        st.markdown("---")
        if st.button("🗑️ 清除预测结果", key="pred_run_clear_btn"):
            st.session_state.predict_results = None
            st.session_state.prediction_completed = False
            st.rerun()
    else:
        st.info("👆 点击【开始预测】按钮执行预测")


# ===== 子分区 2：结果分析（来自原 7_结果分析.py） =====


def _render_result_analysis() -> None:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from web_app.components.charts import (
        render_prediction_distribution,
        render_threshold_analysis,
        render_roc_curve,
        render_precision_recall_curve,
    )

    st.subheader("📋 结果分析")

    if st.session_state.get("predict_results") is None:
        st.warning("⚠️ 暂无预测结果，请先在「🔮 预测」子 Tab 进行预测")
        return

    results_df = st.session_state.predict_results
    has_train_results = st.session_state.get("training_results") is not None

    st.markdown("**📊 预测结果概览**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{len(results_df):,}")
    with col2:
        pred_positive = int((results_df["预测标签"] == 1).sum())
        st.metric("预测正例", f"{pred_positive:,}")
    with col3:
        pred_negative = int((results_df["预测标签"] == 0).sum())
        st.metric("预测负例", f"{pred_negative:,}")
    with col4:
        positive_rate = pred_positive / len(results_df) * 100
        st.metric("正例比例", f"{positive_rate:.2f}%")

    st.markdown("---")

    sub_dist, sub_thr, sub_cmp, sub_rep = st.tabs(
        ["📊 分布分析", "🎯 阈值分析", "📈 对比分析", "📋 导出报告"]
    )

    with sub_dist:
        st.markdown("**📊 预测分布分析**")
        st.markdown("**预测概率分布**")
        fig = render_prediction_distribution(
            results_df["预测概率"].values,
            st.session_state.prediction_threshold,
        )
        st.plotly_chart(fig, use_container_width=True, key="pred_res_dist_chart")

        st.markdown("**📋 分位数统计**")
        proba = results_df["预测概率"]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("最小值", f"{proba.min():.4f}")
        with col2:
            st.metric("25%", f"{proba.quantile(0.25):.4f}")
        with col3:
            st.metric("中位数", f"{proba.median():.4f}")
        with col4:
            st.metric("75%", f"{proba.quantile(0.75):.4f}")
        with col5:
            st.metric("最大值", f"{proba.max():.4f}")

        st.markdown("**📊 分组统计**")
        group_stats = (
            results_df.groupby("预测标签")["预测概率"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        st.dataframe(group_stats, use_container_width=True)

    with sub_thr:
        st.markdown("**🎯 阈值分析**")

        if "真实标签" in results_df.columns:
            y_true = results_df["真实标签"].values
            y_scores = results_df["预测概率"].values

            fig = render_threshold_analysis(y_true, y_scores)
            st.plotly_chart(
                fig, use_container_width=True, key="pred_res_threshold_chart"
            )

            st.markdown("**📊 不同阈值的性能指标**")

            from sklearn.metrics import precision_score, recall_score, f1_score

            thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_metrics = []
            for thresh in thresholds_to_check:
                y_pred = (y_scores >= thresh).astype(int)
                threshold_metrics.append(
                    {
                        "阈值": thresh,
                        "Precision": precision_score(y_true, y_pred, zero_division=0),
                        "Recall": recall_score(y_true, y_pred, zero_division=0),
                        "F1": f1_score(y_true, y_pred, zero_division=0),
                        "预测正例": int((y_pred == 1).sum()),
                        "预测负例": int((y_pred == 0).sum()),
                    }
                )
            threshold_df = pd.DataFrame(threshold_metrics)
            st.dataframe(
                threshold_df.round(4), use_container_width=True, hide_index=True
            )

            import plotly.express as px

            fig = px.line(
                threshold_df,
                x="阈值",
                y=["Precision", "Recall", "F1"],
                title="阈值 - 指标曲线",
                labels={"value": "分数", "variable": "指标"},
            )
            st.plotly_chart(
                fig, use_container_width=True, key="pred_res_threshold_metrics_chart"
            )

            st.markdown("**📈 ROC 曲线**")
            fig = render_roc_curve(y_true, y_scores)
            st.plotly_chart(fig, use_container_width=True, key="pred_res_roc_chart")

            st.markdown("**📈 PR 曲线**")
            fig = render_precision_recall_curve(y_true, y_scores)
            st.plotly_chart(fig, use_container_width=True, key="pred_res_pr_chart")
        else:
            st.info("💡 要使用阈值分析功能，请在预测结果中包含真实标签列")
            st.markdown("**📊 阈值敏感性分析（基于预测概率）**")

            y_scores = results_df["预测概率"].values
            thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_stats = []
            for thresh in thresholds_to_check:
                y_pred = (y_scores >= thresh).astype(int)
                threshold_stats.append(
                    {
                        "阈值": thresh,
                        "预测正例": int((y_pred == 1).sum()),
                        "预测负例": int((y_pred == 0).sum()),
                        "正例比例 (%)": round((y_pred == 1).mean() * 100, 2),
                    }
                )
            threshold_df = pd.DataFrame(threshold_stats)
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    with sub_cmp:
        st.markdown("**📈 对比分析**")

        if has_train_results:
            train_results = st.session_state.training_results

            st.markdown("**📊 训练集 vs 预测集对比**")

            col1, col2 = st.columns(2)
            with col1:
                st.info("**训练集**")
                st.metric("样本数", f"{train_results['test_size']:,}")
                st.metric("ROC-AUC", f"{train_results['test_auc']:.4f}")
                st.metric("最佳 F1", f"{train_results['best_f1']:.4f}")
            with col2:
                st.info("**预测集**")
                st.metric("样本数", f"{len(results_df):,}")
                st.metric(
                    "当前阈值", f"{st.session_state.prediction_threshold:.2f}"
                )
                st.metric(
                    "正例比例", f"{(results_df['预测标签'] == 1).mean():.2%}"
                )

            st.markdown("**📊 概率分布对比**")
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=train_results["y_pred_proba"],
                    name="训练集",
                    opacity=0.6,
                    nbinsx=50,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=results_df["预测概率"],
                    name="预测集",
                    opacity=0.6,
                    nbinsx=50,
                )
            )
            fig.update_layout(
                title="训练集 vs 预测集 概率分布对比",
                xaxis_title="预测概率",
                yaxis_title="频数",
                barmode="overlay",
            )
            st.plotly_chart(
                fig, use_container_width=True, key="pred_res_prob_compare_chart"
            )
        else:
            st.info("💡 要使用对比分析功能，请先进行模型训练")
            st.markdown("**📊 当前预测分布**")
            fig = render_prediction_distribution(
                results_df["预测概率"].values,
                st.session_state.prediction_threshold,
            )
            st.plotly_chart(
                fig, use_container_width=True, key="pred_res_curr_dist_chart"
            )

    with sub_rep:
        st.markdown("**📋 导出分析报告**")
        st.markdown(
            """
        ### 报告内容

        - 预测结果概览
        - 预测分布统计
        - 阈值分析（如有真实标签）
        - 与训练集对比（如可用）
        """
        )

        st.markdown("**📝 报告选项**")

        include_stats = st.checkbox("包含统计信息", value=True, key="pred_res_rep_stats")
        st.checkbox("包含图表数据", value=True, key="pred_res_rep_charts")
        st.checkbox(
            "包含对比分析",
            value=has_train_results,
            disabled=not has_train_results,
            key="pred_res_rep_comparison",
        )

        st.markdown("---")
        if st.button("📄 生成 HTML 报告", type="primary", key="pred_res_rep_generate_btn"):
            report_html = _build_predict_report_html(results_df, include_stats)
            st.download_button(
                label="📥 下载 HTML 报告",
                data=report_html,
                file_name=f"预测分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="pred_res_rep_dl_html_btn",
            )
            st.success("✅ 报告生成成功！")

            st.markdown("---")
            st.markdown("**💾 下载原始数据**")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果 (CSV)",
                data=csv,
                file_name=f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="pred_res_rep_dl_csv_btn",
            )


def _build_predict_report_html(results_df, include_stats: bool) -> str:
    """构造预测分析 HTML 报告。"""
    from datetime import datetime

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>机器学习预测结果分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #ecf0f1; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>🔮 机器学习预测结果分析报告</h1>
        <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>1. 预测结果概览</h2>
        <div class="metric"><div class="metric-value">{len(results_df):,}</div><div class="metric-label">总样本数</div></div>
        <div class="metric"><div class="metric-value">{int((results_df['预测标签'] == 1).sum()):,}</div><div class="metric-label">预测正例</div></div>
        <div class="metric"><div class="metric-value">{int((results_df['预测标签'] == 0).sum()):,}</div><div class="metric-label">预测负例</div></div>
        <div class="metric"><div class="metric-value">{(results_df['预测标签'] == 1).mean():.2%}</div><div class="metric-label">正例比例</div></div>

        <h2>2. 预测分布统计</h2>
        <table>
            <tr><th>统计量</th><th>值</th></tr>
            <tr><td>最小值</td><td>{results_df['预测概率'].min():.4f}</td></tr>
            <tr><td>25%</td><td>{results_df['预测概率'].quantile(0.25):.4f}</td></tr>
            <tr><td>中位数</td><td>{results_df['预测概率'].median():.4f}</td></tr>
            <tr><td>75%</td><td>{results_df['预测概率'].quantile(0.75):.4f}</td></tr>
            <tr><td>最大值</td><td>{results_df['预测概率'].max():.4f}</td></tr>
        </table>
    """

    if "真实标签" in results_df.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score

        y_true = results_df["真实标签"].values
        y_pred = results_df["预测标签"].values

        report_html += f"""
        <h2>3. 模型评估指标</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>Precision</td><td>{precision_score(y_true, y_pred, zero_division=0):.4f}</td></tr>
            <tr><td>Recall</td><td>{recall_score(y_true, y_pred, zero_division=0):.4f}</td></tr>
            <tr><td>F1 Score</td><td>{f1_score(y_true, y_pred, zero_division=0):.4f}</td></tr>
        </table>
        """

    if include_stats:
        report_html += """
        <h2>4. 分组统计</h2>
        <table>
            <tr><th>预测标签</th><th>样本数</th><th>平均概率</th><th>概率标准差</th></tr>
        """
        for label in [0, 1]:
            group = results_df[results_df["预测标签"] == label]["预测概率"]
            report_html += f"""
            <tr>
                <td>{label}</td>
                <td>{len(group):,}</td>
                <td>{group.mean():.4f}</td>
                <td>{group.std():.4f}</td>
            </tr>
            """
        report_html += "</table>"

    report_html += """
    </body>
    </html>
    """
    return report_html
