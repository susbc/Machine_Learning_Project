"""
tab_eda.py — 数据探索 Tab

合并自：web_app/pages/3_数据分析.py
"""

import streamlit as st


def render() -> None:
    """渲染数据探索 Tab。"""
    from web_app.components.charts import (
        render_distribution_chart,
        render_correlation_heatmap,
        render_missing_value_heatmap,
        render_outlier_boxplot,
    )

    st.header("📈 数据探索")

    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        return

    df = st.session_state.uploaded_data

    sub_dist, sub_corr, sub_outlier, sub_report = st.tabs(
        ["📊 分布分析", "🔥 相关性分析", "⚠️ 异常值检测", "📋 生成报告"]
    )

    with sub_dist:
        _render_distribution(df, render_distribution_chart)

    with sub_corr:
        _render_correlation(df, render_correlation_heatmap, render_missing_value_heatmap)

    with sub_outlier:
        _render_outlier(df, render_outlier_boxplot)

    with sub_report:
        _render_report(df)


def _render_distribution(df, render_distribution_chart) -> None:
    import numpy as np

    st.subheader("📊 字段分布分析")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**数值字段分布**")
        if numeric_cols:
            selected_num_col = st.selectbox(
                "选择数值字段", numeric_cols, key="eda_dist_num"
            )
            if selected_num_col:
                try:
                    fig = render_distribution_chart(df, selected_num_col)
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"eda_plot_num_{selected_num_col}"
                    )
                    col_series = df[selected_num_col].dropna()
                    if len(col_series) > 0:
                        st.caption(
                            f"均值：{col_series.mean():.4f} | 中位数：{col_series.median():.4f} | "
                            f"标准差：{col_series.std():.4f} | 最小值：{col_series.min():.4f} | "
                            f"最大值：{col_series.max():.4f}"
                        )
                except Exception as e:
                    st.warning(f"⚠️ 无法绘制 {selected_num_col} 的分布图：{str(e)}")
        else:
            st.info("无数值字段")

    with col2:
        st.markdown("**分类字段分布**")
        if category_cols:
            selected_cat_col = st.selectbox(
                "选择分类字段", category_cols, key="eda_dist_cat"
            )
            if selected_cat_col:
                fig = render_distribution_chart(df, selected_cat_col)
                st.plotly_chart(
                    fig, use_container_width=True, key=f"eda_plot_cat_{selected_cat_col}"
                )
                value_counts = df[selected_cat_col].value_counts().head(10)
                st.caption(f"Top 类别：{value_counts.to_dict()}")
        else:
            st.info("无分类字段")

    st.markdown("---")
    st.markdown("**📊 所有数值字段分布**")
    if numeric_cols:
        show_all = st.checkbox(
            "显示所有数值字段分布（可能较慢）", key="eda_show_all_dist"
        )
        if show_all:
            for col in numeric_cols[:10]:
                fig = render_distribution_chart(df, col)
                st.plotly_chart(fig, use_container_width=True, key=f"eda_dist_all_{col}")


def _render_correlation(df, render_correlation_heatmap, render_missing_value_heatmap) -> None:
    import numpy as np
    import pandas as pd

    st.subheader("🔥 相关性分析")

    st.markdown("**特征相关性热力图**")
    fig = render_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="eda_corr_heatmap")

        st.markdown("**📊 高相关性特征对**")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.7:
                    high_corr_pairs.append(
                        {
                            "特征 1": corr_matrix.columns[i],
                            "特征 2": corr_matrix.columns[j],
                            "相关系数": corr_matrix.iloc[i, j],
                        }
                    )
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
                "相关系数", ascending=False
            )
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            st.info("💡 高相关性特征可能存在多重共线性，建议考虑移除其中一个")
        else:
            st.success("✅ 未发现高相关性特征对（相关系数 > 0.7）")
    else:
        st.warning("⚠️ 数值字段不足，无法计算相关性")

    st.markdown("---")
    st.markdown("**🔥 缺失值分布热力图**")
    fig = render_missing_value_heatmap(df)
    st.plotly_chart(fig, use_container_width=True, key="eda_missing_heatmap")


def _render_outlier(df, render_outlier_boxplot) -> None:
    import numpy as np
    import pandas as pd

    st.subheader("⚠️ 异常值检测")
    st.markdown("**箱线图检测异常值**")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ 无数值字段")
        return

    selected_cols = st.multiselect(
        "选择要分析的字段",
        numeric_cols,
        default=numeric_cols[:5],
        key="eda_outlier_cols",
    )
    if not selected_cols:
        return

    fig = render_outlier_boxplot(df, selected_cols)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"eda_boxplot_{'_'.join(selected_cols[:3])}",
    )

    st.markdown("**📊 异常值统计**")
    outlier_stats = []
    for col in selected_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_stats.append(
            {
                "字段": col,
                "异常值数量": len(outliers),
                "异常值比例 (%)": round(len(outliers) / len(df) * 100, 2),
                "下界": round(lower, 4),
                "上界": round(upper, 4),
            }
        )
    st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True, hide_index=True)


def _render_report(df) -> None:
    """生成 HTML 数据分析报告。"""
    from datetime import datetime

    st.subheader("📋 生成分析报告")
    st.markdown(
        """
    ### 分析报告内容

    - 数据概览（行数、列数、内存占用）
    - 字段类型分布
    - 缺失值分析
    - 唯一值分析
    - 数值字段统计
    - 相关性分析
    - 异常值检测
    """
    )

    st.markdown("**📝 报告选项**")
    col1, col2 = st.columns(2)
    with col1:
        include_stats = st.checkbox("包含统计信息", value=True, key="eda_rep_stats")
        include_missing = st.checkbox("包含缺失值分析", value=True, key="eda_rep_missing")
        st.checkbox("包含相关性分析", value=True, key="eda_rep_corr")
    with col2:
        st.checkbox("包含异常值检测", value=True, key="eda_rep_outliers")
        st.checkbox("包含图表", value=True, key="eda_rep_charts")

    st.markdown("---")
    if not st.button("📄 生成 HTML 报告", type="primary", key="eda_rep_generate_btn"):
        return

    report_html = _build_report_html(df, include_stats, include_missing)
    st.download_button(
        label="📥 下载 HTML 报告",
        data=report_html,
        file_name=f"数据分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        key="eda_rep_download_btn",
    )
    st.success("✅ 报告生成成功！")


def _build_report_html(df, include_stats: bool, include_missing: bool) -> str:
    """构造 HTML 报告字符串。"""
    import numpy as np
    from datetime import datetime

    head = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>机器学习数据分析报告</title>
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
        <h1>📊 机器学习数据分析报告</h1>
        <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>1. 数据概览</h2>
        <div class="metric"><div class="metric-value">{len(df):,}</div><div class="metric-label">总行数</div></div>
        <div class="metric"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">总列数</div></div>
        <div class="metric"><div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div><div class="metric-label">内存占用</div></div>

        <h2>2. 字段列表</h2>
        <table>
            <tr><th>序号</th><th>字段名</th><th>数据类型</th><th>非空值</th><th>唯一值</th></tr>
    """
    for i, col in enumerate(df.columns):
        head += (
            f"<tr><td>{i + 1}</td><td>{col}</td><td>{df[col].dtype}</td>"
            f"<td>{int(df[col].notna().sum()):,}</td>"
            f"<td>{int(df[col].nunique()):,}</td></tr>"
        )
    head += "</table>"

    if include_stats:
        head += """
        <h2>3. 数值字段统计</h2>
        <table><tr><th>字段</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            head += (
                f"<tr><td>{col}</td><td>{df[col].mean():.4f}</td><td>{df[col].median():.4f}</td>"
                f"<td>{df[col].std():.4f}</td><td>{df[col].min():.4f}</td>"
                f"<td>{df[col].max():.4f}</td></tr>"
            )
        head += "</table>"

    if include_missing:
        head += """
        <h2>4. 缺失值分析</h2>
        <table><tr><th>字段</th><th>缺失数量</th><th>缺失率 (%)</th></tr>
        """
        missing_summary = df.isna().sum()
        for col in df.columns:
            missing_count = int(missing_summary[col])
            missing_rate = missing_count / len(df) * 100
            if missing_count > 0:
                head += (
                    f"<tr><td>{col}</td><td>{missing_count:,}</td>"
                    f"<td>{missing_rate:.2f}%</td></tr>"
                )
        head += "</table>"

    head += "</body></html>"
    return head
