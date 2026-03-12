"""
数据分析页面

功能：
- EDA 可视化（分布图、相关性热力图、缺失值热力图）
- 异常值检测可视化
- 分析报告生成（HTML/PDF）
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_state import init_session_state
from components.charts import (
    render_distribution_chart,
    render_correlation_heatmap,
    render_missing_value_heatmap,
    render_outlier_boxplot,
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="数据分析",
    page_icon="📈",
    layout="wide"
)

# 标题
st.title("📈 数据分析")
st.markdown("---")

# 检查是否有数据
if st.session_state.get('uploaded_data') is None:
    st.warning("⚠️ 请先在【数据管理】页面上传数据")
    
    if st.button("📊 前往数据管理"):
        st.switch_page("pages/1_数据管理.py")
else:
    df = st.session_state.uploaded_data
    
    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 分布分析",
        "🔥 相关性分析",
        "⚠️ 异常值检测",
        "📋 生成报告"
    ])
    
    with tab1:
        st.header("📊 字段分布分析")
        
        # 选择要分析的字段
        all_columns = df.columns.tolist()
        
        # 数值列和分类列分开
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        category_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("数值字段分布")
            if numeric_cols:
                selected_num_col = st.selectbox(
                    "选择数值字段",
                    numeric_cols,
                    key="dist_num"
                )
                if selected_num_col:
                    fig = render_distribution_chart(df, selected_num_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示统计信息
                    st.caption(f"""
                    均值：{df[selected_num_col].mean():.4f} | 
                    中位数：{df[selected_num_col].median():.4f} | 
                    标准差：{df[selected_num_col].std():.4f} | 
                    最小值：{df[selected_num_col].min():.4f} | 
                    最大值：{df[selected_num_col].max():.4f}
                    """)
            else:
                st.info("无数值字段")
        
        with col2:
            st.subheader("分类字段分布")
            if category_cols:
                selected_cat_col = st.selectbox(
                    "选择分类字段",
                    category_cols,
                    key="dist_cat"
                )
                if selected_cat_col:
                    fig = render_distribution_chart(df, selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示 Top 10 类别
                    value_counts = df[selected_cat_col].value_counts().head(10)
                    st.caption(f"Top 类别：{value_counts.to_dict()}")
            else:
                st.info("无分类字段")
        
        # 批量显示所有数值字段分布
        st.markdown("---")
        st.subheader("📊 所有数值字段分布")
        
        if numeric_cols:
            show_all = st.checkbox("显示所有数值字段分布（可能较慢）")
            if show_all:
                for col in numeric_cols[:10]:  # 最多显示 10 个
                    fig = render_distribution_chart(df, col)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔥 相关性分析")
        
        # 相关性热力图
        st.subheader("特征相关性热力图")
        fig = render_correlation_heatmap(df)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示高相关性特征对
            st.subheader("📊 高相关性特征对")
            
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            # 找出高相关性对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.7:  # 阈值 0.7
                        high_corr_pairs.append({
                            '特征 1': corr_matrix.columns[i],
                            '特征 2': corr_matrix.columns[j],
                            '相关系数': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df = high_corr_df.sort_values('相关系数', ascending=False)
                st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
                
                st.info("💡 高相关性特征可能存在多重共线性，建议考虑移除其中一个")
            else:
                st.success("✅ 未发现高相关性特征对（相关系数 > 0.7）")
        else:
            st.warning("⚠️ 数值字段不足，无法计算相关性")
        
        # 缺失值热力图
        st.markdown("---")
        st.subheader("🔥 缺失值分布热力图")
        fig = render_missing_value_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("⚠️ 异常值检测")
        
        # 箱线图
        st.subheader("箱线图检测异常值")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_cols = st.multiselect(
                "选择要分析的字段",
                numeric_cols,
                default=numeric_cols[:5]
            )
            
            if selected_cols:
                fig = render_outlier_boxplot(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True)
                
                # 异常值统计
                st.subheader("📊 异常值统计")
                
                outlier_stats = []
                for col in selected_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_rate = outlier_count / len(df) * 100
                    
                    outlier_stats.append({
                        '字段': col,
                        '异常值数量': outlier_count,
                        '异常值比例 (%)': round(outlier_rate, 2),
                        '下界': round(lower_bound, 4),
                        '上界': round(upper_bound, 4)
                    })
                
                outlier_df = pd.DataFrame(outlier_stats)
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 无数值字段")
    
    with tab4:
        st.header("📋 生成分析报告")
        
        st.markdown("""
        ### 分析报告内容
        
        - 数据概览（行数、列数、内存占用）
        - 字段类型分布
        - 缺失值分析
        - 唯一值分析
        - 数值字段统计
        - 相关性分析
        - 异常值检测
        
        """)
        
        # 报告选项
        st.subheader("📝 报告选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_stats = st.checkbox("包含统计信息", value=True)
            include_missing = st.checkbox("包含缺失值分析", value=True)
            include_correlation = st.checkbox("包含相关性分析", value=True)
        
        with col2:
            include_outliers = st.checkbox("包含异常值检测", value=True)
            include_charts = st.checkbox("包含图表", value=True)
        
        # 生成报告
        st.markdown("---")
        
        if st.button("📄 生成 HTML 报告", type="primary"):
            # 创建报告内容
            report_html = f"""
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
                <div class="metric">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">总行数</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">总列数</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div>
                    <div class="metric-label">内存占用</div>
                </div>
                
                <h2>2. 字段列表</h2>
                <table>
                    <tr><th>序号</th><th>字段名</th><th>数据类型</th><th>非空值</th><th>唯一值</th></tr>
            """
            
            for i, col in enumerate(df.columns):
                report_html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{col}</td>
                        <td>{df[col].dtype}</td>
                        <td>{int(df[col].notna().sum()):,}</td>
                        <td>{int(df[col].nunique()):,}</td>
                    </tr>
                """
            
            report_html += """
                </table>
            """
            
            if include_stats:
                report_html += """
                <h2>3. 数值字段统计</h2>
                <table>
                    <tr><th>字段</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>
                """
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    report_html += f"""
                        <tr>
                            <td>{col}</td>
                            <td>{df[col].mean():.4f}</td>
                            <td>{df[col].median():.4f}</td>
                            <td>{df[col].std():.4f}</td>
                            <td>{df[col].min():.4f}</td>
                            <td>{df[col].max():.4f}</td>
                        </tr>
                    """
                
                report_html += "</table>"
            
            if include_missing:
                report_html += """
                <h2>4. 缺失值分析</h2>
                <table>
                    <tr><th>字段</th><th>缺失数量</th><th>缺失率 (%)</th></tr>
                """
                
                missing_summary = df.isna().sum()
                for col in df.columns:
                    missing_count = int(missing_summary[col])
                    missing_rate = missing_count / len(df) * 100
                    if missing_count > 0:
                        report_html += f"""
                            <tr>
                                <td>{col}</td>
                                <td>{missing_count:,}</td>
                                <td>{missing_rate:.2f}%</td>
                            </tr>
                        """
                
                report_html += "</table>"
            
            report_html += """
            </body>
            </html>
            """
            
            # 提供下载
            st.download_button(
                label="📥 下载 HTML 报告",
                data=report_html,
                file_name=f"数据分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
            
            st.success("✅ 报告生成成功！")
