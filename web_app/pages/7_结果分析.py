"""
结果分析页面

功能：
- 预测结果可视化（分布、阈值分析）
- 与历史预测对比
- 导出分析报告
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
    render_prediction_distribution,
    render_threshold_analysis,
    render_roc_curve,
    render_precision_recall_curve,
)

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="结果分析",
    page_icon="📋",
    layout="wide"
)

# 标题
st.title("📋 结果分析")
st.markdown("---")

# ============================================================================
# 检查是否有预测结果
# ============================================================================

if st.session_state.get('predict_results') is None:
    st.warning("⚠️ 暂无预测结果，请先在【模型预测】页面进行预测")
    
    if st.button("🔮 前往模型预测"):
        st.switch_page("pages/6_模型预测.py")
    st.stop()

results_df = st.session_state.predict_results

# 如果有训练结果，也加载用于对比
has_train_results = st.session_state.get('training_results') is not None

# ============================================================================
# 预测结果概览
# ============================================================================

st.header("📊 预测结果概览")

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

st.markdown("---")

# ============================================================================
# 可视化分析
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 分布分析",
    "🎯 阈值分析",
    "📈 对比分析",
    "📋 导出报告"
])

with tab1:
    st.header("📊 预测分布分析")
    
    # 预测概率分布
    st.subheader("预测概率分布")
    
    fig = render_prediction_distribution(
        results_df['预测概率'].values,
        st.session_state.prediction_threshold
    )
    st.plotly_chart(fig, use_container_width=True, key="tab1_dist_chart")
    
    # 分位数统计
    st.subheader("📋 分位数统计")
    
    proba = results_df['预测概率']
    
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
    
    # 按预测标签分组统计
    st.subheader("📊 分组统计")
    
    group_stats = results_df.groupby('预测标签')['预测概率'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    st.dataframe(group_stats, use_container_width=True)

with tab2:
    st.header("🎯 阈值分析")
    
    # 如果有真实标签，可以进行阈值分析
    if '真实标签' in results_df.columns:
        y_true = results_df['真实标签'].values
        y_scores = results_df['预测概率'].values
        
        fig = render_threshold_analysis(y_true, y_scores)
        st.plotly_chart(fig, use_container_width=True, key="threshold_analysis_chart")
        
        # 不同阈值的指标
        st.subheader("📊 不同阈值的性能指标")
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        threshold_metrics = []
        for thresh in thresholds_to_check:
            y_pred = (y_scores >= thresh).astype(int)
            
            threshold_metrics.append({
                '阈值': thresh,
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1': f1_score(y_true, y_pred, zero_division=0),
                '预测正例': int((y_pred == 1).sum()),
                '预测负例': int((y_pred == 0).sum()),
            })
        
        threshold_df = pd.DataFrame(threshold_metrics)
        st.dataframe(threshold_df.round(4), use_container_width=True, hide_index=True)
        
        # 可视化
        import plotly.express as px
        
        fig = px.line(
            threshold_df,
            x='阈值',
            y=['Precision', 'Recall', 'F1'],
            title='阈值 - 指标曲线',
            labels={'value': '分数', 'variable': '指标'}
        )
        st.plotly_chart(fig, use_container_width=True, key="threshold_metrics_chart")
        
        # ROC 曲线（如果有真实标签）
        st.subheader("📈 ROC 曲线")
        fig = render_roc_curve(y_true, y_scores)
        st.plotly_chart(fig, use_container_width=True, key="roc_curve_chart")
        
        st.subheader("📈 PR 曲线")
        fig = render_precision_recall_curve(y_true, y_scores)
        st.plotly_chart(fig, use_container_width=True, key="pr_curve_chart")
    else:
        st.info("💡 要使用阈值分析功能，请在预测结果中包含真实标签列")
        
        # 模拟阈值分析
        st.subheader("📊 阈值敏感性分析（基于预测概率）")
        
        y_scores = results_df['预测概率'].values
        
        thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        threshold_stats = []
        for thresh in thresholds_to_check:
            y_pred = (y_scores >= thresh).astype(int)
            
            threshold_stats.append({
                '阈值': thresh,
                '预测正例': int((y_pred == 1).sum()),
                '预测负例': int((y_pred == 0).sum()),
                '正例比例 (%)': round((y_pred == 1).mean() * 100, 2),
            })
        
        threshold_df = pd.DataFrame(threshold_stats)
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("📈 对比分析")
    
    if has_train_results:
        train_results = st.session_state.training_results
        
        st.subheader("📊 训练集 vs 预测集对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**训练集**")
            st.metric("样本数", f"{train_results['test_size']:,}")
            st.metric("ROC-AUC", f"{train_results['test_auc']:.4f}")
            st.metric("最佳 F1", f"{train_results['best_f1']:.4f}")
        
        with col2:
            st.info("**预测集**")
            st.metric("样本数", f"{len(results_df):,}")
            st.metric("当前阈值", f"{st.session_state.prediction_threshold:.2f}")
            st.metric("正例比例", f"{(results_df['预测标签'] == 1).mean():.2%}")
        
        # 概率分布对比
        st.subheader("📊 概率分布对比")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=train_results['y_pred_proba'],
            name='训练集',
            opacity=0.6,
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=results_df['预测概率'],
            name='预测集',
            opacity=0.6,
            nbinsx=50
        ))
        
        fig.update_layout(
            title='训练集 vs 预测集 概率分布对比',
            xaxis_title='预测概率',
            yaxis_title='频数',
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="prob_comparison_chart")
    else:
        st.info("💡 要使用对比分析功能，请先进行模型训练")
        
        # 仅显示当前预测的分布
        st.subheader("📊 当前预测分布")
        
        fig = render_prediction_distribution(
            results_df['预测概率'].values,
            st.session_state.prediction_threshold
        )
        st.plotly_chart(fig, use_container_width=True, key="prediction_dist_chart")

with tab4:
    st.header("📋 导出分析报告")
    
    st.markdown("""
    ### 报告内容
    
    - 预测结果概览
    - 预测分布统计
    - 阈值分析（如有真实标签）
    - 与训练集对比（如可用）
    
    """)
    
    # 报告选项
    st.subheader("📝 报告选项")
    
    include_stats = st.checkbox("包含统计信息", value=True)
    include_charts = st.checkbox("包含图表数据", value=True)
    include_comparison = st.checkbox("包含对比分析", value=has_train_results, disabled=not has_train_results)
    
    # 生成报告
    st.markdown("---")
    
    if st.button("📄 生成 HTML 报告", type="primary"):
        # 创建报告内容
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
            <div class="metric">
                <div class="metric-value">{len(results_df):,}</div>
                <div class="metric-label">总样本数</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int((results_df['预测标签'] == 1).sum()):,}</div>
                <div class="metric-label">预测正例</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int((results_df['预测标签'] == 0).sum()):,}</div>
                <div class="metric-label">预测负例</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(results_df['预测标签'] == 1).mean():.2%}</div>
                <div class="metric-label">正例比例</div>
            </div>
            
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
        
        if '真实标签' in results_df.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            y_true = results_df['真实标签'].values
            y_pred = results_df['预测标签'].values
            y_scores = results_df['预测概率'].values
            
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
            <h2>3. 分组统计</h2>
            <table>
                <tr><th>预测标签</th><th>样本数</th><th>平均概率</th><th>概率标准差</th></tr>
            """
            
            for label in [0, 1]:
                group = results_df[results_df['预测标签'] == label]['预测概率']
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
        
        # 提供下载
        st.download_button(
            label="📥 下载 HTML 报告",
            data=report_html,
            file_name=f"预测分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
        
        st.success("✅ 报告生成成功！")
        
        # 同时提供 CSV 下载
        st.markdown("---")
        st.subheader("💾 下载原始数据")
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 下载预测结果 (CSV)",
            data=csv,
            file_name=f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
