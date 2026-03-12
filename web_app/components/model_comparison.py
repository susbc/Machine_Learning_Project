"""
模型对比组件

提供多模型性能对比、可视化分析功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, average_precision_score
)


def load_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    加载模型文件
    
    Args:
        path: 模型文件路径
    
    Returns:
        (模型对象，模型信息) 元组
    """
    try:
        # 尝试加载训练摘要
        summary_path = Path(path).parent / f"train_summary_{Path(path).stem.split('_')[-1]}.joblib"
        
        if summary_path.exists():
            summary = joblib.load(summary_path)
            model = summary.get('model')
            if model:
                return model, {
                    'auc': summary.get('test_auc', 0),
                    'f1': summary.get('best_f1', 0),
                    'threshold': summary.get('best_threshold', 0.5),
                    'params': summary.get('best_params', {}),
                    'feature_names': summary.get('feature_names', []),
                    'feature_importances': summary.get('feature_importances', [])
                }
        
        # 直接加载模型
        model = joblib.load(path)
        return model, {}
    
    except Exception as e:
        st.error(f"加载模型失败 {path}: {str(e)}")
        return None, {}


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    评估单个模型
    
    Args:
        model: 模型对象
        X_test: 测试集特征
        y_test: 测试集标签
        threshold: 分类阈值
    
    Returns:
        评估结果字典
    """
    # 预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # 计算指标
    metrics = {
        'auc': roc_auc_score(y_test, y_proba),
        'auc_pr': average_precision_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'accuracy': (y_pred == y_test).mean(),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_proba': y_proba,
        'y_pred': y_pred
    }
    
    return metrics


def compare_models(
    model_paths: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """
    对比多个模型的性能
    
    Args:
        model_paths: 模型文件路径列表
        X_test: 测试集特征
        y_test: 测试集标签
        threshold: 分类阈值
    
    Returns:
        对比结果字典
    """
    results = {}
    
    for path in model_paths:
        model_name = Path(path).stem
        model, info = load_model(path)
        
        if model is None:
            continue
        
        # 评估模型
        metrics = evaluate_model(model, X_test, y_test, threshold)
        
        # 合并信息
        results[model_name] = {
            **metrics,
            'info': info,
            'path': path
        }
    
    return results


def create_metrics_comparison_chart(results: Dict[str, Dict]) -> go.Figure:
    """
    创建指标对比柱状图
    
    Args:
        results: 模型对比结果
    
    Returns:
        Plotly Figure
    """
    metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
    metric_names = {
        'auc': 'ROC-AUC',
        'f1': 'F1 分数',
        'precision': '精确率',
        'recall': '召回率',
        'accuracy': '准确率'
    }
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[name][metric] for name in results.keys()]
        
        fig.add_trace(go.Bar(
            name=metric_names[metric],
            x=list(results.keys()),
            y=values,
            hovertemplate=f'{metric_names[metric]}: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='模型性能指标对比',
        xaxis_title='模型',
        yaxis_title='分数',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_roc_comparison_chart(results: Dict[str, Dict], y_test: pd.Series) -> go.Figure:
    """
    创建 ROC 曲线对比图
    
    Args:
        results: 模型对比结果
        y_test: 测试集标签
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # 添加随机线
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='随机猜测',
        hovertemplate='随机猜测<extra></extra>'
    ))
    
    # 添加各模型 ROC 曲线
    colors = px.colors.qualitative.Set1
    
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        auc = result['auc']
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=f'{name} (AUC={auc:.4f})',
            hovertemplate=f'FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='ROC 曲线对比',
        xaxis_title='假正率 (FPR)',
        yaxis_title='真正率 (TPR)',
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )
    
    return fig


def create_pr_comparison_chart(results: Dict[str, Dict], y_test: pd.Series) -> go.Figure:
    """
    创建 PR 曲线对比图
    
    Args:
        results: 模型对比结果
        y_test: 测试集标签
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (name, result) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_test, result['y_proba'])
        auc_pr = result['auc_pr']
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=f'{name} (AUC-PR={auc_pr:.4f})',
            hovertemplate=f'Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='PR 曲线对比',
        xaxis_title='召回率 (Recall)',
        yaxis_title='精确率 (Precision)',
        height=600,
        showlegend=True
    )
    
    return fig


def create_feature_importance_comparison(results: Dict[str, Dict]) -> go.Figure:
    """
    创建特征重要性对比图
    
    Args:
        results: 模型对比结果
    
    Returns:
        Plotly Figure
    """
    # 收集所有特征名
    all_features = set()
    for name, result in results.items():
        if 'info' in result and 'feature_names' in result['info']:
            all_features.update(result['info']['feature_names'])
    
    if not all_features:
        return go.Figure().add_annotation(text="无特征重要性数据", showarrow=False)
    
    all_features = sorted(list(all_features))
    
    # 构建数据
    data = []
    for name, result in results.items():
        if 'info' in result and 'feature_importances' in result['info']:
            importances = result['info']['feature_importances']
            feature_names = result['info']['feature_names']
            
            importance_dict = dict(zip(feature_names, importances))
            row = [importance_dict.get(f, 0) for f in all_features]
            data.append(row)
    
    if not data:
        return go.Figure().add_annotation(text="无特征重要性数据", showarrow=False)
    
    # 创建热力图
    fig = px.imshow(
        data,
        labels=dict(x="特征", y="模型", color="重要性"),
        x=all_features,
        y=list(results.keys()),
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    
    fig.update_layout(
        title='特征重要性对比',
        height=max(400, len(all_features) * 20),
        xaxis_tickangle=-45
    )
    
    return fig


def create_confusion_matrix_chart(confusion_mat: np.ndarray, class_names: List[str] = None) -> go.Figure:
    """
    创建混淆矩阵热力图
    
    Args:
        confusion_mat: 混淆矩阵
        class_names: 类别名称
    
    Returns:
        Plotly Figure
    """
    if class_names is None:
        class_names = ['负样本', '正样本']
    
    fig = px.imshow(
        confusion_mat,
        labels=dict(x="预测", y="真实", color="数量"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title='混淆矩阵',
        height=400
    )
    
    return fig


def create_model_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    创建模型对比摘要表格
    
    Args:
        results: 模型对比结果
    
    Returns:
        摘要 DataFrame
    """
    data = []
    
    for name, result in results.items():
        data.append({
            '模型': name,
            'ROC-AUC': f"{result['auc']:.6f}",
            'AUC-PR': f"{result['auc_pr']:.6f}",
            '精确率': f"{result['precision']:.6f}",
            '召回率': f"{result['recall']:.6f}",
            'F1 分数': f"{result['f1']:.6f}",
            '准确率': f"{result['accuracy']:.6f}",
            'TP': result['confusion_matrix'][1, 1],
            'FP': result['confusion_matrix'][0, 1],
            'FN': result['confusion_matrix'][1, 0],
            'TN': result['confusion_matrix'][0, 0]
        })
    
    df = pd.DataFrame(data)
    
    # 按 AUC 排序
    df = df.sort_values('ROC-AUC', ascending=False, key=lambda x: x.str.replace(',', '').astype(float))
    
    return df


def export_comparison_report(results: Dict[str, Dict], X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    导出对比报告
    
    Args:
        results: 模型对比结果
        X_test: 测试集特征
        y_test: 测试集标签
    
    Returns:
        报告文本
    """
    report = []
    report.append("=" * 80)
    report.append("模型对比分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 基本信息
    report.append(f"测试集样本数：{len(y_test):,}")
    report.append(f"特征数：{X_test.shape[1]}")
    report.append(f"正样本比例：{y_test.mean():.2%}")
    report.append("")
    
    # 模型列表
    report.append("-" * 80)
    report.append("参评模型")
    report.append("-" * 80)
    for name, result in results.items():
        report.append(f"\n{name}:")
        report.append(f"  路径：{result.get('path', 'N/A')}")
        if 'info' in result:
            if 'params' in result['info']:
                report.append(f"  参数：{result['info']['params']}")
    report.append("")
    
    # 性能对比
    report.append("-" * 80)
    report.append("性能指标对比")
    report.append("-" * 80)
    
    summary_df = create_model_summary_table(results)
    report.append(summary_df.to_string(index=False))
    report.append("")
    
    # 最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    report.append("-" * 80)
    report.append("最佳模型")
    report.append("-" * 80)
    report.append(f"模型名称：{best_model}")
    report.append(f"ROC-AUC: {results[best_model]['auc']:.6f}")
    report.append(f"F1 分数：{results[best_model]['f1']:.6f}")
    report.append("")
    
    # 详细分类报告
    report.append("-" * 80)
    report.append("详细分类报告")
    report.append("-" * 80)
    
    for name, result in results.items():
        report.append(f"\n{name}:")
        report.append(classification_report(
            y_test,
            result['y_pred'],
            target_names=['负样本', '正样本'],
            digits=6
        ))
    
    return "\n".join(report)


def render_model_comparison_section(
    available_models: List[Dict],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    渲染模型对比界面
    
    Args:
        available_models: 可用模型列表
        X_test: 测试集特征
        y_test: 测试集标签
    """
    st.header("🎯 模型选择")
    
    # 模型多选
    model_options = {m['name']: m['path'] for m in available_models}
    selected_models = st.multiselect(
        "选择要对比的模型 (至少 2 个，最多 5 个)",
        options=list(model_options.keys()),
        default=list(model_options.keys())[:2] if len(model_options) >= 2 else [],
        max_selections=5,
        min_selections=2
    )
    
    if len(selected_models) < 2:
        st.warning("请至少选择 2 个模型进行对比")
        return
    
    # 阈值设置
    st.header("⚙️ 对比设置")
    threshold = st.slider(
        "分类阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # 执行对比
    if st.button("🚀 开始对比", type="primary"):
        with st.spinner("正在对比模型..."):
            model_paths = [model_options[name] for name in selected_models]
            results = compare_models(model_paths, X_test, y_test, threshold)
            
            if not results:
                st.error("没有模型成功加载")
                return
            
            st.session_state.comparison_results = results
            
            st.success(f"✅ 完成 {len(results)} 个模型的对比")
            
            # 显示结果
            display_comparison_results(results, y_test)


def display_comparison_results(results: Dict[str, Dict], y_test: pd.Series) -> None:
    """
    显示对比结果
    
    Args:
        results: 对比结果
        y_test: 测试集标签
    """
    # 摘要表格
    st.header("📊 对比结果摘要")
    summary_df = create_model_summary_table(results)
    st.dataframe(summary_df, use_container_width=True)
    
    # 图表展示
    st.header("📈 可视化对比")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "指标对比",
        "ROC 曲线",
        "PR 曲线",
        "特征重要性"
    ])
    
    with tab1:
        fig = create_metrics_comparison_chart(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_roc_comparison_chart(results, y_test)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = create_pr_comparison_chart(results, y_test)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = create_feature_importance_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
    
    # 导出报告
    st.header("💾 导出报告")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_text = export_comparison_report(results, y_test, y_test)
        st.download_button(
            label="📥 下载文本报告",
            data=report_text,
            file_name="model_comparison_report.txt",
            mime="text/plain"
        )
    
    with col2:
        # 导出 CSV
        summary_csv = create_model_summary_table(results).to_csv(index=False)
        st.download_button(
            label="📥 下载对比数据 (CSV)",
            data=summary_csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
