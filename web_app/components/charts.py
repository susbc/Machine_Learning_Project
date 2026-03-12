"""
图表组件

提供 EDA 可视化、模型评估等图表功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


# ============================================================================
# 数据分布图
# ============================================================================

def render_distribution_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """
    渲染单列分布图
    
    Args:
        df: 数据 DataFrame
        column: 列名
    
    Returns:
        Plotly Figure
    """
    series = df[column].dropna()
    
    if pd.api.types.is_numeric_dtype(series):
        # 数值型：直方图 + KDE
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series,
            name='分布',
            nbinsx=50,
            opacity=0.7
        ))
        fig.update_layout(
            title=f'{column} - 分布图',
            xaxis_title=column,
            yaxis_title='频数',
            showlegend=False
        )
    else:
        # 分类型：条形图
        value_counts = series.value_counts().head(20)
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f'{column} - 类别分布 (Top 20)',
            labels={'x': '频数', 'y': column}
        )
    
    return fig


def render_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    渲染相关性热力图
    
    Args:
        df: 数据 DataFrame
    
    Returns:
        Plotly Figure
    """
    # 只选择数值列
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return None
    
    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title='特征相关性热力图',
        xaxis_title='特征',
        yaxis_title='特征',
        height=max(400, len(corr_matrix.columns) * 30),
    )
    
    return fig


def render_missing_value_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    渲染缺失值热力图
    
    Args:
        df: 数据 DataFrame
    
    Returns:
        Plotly Figure
    """
    # 创建缺失值矩阵（前 50 列）
    columns_to_show = df.columns[:50]
    missing_matrix = df[columns_to_show].isna()
    
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values.astype(int),
        x=columns_to_show,
        y=list(range(min(100, len(df)))),
        colorscale=[[0, '#4472C4'], [1, '#ED7D31']],
        showscale=False,
        hovertemplate='列: %{x}<br>行: %{y}<br>缺失: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='缺失值分布热力图 (前 50 列 × 前 100 行)',
        xaxis_title='字段',
        yaxis_title='样本',
        height=400,
    )
    
    return fig


# ============================================================================
# 异常值检测可视化
# ============================================================================

def render_outlier_boxplot(df: pd.DataFrame, columns: list) -> go.Figure:
    """
    渲染箱线图（异常值检测）
    
    Args:
        df: 数据 DataFrame
        columns: 列名列表
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    for col in columns[:10]:  # 最多显示 10 列
        if pd.api.types.is_numeric_dtype(df[col]):
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='outliers',
                outliermarker={'color': 'red', 'size': 6}
            ))
    
    fig.update_layout(
        title='异常值检测箱线图',
        yaxis_title='数值',
        showlegend=True,
        height=500,
    )
    
    return fig


# ============================================================================
# 模型评估图表
# ============================================================================

def render_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> go.Figure:
    """
    渲染 ROC 曲线
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率
    
    Returns:
        Plotly Figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC 曲线 (AUC = {roc_auc:.4f})',
        line=dict(width=3)
    ))
    
    # 添加随机线
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='随机猜测',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC 曲线',
        xaxis_title='假阳性率 (FPR)',
        yaxis_title='真阳性率 (TPR)',
        showlegend=True,
        width=600,
        height=500,
    )
    
    return fig


def render_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray) -> go.Figure:
    """
    渲染 Precision-Recall 曲线
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率
    
    Returns:
        Plotly Figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR 曲线 (AP = {ap_score:.4f})',
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title='Precision-Recall 曲线',
        xaxis_title='召回率 (Recall)',
        yaxis_title='查准率 (Precision)',
        showlegend=True,
        width=600,
        height=500,
    )
    
    return fig


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    渲染混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        Plotly Figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['预测负例', '预测正例'],
        y=['真实负例', '真实正例'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
    ))
    
    fig.update_layout(
        title='混淆矩阵',
        xaxis_title='预测',
        yaxis_title='真实',
        width=500,
        height=400,
    )
    
    return fig


def render_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 20) -> go.Figure:
    """
    渲染特征重要性图
    
    Args:
        feature_names: 特征名列表
        importances: 重要性数组
        top_n: 显示前 N 个特征
    
    Returns:
        Plotly Figure
    """
    # 排序并取 top N
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importances[indices],
            colorscale='Viridis',
        ),
    ))
    
    fig.update_layout(
        title=f'特征重要性 Top {top_n}',
        xaxis_title='重要性',
        yaxis_title='特征',
        showlegend=False,
        height=max(400, top_n * 30),
    )
    
    return fig


# ============================================================================
# 预测结果可视化
# ============================================================================

def render_prediction_distribution(predictions: np.ndarray, threshold: float = 0.5) -> go.Figure:
    """
    渲染预测结果分布图
    
    Args:
        predictions: 预测概率
        threshold: 分类阈值
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # 负例分布
    neg_mask = predictions < threshold
    fig.add_trace(go.Histogram(
        x=predictions[neg_mask],
        name=f'预测负例 (n={neg_mask.sum()})',
        opacity=0.7,
        marker_color='blue'
    ))
    
    # 正例分布
    pos_mask = ~neg_mask
    fig.add_trace(go.Histogram(
        x=predictions[pos_mask],
        name=f'预测正例 (n={pos_mask.sum()})',
        opacity=0.7,
        marker_color='red'
    ))
    
    # 添加阈值线
    fig.add_shape(
        type="line",
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=max(
            len(predictions[neg_mask]) // 10 if neg_mask.sum() > 0 else 0,
            len(predictions[pos_mask]) // 10 if pos_mask.sum() > 0 else 0
        ),
        line=dict(color="green", width=3, dash="dash"),
    )
    
    fig.update_layout(
        title=f'预测概率分布 (阈值={threshold})',
        xaxis_title='预测概率',
        yaxis_title='样本数',
        barmode='overlay',
        showlegend=True,
    )
    
    return fig


def render_threshold_analysis(y_true: np.ndarray, y_scores: np.ndarray) -> go.Figure:
    """
    渲染阈值分析图
    
    Args:
        y_true: 真实标签
        y_scores: 预测概率
    
    Returns:
        Plotly Figure
    """
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precisions[:-1],
        mode='lines',
        name='Precision',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recalls[:-1],
        mode='lines',
        name='Recall',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_scores[:-1],
        mode='lines',
        name='F1 Score',
        line=dict(color='green', width=3)
    ))
    
    # 标记最佳阈值
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    
    fig.add_shape(
        type="line",
        x0=best_threshold,
        y0=0,
        x1=best_threshold,
        y1=1,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    fig.update_layout(
        title='阈值 - 指标曲线分析',
        xaxis_title='阈值',
        yaxis_title='分数',
        showlegend=True,
        width=700,
        height=500,
    )
    
    return fig
