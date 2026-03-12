"""
进度展示组件

提供训练进度、预测进度等实时展示功能
"""

import streamlit as st
import time
from typing import List, Optional
import sys
from io import StringIO


# ============================================================================
# 进度条组件
# ============================================================================

def render_progress_bar(progress: float, label: str = "进度") -> None:
    """
    渲染进度条
    
    Args:
        progress: 进度值 (0.0-1.0)
        label: 进度条标签
    """
    progress = max(0.0, min(1.0, progress))
    st.progress(progress, text=f"{label}: {progress:.1%}")


def render_training_progress(progress: float, logs: List[str]) -> None:
    """
    渲染训练进度（包含日志）
    
    Args:
        progress: 训练进度 (0.0-1.0)
        logs: 训练日志列表
    """
    st.header("🚀 训练进度")
    
    # 进度条
    render_progress_bar(progress, "训练进度")
    
    # 日志区域
    if logs:
        st.subheader("📝 训练日志")
        log_container = st.container()
        with log_container:
            # 使用滚动区域显示日志
            log_text = "\n".join(logs[-50:])  # 显示最近 50 条
            st.code(log_text, language="text")
    
    # 状态指示器
    if progress >= 1.0:
        st.success("✅ 训练完成！")
    elif progress > 0.0:
        st.info("⏳ 训练中...")
    else:
        st.info("⏸️ 等待开始")


# ============================================================================
# 日志流组件
# ============================================================================

class StreamlitLogger:
    """
    Streamlit 日志记录器
    
    将标准输出重定向到 Streamlit 界面
    """
    
    def __init__(self, placeholder, max_lines: int = 100):
        self.placeholder = placeholder
        self.logs = []
        self.max_lines = max_lines
        self.buffer = StringIO()
    
    def write(self, message: str) -> None:
        """写入日志"""
        if message.strip():
            self.logs.append(message.strip())
            # 保留最近的日志
            if len(self.logs) > self.max_lines:
                self.logs = self.logs[-self.max_lines:]
            self._update_display()
    
    def flush(self) -> None:
        """刷新缓冲区"""
        pass
    
    def _update_display(self) -> None:
        """更新显示"""
        log_text = "\n".join(self.logs)
        with self.placeholder:
            st.code(log_text, language="text")
    
    def get_logs(self) -> List[str]:
        """获取日志列表"""
        return self.logs.copy()
    
    def clear(self) -> None:
        """清空日志"""
        self.logs = []
        self._update_display()


# ============================================================================
# 交叉验证结果展示
# ============================================================================

def render_cv_results(cv_results: dict) -> None:
    """
    渲染交叉验证结果
    
    Args:
        cv_results: 交叉验证结果字典
    """
    st.header("📊 交叉验证结果")
    
    if 'params' not in cv_results:
        st.warning("暂无交叉验证结果")
        return
    
    # 转换为 DataFrame
    import pandas as pd
    results_df = pd.DataFrame(cv_results)
    
    # 显示主要指标
    if 'mean_test_score' in results_df.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "最佳 CV 分数",
                f"{results_df['mean_test_score'].max():.4f}"
            )
        
        with col2:
            st.metric(
                "平均 CV 分数",
                f"{results_df['mean_test_score'].mean():.4f}"
            )
        
        with col3:
            st.metric(
                "CV 分数标准差",
                f"{results_df['std_test_score'].mean():.4f}"
            )
        
        # 显示 Top 5 配置
        st.subheader("Top 5 配置")
        top5 = results_df.nlargest(5, 'mean_test_score')[
            ['mean_test_score', 'std_test_score', 'params']
        ]
        st.dataframe(top5, use_container_width=True)
        
        # 可视化
        import plotly.express as px
        fig = px.bar(
            results_df.head(20),
            y='mean_test_score',
            error_y='std_test_score',
            title='交叉验证分数 Top 20',
            labels={'mean_test_score': 'CV 分数'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 训练结果摘要
# ============================================================================

def render_training_summary(results: dict) -> None:
    """
    渲染训练结果摘要
    
    Args:
        results: 训练结果字典
    """
    st.header("📋 训练结果摘要")
    
    # 核心指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'test_auc' in results:
            st.metric("ROC-AUC", f"{results['test_auc']:.4f}")
    
    with col2:
        if 'test_ap' in results:
            st.metric("AUC-PR", f"{results['test_ap']:.4f}")
    
    with col3:
        if 'best_f1' in results:
            st.metric("最佳 F1", f"{results['best_f1']:.4f}")
    
    with col4:
        if 'best_threshold' in results:
            st.metric("最佳阈值", f"{results['best_threshold']:.4f}")
    
    # 最佳参数
    if 'best_params' in results:
        st.subheader("🎯 最佳参数")
        params_df = pd.DataFrame({
            '参数': list(results['best_params'].keys()),
            '值': list(results['best_params'].values())
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    # 分类报告
    if 'classification_report' in results:
        st.subheader("📊 分类报告")
        st.text(results['classification_report'])


# ============================================================================
# 预测进度展示
# ============================================================================

def render_prediction_progress(progress: float, current_step: str = "") -> None:
    """
    渲染预测进度
    
    Args:
        progress: 进度 (0.0-1.0)
        current_step: 当前步骤描述
    """
    st.header("🔮 预测进度")
    
    render_progress_bar(progress, "预测进度")
    
    if current_step:
        st.info(f"当前步骤：{current_step}")
    
    if progress >= 1.0:
        st.success("✅ 预测完成！")
