"""
AI 智能分析页面

功能：
- 数据集智能分析 + 模型推荐
- 训练结果分析 + 调参建议
- 预测结果分析 + 优化方向
- 一键生成分析报告
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.llm import get_intelligent_analyzer, get_llm_manager


# 页面配置
st.set_page_config(
    page_title="AI 智能分析",
    page_icon="🧠",
    layout="wide"
)

# 标题
st.title("🧠 AI 智能分析")
st.markdown("---")


def init_analyzer():
    """初始化分析器"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = get_intelligent_analyzer()
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = get_llm_manager()
    return st.session_state.analyzer


def render_data_analysis_tab(analyzer):
    """数据集分析标签页"""
    st.header("📊 数据集智能分析")
    
    # 检查是否有上传的数据
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ 请先在【数据管理】页面上传数据")
        st.link_button("前往数据管理", "/数据管理")
        return
    
    df = st.session_state.uploaded_data
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 数据概览
        st.subheader("数据概览")
        st.metric("样本数", df.shape[0])
        st.metric("特征数", df.shape[1] - 1)  # 减去目标列
    
    with col2:
        # 目标列选择
        target_col = st.selectbox(
            "选择目标变量",
            df.columns.tolist(),
            index=len(df.columns) - 1 if 'label' in df.columns else 0
        )
    
    # 业务场景输入
    st.subheader("📝 业务场景描述")
    business_context = st.text_area(
        "请描述您的业务场景（可选，但建议填写以获得更精准的建议）",
        placeholder="""例如：
- 这是一个金融风控场景，需要识别潜在的欺诈交易
- 电商用户购买预测，目标是提升营销转化率
- 医疗诊断辅助，需要高召回率避免漏诊
- 客户流失预警，提前识别可能流失的用户""",
        height=100
    )
    
    # 分析按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 分析数据集", use_container_width=True):
            with st.spinner("🤖 AI 正在分析数据集..."):
                analysis = analyzer.analyze_dataset(df, target_col, business_context)
                st.session_state.data_analysis = analysis
                st.success("✅ 分析完成！")
    
    with col2:
        if st.button("🎯 推荐模型", use_container_width=True):
            with st.spinner("🤖 AI 正在推荐模型..."):
                requirements = {
                    'speed_requirement': st.selectbox("预测速度要求", ["低", "中", "高"]),
                    'interpretability_requirement': st.selectbox("可解释性要求", ["低", "中", "高"]),
                    'accuracy_requirement': st.selectbox("准确率要求", ["中", "高", "极高"])
                }
                recommendation = analyzer.recommend_model(df, target_col, business_context, requirements)
                st.session_state.model_recommendation = recommendation
                st.success("✅ 推荐完成！")
    
    with col3:
        if st.button("📄 生成报告", use_container_width=True, key="data_analysis_report"):
            if hasattr(st.session_state, 'data_analysis'):
                filepath = analyzer.save_analysis(
                    'data_analysis',
                    st.session_state.data_analysis,
                    {'dataset': df.shape, 'target': target_col}
                )
                st.success(f"✅ 报告已保存：{filepath}")
            else:
                st.warning("请先执行分析")
    
    # 显示分析结果
    if hasattr(st.session_state, 'data_analysis'):
        st.markdown("---")
        st.subheader("📋 数据分析结果")
        st.markdown(st.session_state.data_analysis)
    
    if hasattr(st.session_state, 'model_recommendation'):
        st.markdown("---")
        st.subheader("🎯 模型推荐结果")
        st.markdown(st.session_state.model_recommendation)


def render_training_analysis_tab(analyzer):
    """训练结果分析标签页"""
    st.header("📈 训练结果智能分析")
    
    # 检查是否有训练结果
    if 'training_metrics' not in st.session_state:
        st.info("ℹ️ 暂无训练结果，请先训练模型")
        
        # 模拟训练结果用于演示
        st.subheader("📝 手动输入训练结果（用于演示）")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_auc = st.number_input("训练集 AUC", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
            train_f1 = st.number_input("训练集 F1", min_value=0.0, max_value=1.0, value=0.92, step=0.01)
            train_size = st.number_input("训练样本数", min_value=100, value=10000, step=100)
        
        with col2:
            val_auc = st.number_input("验证集 AUC", min_value=0.0, max_value=1.0, value=0.88, step=0.01)
            val_f1 = st.number_input("验证集 F1", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
            val_size = st.number_input("验证样本数", min_value=100, value=2000, step=100)
        
        st.session_state.demo_metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_size': train_size,
            'val_size': val_size,
            'n_features': st.number_input("特征数量", min_value=1, value=50)
        }
        
        st.session_state.demo_params = {
            'model_type': 'LightGBM',
            'n_estimators': st.number_input("n_estimators", min_value=10, value=100),
            'max_depth': st.number_input("max_depth", min_value=1, value=6),
            'learning_rate': st.number_input("learning_rate", min_value=0.01, value=0.1, step=0.01),
            'min_child_samples': st.number_input("min_child_samples", min_value=1, value=20)
        }
    
    # 分析按钮
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 分析训练结果", use_container_width=True):
            with st.spinner("🤖 AI 正在分析训练结果..."):
                if 'training_metrics' in st.session_state:
                    metrics = st.session_state.training_metrics
                    params = st.session_state.training_params
                else:
                    metrics = st.session_state.demo_metrics
                    params = st.session_state.demo_params
                
                analysis = analyzer.analyze_training_results(metrics, params)
                st.session_state.training_analysis = analysis
                st.success("✅ 分析完成！")
    
    with col2:
        if st.button("📄 生成报告", use_container_width=True, key="training_analysis_report"):
            if hasattr(st.session_state, 'training_analysis'):
                filepath = analyzer.save_analysis(
                    'training_analysis',
                    st.session_state.training_analysis,
                    {'metrics': st.session_state.get('demo_metrics', {})}
                )
                st.success(f"✅ 报告已保存：{filepath}")
            else:
                st.warning("请先执行分析")
    
    # 显示分析结果
    if hasattr(st.session_state, 'training_analysis'):
        st.markdown("---")
        st.subheader("📋 训练结果分析")
        st.markdown(st.session_state.training_analysis)


def render_hyperparameter_tab(analyzer):
    """超参数调优标签页"""
    st.header("⚙️ 超参数智能调优")
    
    # 当前参数
    st.subheader("当前模型参数")
    current_params = st.text_area(
        "参数字典（JSON 格式）",
        value=json.dumps({
            "model_type": "LightGBM",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }, indent=2),
        height=150
    )
    
    # 历史记录
    st.subheader("历史调参记录")
    history_text = st.text_area(
        "历史记录（每行一个，格式：参数→AUC）",
        placeholder="n_estimators=100,max_depth=6 → AUC: 0.85\nn_estimators=200,max_depth=8 → AUC: 0.87\n...",
        height=150
    )
    
    # 约束条件
    col1, col2 = st.columns(2)
    
    with col1:
        max_time = st.selectbox("最大训练时间", ["10 分钟", "30 分钟", "1 小时", "2 小时"])
    
    with col2:
        compute = st.selectbox("计算资源", ["单 CPU", "多 CPU", "GPU", "多 GPU"])
    
    # 分析按钮
    if st.button("🎯 获取调参建议", use_container_width=True):
        with st.spinner("🤖 AI 正在分析调优方向..."):
            try:
                params = json.loads(current_params)
            except:
                params = {"model_type": "LightGBM"}
            
            # 解析历史记录
            history = []
            for line in history_text.strip().split('\n'):
                if '→' in line and 'AUC' in line:
                    try:
                        parts = line.split('→')
                        auc = float(parts[1].split(':')[1].strip())
                        history.append({'params': parts[0], 'auc': auc})
                    except:
                        pass
            
            constraints = {
                'max_train_time': max_time,
                'compute_resources': compute
            }
            
            analysis = analyzer.suggest_hyperparameter_tuning(params, history, constraints)
            st.session_state.tuning_analysis = analysis
            st.success("✅ 分析完成！")
    
    # 显示结果
    if hasattr(st.session_state, 'tuning_analysis'):
        st.markdown("---")
        st.subheader("💡 调参建议")
        st.markdown(st.session_state.tuning_analysis)


def render_prediction_analysis_tab(analyzer):
    """预测结果分析标签页"""
    st.header("🔮 预测结果智能分析")
    
    # 检查是否有预测结果
    if 'predictions' not in st.session_state:
        st.info("ℹ️ 暂无预测结果，请先进行预测")
        
        # 模拟数据
        import numpy as np
        st.session_state.demo_predictions = np.random.beta(2, 5, 1000)
    
    st.subheader("预测结果统计")
    
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
    else:
        predictions = st.session_state.demo_predictions
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总样本数", len(predictions))
    
    with col2:
        threshold = st.number_input("分类阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        positive = (predictions >= threshold).sum()
        st.metric("预测正例", f"{positive} ({positive/len(predictions)*100:.1f}%)")
    
    with col3:
        st.metric("平均置信度", f"{predictions.mean():.3f}")
    
    # 分布直方图
    st.subheader("预测概率分布")
    
    import numpy as np
    high_conf_pos = (predictions >= 0.8).sum()
    high_conf_neg = (predictions <= 0.2).sum()
    low_conf = ((predictions >= 0.4) & (predictions <= 0.6)).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("高置信度正例 (>0.8)", high_conf_pos)
    
    with col2:
        st.metric("低置信度 (0.4-0.6)", low_conf)
    
    with col3:
        st.metric("高置信度负例 (<0.2)", high_conf_neg)
    
    # 分析按钮
    if st.button("🔍 分析预测结果", use_container_width=True):
        with st.spinner("🤖 AI 正在分析预测结果..."):
            import pandas as pd
            analysis = analyzer.analyze_prediction_results(
                pd.Series(predictions),
                threshold=threshold
            )
            st.session_state.prediction_analysis = analysis
            st.success("✅ 分析完成！")
    
    # 显示结果
    if hasattr(st.session_state, 'prediction_analysis'):
        st.markdown("---")
        st.subheader("📋 预测结果分析")
        st.markdown(st.session_state.prediction_analysis)


def main():
    """主函数"""
    # 初始化
    analyzer = init_analyzer()
    
    # 检查 LLM 配置
    llm_manager = st.session_state.llm_manager
    if len(llm_manager) == 0:
        st.warning("⚠️ 请先配置大模型 API")
        if st.link_button("前往配置", "/大模型配置管理"):
            pass
    
    # 标签页
    tabs = st.tabs([
        "📊 数据集分析",
        "📈 训练结果分析",
        "⚙️ 超参数调优",
        "🔮 预测结果分析"
    ])
    
    with tabs[0]:
        render_data_analysis_tab(analyzer)
    
    with tabs[1]:
        render_training_analysis_tab(analyzer)
    
    with tabs[2]:
        render_hyperparameter_tab(analyzer)
    
    with tabs[3]:
        render_prediction_analysis_tab(analyzer)
    
    # 侧边栏
    with st.sidebar:
        st.header("📚 使用说明")
        
        st.markdown("""
        ### 数据集分析
        1. 上传数据后选择目标变量
        2. 描述业务场景（可选）
        3. 获取数据质量评估和建模建议
        
        ### 训练结果分析
        1. 输入训练/验证集指标
        2. AI 诊断过拟合/欠拟合
        3. 获取优化建议
        
        ### 超参数调优
        1. 输入当前参数
        2. 提供历史记录（可选）
        3. 获取下一步调参方向
        
        ### 预测结果分析
        1. 查看预测分布
        2. AI 分析预测质量
        3. 获取业务洞察
        """)
        
        st.markdown("---")
        
        st.header("🤖 当前模型")
        active = llm_manager.get_active_provider()
        if active:
            st.success(f"✅ {active.name}")
        else:
            st.warning("未配置")


if __name__ == "__main__":
    main()
