"""
模型管理页面

功能：
- 已训练模型列表展示
- 模型详情（参数、指标、特征重要性）
- 模型下载/删除
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_state import init_session_state
from components.charts import render_feature_importance

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="模型管理",
    page_icon="📦",
    layout="wide"
)

# 标题
st.title("📦 模型管理")
st.markdown("---")

# ============================================================================
# 扫描已保存的模型
# ============================================================================

def scan_saved_models() -> list:
    """扫描模型目录中的所有模型文件"""
    model_dir = st.session_state.model_dir
    
    models = []
    
    if model_dir.exists():
        for file_path in model_dir.glob("*.joblib"):
            try:
                # 尝试加载模型信息
                model_data = joblib.load(file_path)
                
                # 判断是模型对象还是训练摘要
                if isinstance(model_data, dict):
                    # 训练摘要
                    model_info = {
                        'name': file_path.stem,
                        'path': str(file_path),
                        'type': 'summary',
                        'timestamp': file_path.stat().st_mtime,
                        'size': file_path.stat().st_size / 1024 / 1024,  # MB
                    }
                    
                    # 提取指标
                    if 'test_auc' in model_data:
                        model_info['auc'] = model_data['test_auc']
                    if 'best_f1' in model_data:
                        model_info['f1'] = model_data['best_f1']
                    if 'model_params' in model_data:
                        model_info['params'] = model_data['model_params']
                    
                    models.append(model_info)
                else:
                    # 直接是模型对象
                    models.append({
                        'name': file_path.stem,
                        'path': str(file_path),
                        'type': 'model',
                        'timestamp': file_path.stat().st_mtime,
                        'size': file_path.stat().st_size / 1024 / 1024,
                    })
            except Exception as e:
                # 无法加载的文件
                models.append({
                    'name': file_path.stem,
                    'path': str(file_path),
                    'type': 'unknown',
                    'timestamp': file_path.stat().st_mtime,
                    'size': file_path.stat().st_size / 1024 / 1024,
                    'error': str(e)
                })
    
    # 按时间排序
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return models

# ============================================================================
# 模型列表展示
# ============================================================================

st.header("📦 已训练模型")

models = scan_saved_models()

if not models:
    st.info("📭 暂无已训练的模型，请先在【模型训练】页面训练模型")
    
    if st.button("🚀 前往模型训练"):
        st.switch_page("pages/4_模型训练.py")
else:
    # 模型列表表格
    model_df = pd.DataFrame(models)
    
    # 格式化显示
    display_df = model_df.copy()
    display_df['训练时间'] = pd.to_datetime(display_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
    display_df['文件大小'] = display_df['size'].apply(lambda x: f"{x:.2f} MB")
    
    # 选择要显示的列
    show_cols = ['name', '训练时间', '文件大小']
    if 'auc' in display_df.columns:
        display_df['AUC'] = display_df['auc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
        show_cols.append('AUC')
    if 'f1' in display_df.columns:
        display_df['F1'] = display_df['f1'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
        show_cols.append('F1')
    
    st.dataframe(
        display_df[show_cols],
        use_container_width=True,
        hide_index=True,
        height=300
    )
    
    st.markdown("---")
    
    # ============================================================================
    # 模型详情和选择
    # ============================================================================
    
    st.header("🔍 模型详情")
    
    # 模型选择器
    selected_model_name = st.selectbox(
        "选择模型",
        [m['name'] for m in models],
        format_func=lambda x: f"{x}"
    )
    
    if selected_model_name:
        selected_model = next(m for m in models if m['name'] == selected_model_name)
        
        # 保存到 session state
        st.session_state.selected_model = selected_model['path']
        
        # 加载模型详情
        try:
            model_data = joblib.load(selected_model['path'])
            
            # 创建选项卡
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 模型指标",
                "⚙️ 模型参数",
                "📈 特征重要性",
                "💾 操作"
            ])
            
            with tab1:
                st.subheader("📊 模型评估指标")
                
                if isinstance(model_data, dict):
                    # 训练摘要
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'test_auc' in model_data:
                            st.metric("ROC-AUC", f"{model_data['test_auc']:.4f}")
                    
                    with col2:
                        if 'best_f1' in model_data:
                            st.metric("最佳 F1", f"{model_data['best_f1']:.4f}")
                    
                    with col3:
                        if 'best_threshold' in model_data:
                            st.metric("最佳阈值", f"{model_data['best_threshold']:.4f}")
                    
                    with col4:
                        if 'test_ap' in model_data:
                            st.metric("AUC-PR", f"{model_data['test_ap']:.4f}")
                    
                    # 训练信息
                    st.subheader("📋 训练信息")
                    
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.metric("训练集大小", f"{model_data.get('train_size', '-'):,}")
                    
                    with info_col2:
                        st.metric("测试集大小", f"{model_data.get('test_size', '-'):,}")
                    
                    # 分类报告
                    if 'y_test' in model_data and 'y_pred' in model_data:
                        from sklearn.metrics import classification_report
                        
                        report = classification_report(
                            model_data['y_test'],
                            model_data['y_pred'],
                            output_dict=True
                        )
                        
                        st.subheader("📋 分类报告")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                else:
                    st.info("这是模型对象文件，不包含评估指标")
            
            with tab2:
                st.subheader("⚙️ 模型参数")
                
                if isinstance(model_data, dict) and 'model_params' in model_data:
                    params_df = pd.DataFrame({
                        '参数': list(model_data['model_params'].keys()),
                        '值': list(model_data['model_params'].values())
                    })
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                elif hasattr(model_data, 'get_params'):
                    params = model_data.get_params()
                    params_df = pd.DataFrame({
                        '参数': list(params.keys()),
                        '值': list(params.values())
                    })
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                else:
                    st.info("无参数信息")
            
            with tab3:
                st.subheader("📈 特征重要性")
                
                if isinstance(model_data, dict):
                    if 'feature_names' in model_data and 'feature_importances' in model_data:
                        fig = render_feature_importance(
                            model_data['feature_names'],
                            model_data['feature_importances'],
                            top_n=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 导出特征重要性
                        importance_df = pd.DataFrame({
                            '特征名': model_data['feature_names'],
                            '重要性': model_data['feature_importances']
                        }).sort_values('重要性', ascending=False)
                        
                        csv = importance_df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载特征重要性 (CSV)",
                            data=csv,
                            file_name=f"feature_importance_{selected_model_name}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("无特征重要性信息")
                elif hasattr(model_data, 'feature_importances_'):
                    if 'feature_names' in st.session_state:
                        fig = render_feature_importance(
                            st.session_state.feature_names,
                            model_data.feature_importances_,
                            top_n=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("需要特征名来显示特征重要性")
                else:
                    st.info("无特征重要性信息")
            
            with tab4:
                st.subheader("💾 模型操作")
                
                # 模型信息
                st.info(f"""
                **模型名称**: {selected_model['name']}  
                **文件路径**: {selected_model['path']}  
                **文件大小**: {selected_model['size']:.2f} MB  
                **创建时间**: {datetime.fromtimestamp(selected_model['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                # 操作按钮
                col1, col2 = st.columns(2)
                
                with col1:
                    # 下载模型
                    with open(selected_model['path'], 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="📥 下载模型文件",
                        data=model_bytes,
                        file_name=f"{selected_model_name}.joblib",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                with col2:
                    # 删除模型
                    if st.button(
                        "🗑️ 删除模型",
                        use_container_width=True,
                        type="secondary"
                    ):
                        try:
                            Path(selected_model['path']).unlink()
                            st.success(f"模型 {selected_model_name} 已删除")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败：{str(e)}")
                
                # 设置为当前使用模型
                st.markdown("---")
                st.subheader("🎯 设为当前模型")
                
                if st.button(
                    "✅ 设为当前使用模型",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.current_model_path = selected_model['path']
                    st.session_state.current_model_data = model_data
                    st.success(f"已将 {selected_model_name} 设为当前模型")
                    
                    st.info("💡 现在可以在【模型预测】页面使用此模型进行预测")
        
        except Exception as e:
            st.error(f"加载模型失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# 从 session state 加载模型
# ============================================================================

st.markdown("---")
st.header("🔄 会话中的模型")

if st.session_state.get('training_results') is not None:
    st.success("✅ 当前会话中有刚训练的模型")
    
    results = st.session_state.training_results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **ROC-AUC**: {results['test_auc']:.4f}  
        **最佳 F1**: {results['best_f1']:.4f}  
        **最佳阈值**: {results['best_threshold']:.4f}
        """)
    
    with col2:
        if st.button("💾 保存当前模型到列表"):
            # 保存模型
            model_path = st.session_state.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(results['model'], model_path)
            st.success(f"模型已保存：{model_path.name}")
            st.rerun()
