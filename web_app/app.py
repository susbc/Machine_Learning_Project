"""
机器学习 Web 应用 - 主应用

基于 Streamlit 构建的机器学习流水线 Web UI
"""

import streamlit as st
from pathlib import Path
import sys

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入工具函数
from utils.session_state import init_session_state

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="机器学习平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ml-project',
        'Report a bug': 'https://github.com/your-repo/ml-project/issues',
        'About': "# 机器学习 Web 平台\n基于 Streamlit 构建的完整机器学习流水线 UI"
    }
)

# ============================================================================
# 初始化会话状态
# ============================================================================

init_session_state()

# ============================================================================
# 侧边栏导航
# ============================================================================

with st.sidebar:
    st.title("🤖 ML 平台")
    st.markdown("---")
    
    # 主导航
    st.page_link("app.py", label="🏠 首页", icon="🏠")
    st.page_link("pages/1_数据管理.py", label="📊 数据管理", icon="📊")
    st.page_link("pages/2_配置管理.py", label="⚙️ 配置管理", icon="⚙️")
    st.page_link("pages/3_数据分析.py", label="📈 数据分析", icon="📈")
    
    st.markdown("---")
    
    # 模型相关
    st.page_link("pages/4_模型训练.py", label="🚀 模型训练", icon="🚀")
    st.page_link("pages/5_模型管理.py", label="📦 模型管理", icon="📦")
    st.page_link("pages/6_模型预测.py", label="🔮 模型预测", icon="🔮")
    
    st.markdown("---")
    
    # 结果分析
    st.page_link("pages/7_结果分析.py", label="📋 结果分析", icon="📋")
    
    st.markdown("---")
    
    # 高级功能
    st.page_link("pages/8_数据源管理.py", label="🗄️ 数据源管理", icon="🗄️")
    st.page_link("pages/9_模型对比.py", label="⚖️ 模型对比", icon="⚖️")
    st.page_link("pages/10_自动迭代.py", label="🔄 自动迭代", icon="🔄")
    
    st.markdown("---")
    
    # 数据库连接
    st.page_link("pages/11_数据库连接管理.py", label="🔗 数据库连接", icon="🔗")
    
    st.markdown("---")
    
    # AI 智能分析
    st.page_link("pages/12_大模型配置管理.py", label="🤖 大模型配置", icon="🤖")
    st.page_link("pages/13_AI_智能分析.py", label="🧠 AI 智能分析", icon="🧠")
    
    st.markdown("---")
    
    # 状态信息
    st.subheader("📌 状态")
    
    if st.session_state.get('uploaded_data') is not None:
        st.success("✅ 数据已加载")
    else:
        st.info("⏳ 等待数据上传")
    
    if st.session_state.get('field_config') is not None:
        st.success("✅ 配置已加载")
    else:
        st.info("⏳ 等待配置")
    
    if st.session_state.get('training_results') is not None:
        st.success("✅ 模型已训练")
    else:
        st.info("⏳ 未训练模型")

# ============================================================================
# 主页面内容
# ============================================================================

st.title("🤖 机器学习 Web 平台")

st.markdown("""
### 欢迎使用机器学习 Web 平台！

本平台提供完整的机器学习流水线 Web 界面，包括：

#### 📊 数据管理
- 上传 CSV/Excel 数据文件
- 数据预览和统计信息
- 字段名自动识别

#### ⚙️ 配置管理
- 在线编辑字段配置
- 异常值处理、缺失值填充
- 编码方式和标准化设置

#### 📈 数据分析
- EDA 可视化（分布图、相关性热力图）
- 异常值检测
- 分析报告生成

#### 🚀 模型训练
- LightGBM 模型训练
- 超参数调优
- 交叉验证
- 实时训练进度

#### 📦 模型管理
- 已训练模型列表
- 模型详情和指标
- 模型下载/删除

#### 🔮 模型预测
- 上传预测数据
- 选择训练模型
- 预测结果下载

#### 📋 结果分析
- 预测结果可视化
- 阈值分析
- 历史对比

#### 🔗 数据库连接 (新增 🆕)
- 多数据库连接管理（MySQL/PostgreSQL/SQLite/SQL Server）
- 连接配置保存和切换
- 一键测试连接

#### 🤖 大模型配置 (新增 🆕)
- 配置各大模型 API（OpenAI/Claude/通义千问/智谱 AI/百度）
- 多模型管理和切换
- 连接测试

#### 🧠 AI 智能分析 (新增 🆕)
- 数据集智能分析 + 模型推荐
- 训练结果诊断 + 调参建议
- 预测结果分析 + 业务洞察
- 超参数智能调优

---

### 快速开始

#### 基础流程
1. 前往 **📊 数据管理** 上传您的数据
2. 在 **⚙️ 配置管理** 配置字段规则
3. 在 **📈 数据分析** 探索数据特征
4. 在 **🚀 模型训练** 训练模型
5. 在 **🔮 模型预测** 使用模型进行预测

#### 高级功能 (新增)
- 🔗 **数据库连接**: 直接连接数据库读取数据
- 🤖 **大模型配置**: 配置 AI 分析所需的 LLM API
- 🧠 **AI 智能分析**: 获取 AI 驱动的数据分析和调参建议

""")

# 快捷操作
st.markdown("### ⚡ 快捷操作")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 前往数据管理", use_container_width=True):
        st.switch_page("pages/1_数据管理.py")

with col2:
    if st.button("🚀 开始模型训练", use_container_width=True):
        st.switch_page("pages/4_模型训练.py")

with col3:
    if st.button("🔮 进行预测", use_container_width=True):
        st.switch_page("pages/6_模型预测.py")

# 项目信息
st.markdown("---")
st.markdown("""
**技术栈**: Streamlit | Plotly | LightGBM | scikit-learn | OpenAI | Anthropic

**版本**: 2.0.0 | **最后更新**: 2026-03-09

**新增功能**:
- 🔗 数据库连接管理（支持 MySQL/PostgreSQL/SQLite/SQL Server）
- 🤖 大模型集成（OpenAI/Claude/通义千问/智谱 AI/百度）
- 🧠 AI 智能分析（数据诊断/模型推荐/调参建议）
""")
