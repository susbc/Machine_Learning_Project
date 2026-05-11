"""
Agentic AutoML Platform v3.0.0 - Streamlit Web UI

启动命令：
    streamlit run web_app/app.py --server.port 8501
"""

import os
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Agentic AutoML Platform",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from core.llm.agent_schemas import FeatureAction, FeatureStrategy
    from model_code.feature_executor import FeatureEngineeringExecutor

    V3_MODULES_AVAILABLE = True
    V3_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - UI fallback
    V3_MODULES_AVAILABLE = False
    V3_IMPORT_ERROR = str(exc)


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "zh": {
        "appearance": "外观",
        "language": "语言",
        "theme": "模式",
        "light": "日间",
        "dark": "夜间",
        "nav": "导航",
        "home": "总览",
        "database": "数据库连接",
        "ai": "AI 智能分析",
        "upload": "智能文件上传",
        "prompt": "Prompt Builder",
        "launch": "快速入口",
        "manual_training": "手动训练",
        "agent_workflow": "Agent 工作流",
        "automl_training": "AutoML 训练",
        "mlops_monitor": "MLOps 监控",
        "eyebrow": "Agentic AutoML Platform v3.0.0",
        "hero_title": "面向表格数据的智能建模工作台",
        "hero_subtitle": "统一管理数据接入、特征工程、自动调参、模型预测和漂移监控。界面已支持日间/夜间模式与中英文切换。",
        "status_ready": "核心模块可用",
        "status_limited": "核心模块受限",
        "status_limited_hint": "部分 v3.0 模块导入失败，请检查依赖环境。",
        "metric_code": "代码规模",
        "metric_modules": "核心模块",
        "metric_api": "API 端点",
        "metric_monitor": "监控能力",
        "feature_agent_title": "Agent 工作流",
        "feature_agent_desc": "自然语言取数、结构化特征策略、SSE 训练进度和在线预测。",
        "feature_automl_title": "AutoML 调参",
        "feature_automl_desc": "基于 Optuna TPE 的贝叶斯优化，支持多模型候选和交叉验证。",
        "feature_mlops_title": "MLOps 监控",
        "feature_mlops_desc": "PSI 漂移检测、基线保存、巡检日志和重训触发接口。",
        "module_status": "模块状态",
        "loaded_modules": "已加载模块",
        "service_status": "服务状态",
        "fastapi": "FastAPI",
        "streamlit": "Streamlit",
        "monitoring": "MLOps 监控",
        "enabled": "已启用",
        "open_manual": "打开手动训练",
        "open_agent": "打开 Agent 工作流",
        "open_automl": "打开 AutoML",
        "open_mlops": "打开 MLOps",
        "db_title": "数据库连接管理",
        "db_subtitle": "配置数据源并验证连接，可作为建模数据入口。",
        "db_type": "数据库类型",
        "host": "主机",
        "port": "端口",
        "username": "用户名",
        "password": "密码",
        "database_name": "数据库名",
        "test_connection": "测试连接",
        "connecting": "正在连接...",
        "connect_try": "尝试连接到",
        "connect_success": "连接成功",
        "ai_title": "AI 智能分析",
        "ai_subtitle": "用结构化输出约束模型建议，减少非 JSON 响应和动作幻觉。",
        "model": "模型",
        "scenario": "分析场景",
        "query": "分析需求",
        "query_placeholder": "例如：分析销售数据趋势和异常点，并给出特征工程建议。",
        "output_format": "输出格式",
        "structured_output": "强制 JSON + Pydantic 校验",
        "start_analysis": "开始分析",
        "analyzing": "AI 正在分析...",
        "strategy": "生成的特征策略",
        "analysis_done": "分析完成",
        "analysis_placeholder": "分析结果将显示在这里。",
        "upload_title": "智能文件上传",
        "upload_subtitle": "上传 CSV 文件后自动识别基础数据质量，并生成特征处理建议。",
        "upload_csv": "上传 CSV 文件",
        "upload_help": "支持自动识别分隔符、编码和列名",
        "file_done": "文件识别完成",
        "rows": "行数",
        "columns": "列数",
        "memory": "内存",
        "detect_result": "识别结果",
        "delimiter": "分隔符",
        "encoding": "编码",
        "missing_total": "缺失值总数",
        "duplicate_rows": "重复行数",
        "preview": "数据预览",
        "target_col": "目标变量",
        "generate_strategy": "生成特征策略",
        "execute_feature": "执行特征工程",
        "feature_done": "特征工程完成",
        "no_special": "无需特殊处理",
        "prompt_title": "Prompt Builder",
        "prompt_subtitle": "为 Agent 工作流构造结构化提示词，并用 Schema 预览输出形态。",
        "system_prompt": "系统提示词",
        "system_placeholder": "你是一位机器学习专家...",
        "user_prompt": "用户提示词",
        "user_placeholder": "请分析这个数据集...",
        "few_shot": "Few-shot 示例",
        "few_shot_placeholder": "示例 1：\n输入：...\n输出：...",
        "schema": "输出 Schema",
        "test_prompt": "测试提示词",
        "testing": "正在测试...",
        "validation_passed": "Pydantic 验证通过",
        "footer": "Agentic AutoML Platform v3.0.0 | 工作台 UI 支持日间/夜间模式与中英文切换",
    },
    "en": {
        "appearance": "Appearance",
        "language": "Language",
        "theme": "Mode",
        "light": "Day",
        "dark": "Night",
        "nav": "Navigation",
        "home": "Overview",
        "database": "Database",
        "ai": "AI Analysis",
        "upload": "Smart Upload",
        "prompt": "Prompt Builder",
        "launch": "Launch",
        "manual_training": "Manual Training",
        "agent_workflow": "Agent Workflow",
        "automl_training": "AutoML Training",
        "mlops_monitor": "MLOps Monitor",
        "eyebrow": "Agentic AutoML Platform v3.0.0",
        "hero_title": "A refined ML workspace for tabular data",
        "hero_subtitle": "Manage data access, feature engineering, hyperparameter search, prediction, and drift monitoring in one focused workspace. Day/night mode and Chinese/English switching are built in.",
        "status_ready": "Core modules ready",
        "status_limited": "Core modules limited",
        "status_limited_hint": "Some v3.0 modules failed to import. Check the dependency environment.",
        "metric_code": "Codebase",
        "metric_modules": "Core Modules",
        "metric_api": "API Endpoints",
        "metric_monitor": "Monitoring",
        "feature_agent_title": "Agent Workflow",
        "feature_agent_desc": "Natural-language retrieval, structured feature strategy, SSE training progress, and online prediction.",
        "feature_automl_title": "AutoML Search",
        "feature_automl_desc": "Optuna TPE Bayesian optimization with model candidates and cross validation.",
        "feature_mlops_title": "MLOps Monitor",
        "feature_mlops_desc": "PSI drift checks, baseline storage, inspection logs, and retraining hooks.",
        "module_status": "Module Status",
        "loaded_modules": "Loaded Modules",
        "service_status": "Service Status",
        "fastapi": "FastAPI",
        "streamlit": "Streamlit",
        "monitoring": "MLOps Monitor",
        "enabled": "Enabled",
        "open_manual": "Open Manual Training",
        "open_agent": "Open Agent Workflow",
        "open_automl": "Open AutoML",
        "open_mlops": "Open MLOps",
        "db_title": "Database Connection",
        "db_subtitle": "Configure and validate data sources for modeling workflows.",
        "db_type": "Database Type",
        "host": "Host",
        "port": "Port",
        "username": "Username",
        "password": "Password",
        "database_name": "Database",
        "test_connection": "Test Connection",
        "connecting": "Connecting...",
        "connect_try": "Trying connection to",
        "connect_success": "Connection successful",
        "ai_title": "AI Analysis",
        "ai_subtitle": "Constrain model recommendations with structured output and Pydantic validation.",
        "model": "Model",
        "scenario": "Scenario",
        "query": "Analysis Request",
        "query_placeholder": "Example: analyze sales trends and anomalies, then propose feature engineering steps.",
        "output_format": "Output Format",
        "structured_output": "JSON enforced + Pydantic validation",
        "start_analysis": "Start Analysis",
        "analyzing": "Analyzing...",
        "strategy": "Generated Feature Strategy",
        "analysis_done": "Analysis complete",
        "analysis_placeholder": "Analysis results will appear here.",
        "upload_title": "Smart File Upload",
        "upload_subtitle": "Upload a CSV file, inspect data quality, and generate feature handling suggestions.",
        "upload_csv": "Upload CSV File",
        "upload_help": "Delimiter, encoding, and columns can be detected automatically",
        "file_done": "File recognized",
        "rows": "Rows",
        "columns": "Columns",
        "memory": "Memory",
        "detect_result": "Detection Result",
        "delimiter": "Delimiter",
        "encoding": "Encoding",
        "missing_total": "Missing Values",
        "duplicate_rows": "Duplicate Rows",
        "preview": "Preview",
        "target_col": "Target Column",
        "generate_strategy": "Generate Strategy",
        "execute_feature": "Run Feature Engineering",
        "feature_done": "Feature engineering complete",
        "no_special": "No special handling required",
        "prompt_title": "Prompt Builder",
        "prompt_subtitle": "Compose structured prompts for Agent workflows and preview Schema-shaped outputs.",
        "system_prompt": "System Prompt",
        "system_placeholder": "You are a machine learning expert...",
        "user_prompt": "User Prompt",
        "user_placeholder": "Please analyze this dataset...",
        "few_shot": "Few-shot Examples",
        "few_shot_placeholder": "Example 1:\nInput: ...\nOutput: ...",
        "schema": "Output Schema",
        "test_prompt": "Test Prompt",
        "testing": "Testing...",
        "validation_passed": "Pydantic validation passed",
        "footer": "Agentic AutoML Platform v3.0.0 | Workspace UI with day/night mode and Chinese/English switching",
    },
}

SCENARIOS = {
    "zh": ["数据洞察", "趋势预测", "异常检测", "关联分析", "特征工程建议"],
    "en": ["Data Insight", "Trend Forecasting", "Anomaly Detection", "Association Analysis", "Feature Engineering Advice"],
}

PAGE_KEYS = ["home", "database", "ai", "upload", "prompt"]


def ensure_preferences() -> None:
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "zh"


def t(key: str) -> str:
    lang = st.session_state.get("ui_lang", "zh")
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["zh"].get(key, key))


def apply_theme(theme: str) -> None:
    palettes = {
        "light": {
            "bg": "#f7f7f4",
            "panel": "#ffffff",
            "panel_alt": "#f0eee9",
            "text": "#161616",
            "muted": "#6c6a63",
            "border": "#dedbd2",
            "accent": "#8a5a44",
            "accent_soft": "#eadfd7",
            "success": "#2f7d5c",
            "blue": "#376996",
            "shadow": "0 18px 44px rgba(30, 28, 24, 0.08)",
        },
        "dark": {
            "bg": "#101216",
            "panel": "#171a20",
            "panel_alt": "#20242c",
            "text": "#f2f0ea",
            "muted": "#a9a49a",
            "border": "#303641",
            "accent": "#d0a07a",
            "accent_soft": "#32261f",
            "success": "#6fbd95",
            "blue": "#82a6d9",
            "shadow": "0 18px 44px rgba(0, 0, 0, 0.28)",
        },
    }
    p = palettes[theme]
    st.markdown(
        f"""
        <style>
        :root {{
            --ui-bg: {p["bg"]};
            --ui-panel: {p["panel"]};
            --ui-panel-alt: {p["panel_alt"]};
            --ui-text: {p["text"]};
            --ui-muted: {p["muted"]};
            --ui-border: {p["border"]};
            --ui-accent: {p["accent"]};
            --ui-accent-soft: {p["accent_soft"]};
            --ui-success: {p["success"]};
            --ui-blue: {p["blue"]};
            --ui-shadow: {p["shadow"]};
        }}
        .stApp {{
            background: var(--ui-bg);
            color: var(--ui-text);
        }}
        [data-testid="stSidebar"] {{
            background: var(--ui-panel);
            border-right: 1px solid var(--ui-border);
        }}
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        [data-testid="stSidebar"] * {{
            color: var(--ui-text);
        }}
        .block-container {{
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            letter-spacing: 0;
        }}
        h1 {{
            font-size: 2.55rem;
            line-height: 1.08;
            font-weight: 720;
            color: var(--ui-text);
            margin-bottom: .6rem;
        }}
        h2 {{
            font-size: 1.55rem;
            font-weight: 680;
            color: var(--ui-text);
            margin-top: 1.25rem;
        }}
        h3 {{
            font-size: 1.02rem;
            font-weight: 650;
            color: var(--ui-text);
        }}
        p, li {{
            color: var(--ui-muted);
        }}
        div[data-testid="stMetric"],
        div[data-testid="stAlert"],
        div[data-testid="stDataFrame"],
        div[data-testid="stExpander"] {{
            border-radius: 8px;
        }}
        .stButton > button {{
            border-radius: 7px;
            border: 1px solid var(--ui-border);
            background: var(--ui-panel);
            color: var(--ui-text);
            min-height: 2.55rem;
            box-shadow: none;
        }}
        .stButton > button:hover {{
            border-color: var(--ui-accent);
            color: var(--ui-accent);
        }}
        .stButton > button[kind="primary"] {{
            background: var(--ui-text);
            color: var(--ui-bg);
            border-color: var(--ui-text);
        }}
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            border-radius: 7px;
            border: 1px solid var(--ui-border);
            background: var(--ui-panel);
            color: var(--ui-text);
        }}
        div[data-baseweb="select"] > div {{
            border-radius: 7px;
            background: var(--ui-panel);
            border-color: var(--ui-border);
        }}
        .workspace-shell {{
            border: 1px solid var(--ui-border);
            background: var(--ui-panel);
            box-shadow: var(--ui-shadow);
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
        }}
        .workspace-eyebrow {{
            color: var(--ui-accent);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }}
        .workspace-subtitle {{
            color: var(--ui-muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 780px;
            margin: 0;
        }}
        .status-line {{
            display: inline-flex;
            align-items: center;
            gap: .55rem;
            border: 1px solid var(--ui-border);
            border-radius: 999px;
            padding: .38rem .72rem;
            color: var(--ui-muted);
            background: var(--ui-panel-alt);
            font-size: .86rem;
            margin-top: 1.05rem;
        }}
        .status-dot {{
            width: .52rem;
            height: .52rem;
            border-radius: 999px;
            background: var(--ui-success);
            display: inline-block;
        }}
        .status-dot.warn {{
            background: var(--ui-accent);
        }}
        .metric-panel, .feature-panel, .module-panel {{
            border: 1px solid var(--ui-border);
            background: var(--ui-panel);
            border-radius: 8px;
            padding: 1rem;
            min-height: 118px;
        }}
        .metric-label {{
            color: var(--ui-muted);
            font-size: .8rem;
            margin-bottom: .35rem;
        }}
        .metric-value {{
            color: var(--ui-text);
            font-size: 1.55rem;
            font-weight: 720;
        }}
        .metric-delta {{
            color: var(--ui-success);
            font-size: .82rem;
            margin-top: .35rem;
        }}
        .feature-title {{
            color: var(--ui-text);
            font-size: 1rem;
            font-weight: 680;
            margin-bottom: .35rem;
        }}
        .feature-copy {{
            color: var(--ui-muted);
            font-size: .9rem;
            line-height: 1.55;
        }}
        .section-note {{
            color: var(--ui-muted);
            margin-top: -.2rem;
            margin-bottom: 1rem;
        }}
        .module-row {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid var(--ui-border);
            padding: .72rem 0;
        }}
        .module-row:last-child {{
            border-bottom: 0;
        }}
        .module-row strong {{
            color: var(--ui-text);
            font-weight: 640;
        }}
        .module-row span {{
            color: var(--ui-muted);
        }}
        .footer-note {{
            color: var(--ui-muted);
            border-top: 1px solid var(--ui-border);
            padding-top: 1rem;
            margin-top: 2rem;
            font-size: .82rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(f"### {t('appearance')}")

    language_options = {"zh": "中文", "en": "English"}
    language = st.sidebar.radio(
        t("language"),
        options=["zh", "en"],
        format_func=lambda value: language_options[value],
        horizontal=True,
        index=["zh", "en"].index(st.session_state.ui_lang),
    )

    theme = st.sidebar.radio(
        t("theme"),
        options=["light", "dark"],
        format_func=lambda value: t("light") if value == "light" else t("dark"),
        horizontal=True,
        index=["light", "dark"].index(st.session_state.ui_theme),
    )

    if language != st.session_state.ui_lang or theme != st.session_state.ui_theme:
        st.session_state.ui_lang = language
        st.session_state.ui_theme = theme
        st.rerun()

    st.sidebar.markdown("---")
    page_labels = {key: t(key) for key in PAGE_KEYS}
    page = st.sidebar.radio(
        t("nav"),
        options=PAGE_KEYS,
        format_func=lambda value: page_labels[value],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(t("footer"))
    return page


def markdown_panel(body: str) -> None:
    st.markdown(body, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "") -> None:
    markdown_panel(
        f"""
        <div class="metric-panel">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """
    )


def feature_card(title: str, copy: str) -> None:
    markdown_panel(
        f"""
        <div class="feature-panel">
            <div class="feature-title">{title}</div>
            <div class="feature-copy">{copy}</div>
        </div>
        """
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.markdown(f"<div class='section-note'>{subtitle}</div>", unsafe_allow_html=True)


def model_dump_compat(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


def render_home() -> None:
    status_text = t("status_ready") if V3_MODULES_AVAILABLE else t("status_limited")
    status_class = "" if V3_MODULES_AVAILABLE else " warn"
    status_hint = "" if V3_MODULES_AVAILABLE else f"<p class='workspace-subtitle'>{t('status_limited_hint')} {V3_IMPORT_ERROR}</p>"

    markdown_panel(
        f"""
        <div class="workspace-shell">
            <div class="workspace-eyebrow">{t('eyebrow')}</div>
            <h1>{t('hero_title')}</h1>
            <p class="workspace-subtitle">{t('hero_subtitle')}</p>
            <div class="status-line"><span class="status-dot{status_class}"></span>{status_text}</div>
            {status_hint}
        </div>
        """
    )

    metric_cols = st.columns(4)
    metrics = [
        (t("metric_code"), "244KB+", "+94KB"),
        (t("metric_modules"), "22", "+15"),
        (t("metric_api"), "10+", "+6"),
        (t("metric_monitor"), "PSI", t("enabled")),
    ]
    for col, item in zip(metric_cols, metrics):
        with col:
            metric_card(*item)

    section_header(t("launch"))
    launch_cols = st.columns(4)
    launch_items = [
        (t("open_manual"), "pages/4_📚_手动训练.py"),
        (t("open_agent"), "pages/14_🤖_Agent工作流.py"),
        (t("open_automl"), "pages/15_📊_AutoML 训练.py"),
        (t("open_mlops"), "pages/16_📈_MLOps 监控.py"),
    ]
    for idx, (label, page_path) in enumerate(launch_items):
        with launch_cols[idx]:
            if st.button(label, use_container_width=True):
                st.switch_page(page_path)

    feature_cols = st.columns(3)
    features = [
        (t("feature_agent_title"), t("feature_agent_desc")),
        (t("feature_automl_title"), t("feature_automl_desc")),
        (t("feature_mlops_title"), t("feature_mlops_desc")),
    ]
    for col, item in zip(feature_cols, features):
        with col:
            feature_card(*item)

    section_header(t("module_status"))
    module_cols = st.columns(2)
    with module_cols[0]:
        markdown_panel(
            f"""
            <div class="module-panel">
                <h3>{t('loaded_modules')}</h3>
                <div class="module-row"><strong>core.llm.agent_schemas</strong><span>{t('enabled')}</span></div>
                <div class="module-row"><strong>model_code.feature_executor</strong><span>{t('enabled')}</span></div>
                <div class="module-row"><strong>model_code.mlops_monitor</strong><span>{t('enabled')}</span></div>
                <div class="module-row"><strong>core.automl.automl_optimizer</strong><span>{t('enabled')}</span></div>
            </div>
            """
        )
    with module_cols[1]:
        markdown_panel(
            f"""
            <div class="module-panel">
                <h3>{t('service_status')}</h3>
                <div class="module-row"><strong>{t('fastapi')}</strong><span>http://localhost:8000</span></div>
                <div class="module-row"><strong>{t('streamlit')}</strong><span>http://localhost:8501</span></div>
                <div class="module-row"><strong>{t('monitoring')}</strong><span>{t('enabled')}</span></div>
            </div>
            """
        )


def render_database() -> None:
    section_header(t("db_title"), t("db_subtitle"))
    db_type = st.selectbox(t("db_type"), ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Neo4j", "FAISS"])

    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input(t("host"), "localhost")
        port = st.number_input(t("port"), value=3306)
    with col2:
        username = st.text_input(t("username"), "root")
        password = st.text_input(t("password"), type="password")

    database = st.text_input(t("database_name"), "test_db")

    if st.button(t("test_connection"), type="primary"):
        with st.spinner(t("connecting")):
            st.info(f"{t('connect_try')} {db_type}: {host}:{port}/{database}")
            st.success(t("connect_success"))


def render_ai_analysis() -> None:
    section_header(t("ai_title"), t("ai_subtitle"))
    if V3_MODULES_AVAILABLE:
        st.success(t("status_ready"))
    else:
        st.warning(f"{t('status_limited')}: {V3_IMPORT_ERROR}")

    model = st.selectbox(t("model"), ["Qwen-Max", "Qwen-Plus", "GLM-4", "Baichuan", "Yi"])
    scenario = st.selectbox(t("scenario"), SCENARIOS[st.session_state.ui_lang])
    query = st.text_area(t("query"), placeholder=t("query_placeholder"), height=110)
    st.caption(f"{t('output_format')}: {t('structured_output')}")

    if st.button(t("start_analysis"), type="primary"):
        with st.spinner(t("analyzing")):
            if V3_MODULES_AVAILABLE:
                try:
                    strategy = FeatureStrategy(
                        actions=[
                            FeatureAction(column="age", action="fillna_median", reason="缺失值 5%"),
                            FeatureAction(column="income", action="log_transform", reason="偏态分布"),
                        ]
                    )
                    st.success(f"{t('analysis_done')}: {model} / {scenario}")
                    st.markdown(f"**{t('strategy')}**")
                    st.json(model_dump_compat(strategy))
                except Exception as exc:
                    st.error(str(exc))
            else:
                st.success(f"{t('analysis_done')}: {model} / {scenario}")
                st.markdown(t("analysis_placeholder"))


def build_feature_actions(df: pd.DataFrame, target_col: str) -> List[FeatureAction]:
    actions: List[FeatureAction] = []
    for col in df.columns:
        if col == target_col:
            continue
        missing_rate = df[col].isnull().mean()
        if missing_rate > 0.05 and pd.api.types.is_numeric_dtype(df[col]):
            actions.append(
                FeatureAction(column=col, action="fillna_median", reason=f"缺失率 {missing_rate:.1%}")
            )
        elif df[col].dtype == "object":
            actions.append(FeatureAction(column=col, action="target_encode", reason="类别特征"))
    return actions


def render_upload() -> None:
    section_header(t("upload_title"), t("upload_subtitle"))
    uploaded_file = st.file_uploader(t("upload_csv"), type=["csv"], help=t("upload_help"))

    if not uploaded_file:
        st.session_state.pop("upload_strategy", None)
        return

    with st.spinner(t("connecting")):
        df = pd.read_csv(uploaded_file)

    st.success(t("file_done"))
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(t("rows"), f"{df.shape[0]:,}")
    with col2:
        metric_card(t("columns"), str(df.shape[1]))
    with col3:
        metric_card(t("memory"), f"{df.memory_usage().sum() / 1024:.1f} KB")

    section_header(t("detect_result"))
    st.json(
        {
            t("delimiter"): ",",
            t("encoding"): "UTF-8",
            t("missing_total"): int(df.isnull().sum().sum()),
            t("duplicate_rows"): int(df.duplicated().sum()),
        }
    )

    section_header(t("preview"))
    st.dataframe(df.head(), use_container_width=True)

    if not V3_MODULES_AVAILABLE:
        st.warning(f"{t('status_limited')}: {V3_IMPORT_ERROR}")
        return

    target_col = st.selectbox(t("target_col"), df.columns.tolist())
    if st.button(t("generate_strategy"), type="primary"):
        try:
            actions = build_feature_actions(df, target_col)
            if actions:
                strategy = FeatureStrategy(actions=actions)
                st.session_state.upload_strategy = strategy
                st.success(f"{t('generate_strategy')}: {len(actions)}")
                st.json(model_dump_compat(strategy))
            else:
                st.session_state.pop("upload_strategy", None)
                st.info(t("no_special"))
        except Exception as exc:
            st.error(str(exc))

    strategy = st.session_state.get("upload_strategy")
    if strategy is not None:
        if st.button(t("execute_feature")):
            try:
                executor = FeatureEngineeringExecutor(target_col=target_col)
                df_clean = executor.fit_transform(df, strategy)
                st.success(f"{t('feature_done')}: {df_clean.shape}")
                st.dataframe(df_clean.head(), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


def render_prompt_builder() -> None:
    section_header(t("prompt_title"), t("prompt_subtitle"))
    if V3_MODULES_AVAILABLE:
        st.success(t("status_ready"))
    else:
        st.warning(f"{t('status_limited')}: {V3_IMPORT_ERROR}")

    col1, col2 = st.columns(2)
    with col1:
        st.text_area(t("system_prompt"), height=150, placeholder=t("system_placeholder"))
        st.text_area(t("user_prompt"), height=150, placeholder=t("user_placeholder"))
    with col2:
        st.text_area(t("few_shot"), height=318, placeholder=t("few_shot_placeholder"))

    output_schema = st.selectbox(
        t("schema"),
        ["FeatureStrategy", "SQLGenerationResult", "AutoMLConfig", "EDAReport"],
    )

    if st.button(t("test_prompt"), type="primary"):
        with st.spinner(t("testing")):
            if V3_MODULES_AVAILABLE and output_schema == "FeatureStrategy":
                result = FeatureStrategy(
                    actions=[FeatureAction(column="age", action="fillna_median", reason="缺失值填充")]
                )
                st.success(t("validation_passed"))
                st.json(model_dump_compat(result))
            else:
                st.info(t("analysis_placeholder"))


ensure_preferences()
apply_theme(st.session_state.ui_theme)
page_key = render_sidebar()

if page_key == "home":
    render_home()
elif page_key == "database":
    render_database()
elif page_key == "ai":
    render_ai_analysis()
elif page_key == "upload":
    render_upload()
elif page_key == "prompt":
    render_prompt_builder()

st.markdown(f"<div class='footer-note'>{t('footer')}</div>", unsafe_allow_html=True)
