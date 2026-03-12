"""
UI/UX 优化组件

提供:
- 全局主题切换 (亮色/暗色)
- Toast 通知系统
- 页面加载进度条
- 数据刷新按钮
- 快捷操作菜单
"""

import streamlit as st
import time
from typing import Optional, Callable
import base64


def set_page_theme(theme: str = "light") -> None:
    """
    设置页面主题
    
    Args:
        theme: 主题名称 ("light" 或 "dark")
    """
    st.session_state.page_theme = theme
    
    # 使用 Streamlit 内置主题
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
        }
        .stDataFrame, .stMetric {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
        }
        .stDataFrame, .stMetric {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """
    渲染主题切换按钮
    """
    # 初始化主题
    if 'page_theme' not in st.session_state:
        st.session_state.page_theme = 'light'
    
    # 侧边栏主题切换
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎨 外观设置")
        
        theme = st.radio(
            "主题",
            options=["light", "dark"],
            format_func=lambda x: "☀️ 亮色" if x == "light" else "🌙 暗色",
            index=0 if st.session_state.page_theme == "light" else 1,
            horizontal=True
        )
        
        if theme != st.session_state.page_theme:
            set_page_theme(theme)
            st.rerun()


def show_toast(
    message: str,
    type: str = "info",
    duration: int = 3000,
    key: Optional[str] = None
) -> None:
    """
    显示 Toast 通知
    
    Args:
        message: 通知消息
        type: 类型 ("success", "error", "warning", "info")
        duration: 持续时间 (毫秒)
        key: 唯一键
    """
    if key is None:
        key = f"toast_{int(time.time() * 1000)}"
    
    # 图标映射
    icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }
    
    icon = icons.get(type, 'ℹ️')
    
    # 使用 Streamlit 的通知组件
    if type == "success":
        st.success(f"{icon} {message}")
    elif type == "error":
        st.error(f"{icon} {message}")
    elif type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")


def render_progress_bar(
    progress: float,
    text: Optional[str] = None,
    key: str = "progress"
) -> None:
    """
    渲染进度条
    
    Args:
        progress: 进度 (0-1)
        text: 进度文本
        key: 唯一键
    """
    if text is None:
        text = f"进度：{progress:.0%}"
    
    st.progress(progress, text=text)


def render_refresh_button(
    callback: Optional[Callable] = None,
    text: str = "🔄 刷新",
    key: str = "refresh",
    use_container_width: bool = False
) -> bool:
    """
    渲染刷新按钮
    
    Args:
        callback: 点击回调函数
        text: 按钮文本
        key: 唯一键
        use_container_width: 是否占满容器
    
    Returns:
        是否被点击
    """
    clicked = st.button(
        text,
        key=key,
        use_container_width=use_container_width
    )
    
    if clicked and callback:
        callback()
    
    return clicked


def render_quick_actions(actions: list) -> None:
    """
    渲染快捷操作菜单
    
    Args:
        actions: 操作列表，每项为 dict:
            - label: 按钮文本
            - callback: 回调函数
            - icon: 图标 (可选)
            - type: 按钮类型 ("primary", "secondary", etc.)
    """
    cols = st.columns(len(actions))
    
    for i, action in enumerate(actions):
        with cols[i]:
            label = f"{action.get('icon', '')} {action['label']}"
            clicked = st.button(
                label,
                key=f"action_{i}",
                type=action.get('type', 'secondary'),
                use_container_width=True
            )
            
            if clicked and 'callback' in action:
                action['callback']()


def render_loading_spinner(text: str = "加载中...") -> None:
    """
    渲染加载动画
    
    Args:
        text: 加载文本
    """
    with st.spinner(text):
        # 这是一个上下文管理器，用于包装加载操作
        pass


def render_card(
    title: str,
    content: str,
    icon: Optional[str] = None,
    color: Optional[str] = None
) -> None:
    """
    渲染卡片组件
    
    Args:
        title: 卡片标题
        content: 卡片内容
        icon: 图标
        color: 颜色主题
    """
    # 使用 Streamlit 的容器
    with st.container():
        if icon:
            st.markdown(f"### {icon} {title}")
        else:
            st.markdown(f"### {title}")
        
        st.markdown(content)
        st.markdown("---")


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: Optional[str] = None
) -> None:
    """
    渲染指标卡片
    
    Args:
        label: 指标标签
        value: 指标值
        delta: 变化值
        delta_color: 变化颜色
        icon: 图标
    """
    if icon:
        label = f"{icon} {label}"
    
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None
) -> None:
    """
    渲染章节标题
    
    Args:
        title: 标题
        subtitle: 副标题
        icon: 图标
    """
    if icon:
        st.title(f"{icon} {title}")
    else:
        st.title(title)
    
    if subtitle:
        st.markdown(f"*{subtitle}*")
    
    st.markdown("---")


def render_data_table(
    df,
    height: int = 400,
    use_container_width: bool = True,
    hide_index: bool = False,
    column_config: Optional[dict] = None
) -> None:
    """
    渲染数据表格
    
    Args:
        df: DataFrame
        height: 表格高度
        use_container_width: 是否占满容器
        hide_index: 是否隐藏索引
        column_config: 列配置
    """
    st.dataframe(
        df,
        height=height,
        use_container_width=use_container_width,
        hide_index=hide_index,
        column_config=column_config
    )


def render_expander_section(
    title: str,
    content_callback: Callable,
    expanded: bool = False,
    icon: Optional[str] = None
) -> None:
    """
    渲染可折叠章节
    
    Args:
        title: 章节标题
        content_callback: 内容渲染回调
        expanded: 是否默认展开
        icon: 图标
    """
    if icon:
        title = f"{icon} {title}"
    
    with st.expander(title, expanded=expanded):
        content_callback()


def render_tabs_with_icons(tab_configs: list) -> list:
    """
    渲染带图标的选项卡
    
    Args:
        tab_configs: 选项卡配置列表，每项为 (label, icon)
    
    Returns:
        选项卡对象列表
    """
    tab_labels = [f"{config[1]} {config[0]}" for config in tab_configs]
    return st.tabs(tab_labels)


def render_sidebar_navigation(pages: list) -> None:
    """
    渲染侧边栏导航
    
    Args:
        pages: 页面列表，每项为 dict:
            - label: 页面名称
            - icon: 图标
            - page: 页面路径
    """
    st.sidebar.markdown("### 🧭 导航")
    
    for page in pages:
        icon = page.get('icon', '📄')
        label = page.get('label', 'Unknown')
        page_path = page.get('page', '')
        
        if st.sidebar.button(
            f"{icon} {label}",
            key=f"nav_{page_path}",
            use_container_width=True
        ):
            st.switch_page(page_path)


def init_ui_enhancements() -> None:
    """
    初始化 UI 增强功能
    """
    # 初始化主题
    if 'page_theme' not in st.session_state:
        st.session_state.page_theme = 'light'
    
    # 应用主题
    set_page_theme(st.session_state.page_theme)
    
    # 渲染主题切换
    render_theme_toggle()
