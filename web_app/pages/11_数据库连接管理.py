"""
数据库连接管理页面

功能：
- 添加新数据库连接
- 测试连接
- 查看历史连接列表
- 切换激活连接
- 删除连接
- 编辑连接
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.database.db_connector import DatabaseConnector
from model_code.database.connection_manager import ConnectionManager, get_connection_manager


# 页面配置
st.set_page_config(
    page_title="数据库连接管理",
    page_icon="🔗",
    layout="wide"
)

# 标题
st.title("🔗 数据库连接管理")
st.markdown("---")


def init_connection_manager():
    """初始化连接管理器"""
    if 'connection_manager' not in st.session_state:
        st.session_state.connection_manager = get_connection_manager()
    return st.session_state.connection_manager


def render_connection_form(manager: ConnectionManager, edit_id: Optional[str] = None):
    """
    渲染连接配置表单
    
    Args:
        manager: 连接管理器
        edit_id: 编辑模式下的连接 ID
    """
    st.subheader("➕ 添加新连接" if not edit_id else "✏️ 编辑连接")
    
    # 获取编辑的连接数据
    edit_data = None
    if edit_id:
        edit_data = manager.get_connection(edit_id)
    
    with st.form("connection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # 连接名称
            conn_name = st.text_input(
                "连接名称 *",
                value=edit_data['name'] if edit_data else "",
                help="自定义连接名称，便于识别"
            )
            
            # 数据库类型
            db_type = st.selectbox(
                "数据库类型 *",
                ["mysql", "postgresql", "sqlite", "sqlserver"],
                index=["mysql", "postgresql", "sqlite", "sqlserver"].index(
                    edit_data['db_type'] if edit_data else "mysql"
                ) if edit_data else 0
            )
            
            # 描述
            description = st.text_area(
                "描述",
                value=edit_data.get('description', '') if edit_data else "",
                help="可选的连接描述"
            )
        
        with col2:
            # 标签
            tags_str = st.text_input(
                "标签（逗号分隔）",
                value=", ".join(edit_data.get('tags', [])) if edit_data else "",
                help="例如：生产，测试，开发"
            )
        
        st.markdown("---")
        st.markdown("### 🔌 连接配置")
        
        # SQLite 不需要主机配置
        if db_type != 'sqlite':
            col1, col2, col3 = st.columns(3)
            
            with col1:
                host = st.text_input(
                    "主机地址 *",
                    value=edit_data['config'].get('host', 'localhost') if edit_data else "localhost",
                    help="数据库服务器地址"
                )
            
            with col2:
                port = st.number_input(
                    "端口",
                    min_value=1,
                    max_value=65535,
                    value=edit_data['config'].get('port', get_default_port(db_type)) if edit_data else get_default_port(db_type),
                    help="数据库端口"
                )
            
            with col3:
                database = st.text_input(
                    "数据库名 *",
                    value=edit_data['config'].get('database', '') if edit_data else "",
                    help="数据库名称"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                user = st.text_input(
                    "用户名",
                    value=edit_data['config'].get('user', '') if edit_data else "",
                    help="数据库用户名"
                )
            
            with col2:
                password = st.text_input(
                    "密码",
                    type="password",
                    value=edit_data['config'].get('password', '') if edit_data else "",
                    help="数据库密码"
                )
        else:
            # SQLite 配置
            database = st.text_input(
                "SQLite 文件路径 *",
                value=edit_data['config'].get('database', '') if edit_data else "",
                help="SQLite 数据库文件路径"
            )
            host = port = user = password = ""
        
        # 额外参数
        with st.expander("⚙️ 高级选项"):
            extra_params = st.text_area(
                "额外连接参数（JSON 格式）",
                value=str(edit_data['config'].get('extra_params', {})) if edit_data and edit_data['config'].get('extra_params') else "{}",
                help="例如：{\"charset\": \"utf8mb4\"}"
            )
        
        # 表单按钮
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submit_button = st.form_submit_button("💾 保存连接", use_container_width=True)
        
        with col2:
            test_button = st.form_submit_button("🧪 测试连接", use_container_width=True)
        
        with col3:
            if edit_id:
                cancel_button = st.form_submit_button("❌ 取消编辑", use_container_width=True)
            else:
                cancel_button = False
        
        # 处理表单提交
        if submit_button:
            # 验证必填字段
            if not conn_name:
                st.error("❌ 连接名称不能为空")
                return None
            
            if db_type == 'sqlite':
                if not database:
                    st.error("❌ SQLite 文件路径不能为空")
                    return None
                config = {
                    'db_type': db_type,
                    'database': database
                }
            else:
                if not host or not database:
                    st.error("❌ 主机地址和数据库名不能为空")
                    return None
                config = {
                    'db_type': db_type,
                    'host': host,
                    'port': port,
                    'database': database,
                    'user': user,
                    'password': password
                }
                
                # 解析额外参数
                if extra_params and extra_params != "{}":
                    try:
                        import json
                        config['extra_params'] = json.loads(extra_params)
                    except:
                        st.warning("⚠️ 额外参数格式错误，将忽略")
            
            # 保存连接
            tags = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else []
            
            if edit_id:
                # 更新现有连接
                manager.update_connection(edit_id, config=config, name=conn_name, 
                                         description=description, tags=tags)
                st.success(f"✅ 连接 '{conn_name}' 已更新")
            else:
                # 添加新连接
                conn_id = manager.add_connection(conn_name, config, description, tags)
                st.success(f"✅ 连接 '{conn_name}' 已保存 (ID: {conn_id})")
            
            return config
        
        # 测试连接
        if test_button:
            if db_type == 'sqlite':
                if not database:
                    st.error("❌ SQLite 文件路径不能为空")
                    return None
                config = {
                    'db_type': db_type,
                    'database': database
                }
            else:
                if not host or not database:
                    st.error("❌ 主机地址和数据库名不能为空")
                    return None
                config = {
                    'db_type': db_type,
                    'host': host,
                    'port': port,
                    'database': database,
                    'user': user,
                    'password': password
                }
            
            with st.spinner("🔄 正在测试连接..."):
                connector = DatabaseConnector(config)
                success = connector.test_connection()
                connector.disconnect()
            
            if success:
                st.success("✅ 连接测试成功！")
            else:
                st.error("❌ 连接测试失败，请检查配置")
        
        # 取消编辑
        if cancel_button:
            st.session_state.edit_mode = False
            st.session_state.edit_id = None
            st.rerun()
    
    return None


def get_default_port(db_type: str) -> int:
    """获取数据库默认端口"""
    ports = {
        'mysql': 3306,
        'postgresql': 5432,
        'sqlserver': 1433,
        'sqlite': 0
    }
    return ports.get(db_type, 3306)


def render_connection_list(manager: ConnectionManager):
    """渲染连接列表"""
    st.subheader(f"📚 历史连接 ({len(manager)})")
    
    connections = manager.get_all_connections()
    
    if not connections:
        st.info("📭 暂无保存的连接，请添加新连接")
        return
    
    # 按类型分组
    by_type = {}
    for conn in connections:
        db_type = conn.get('db_type', 'unknown')
        if db_type not in by_type:
            by_type[db_type] = []
        by_type[db_type].append(conn)
    
    # 显示统计
    col1, col2, col3 = st.columns(3)
    stats = manager.get_statistics()
    
    with col1:
        st.metric("总连接数", stats['total_connections'])
    
    with col2:
        active = stats['active_connection'] or "无"
        st.metric("当前激活", active)
    
    with col3:
        if stats['most_used']:
            st.metric("最常用", stats['most_used']['name'])
    
    st.markdown("---")
    
    # 按类型显示连接
    for db_type, type_connections in by_type.items():
        with st.expander(f"{get_db_type_icon(db_type)} {db_type.upper()} ({len(type_connections)})"):
            for conn in type_connections:
                render_connection_card(manager, conn)


def render_connection_card(manager: ConnectionManager, conn: Dict[str, Any]):
    """渲染单个连接卡片"""
    conn_id = conn['id']
    name = conn['name']
    db_type = conn.get('db_type', 'unknown')
    is_active = conn.get('is_active', False)
    use_count = conn.get('use_count', 0)
    last_used = conn.get('last_used', 'Never')
    
    # 卡片样式
    card_class = "active-connection" if is_active else ""
    
    col1, col2, col3 = st.columns([4, 1, 2])
    
    with col1:
        # 连接名称和描述
        if is_active:
            st.markdown(f"### ✅ {name} (当前激活)")
        else:
            st.markdown(f"### {name}")
        
        if conn.get('description'):
            st.caption(conn['description'])
        
        # 标签
        if conn.get('tags'):
            tags_html = " ".join([f"`{tag}`" for tag in conn['tags']])
            st.markdown(tags_html, unsafe_allow_html=True)
        
        # 连接信息
        config = conn.get('config', {})
        if db_type != 'sqlite':
            host = config.get('host', 'N/A')
            port = config.get('port', '')
            database = config.get('database', 'N/A')
            st.caption(f"📍 {host}:{port}/{database}")
        else:
            database = config.get('database', 'N/A')
            st.caption(f"📁 {database}")
    
    with col2:
        # 使用统计
        st.metric("使用次数", use_count)
        st.caption(f"上次：{last_used[:16] if last_used else 'N/A'}")
    
    with col3:
        # 操作按钮
        col_a, col_b = st.columns(2)
        
        with col_a:
            if not is_active:
                if st.button("🔌 激活", key=f"activate_{conn_id}", use_container_width=True):
                    manager.set_active_connection(conn_id)
                    st.success(f"已激活连接：{name}")
                    st.rerun()
            else:
                st.button("✅ 已激活", disabled=True, use_container_width=True)
        
        with col_b:
            if st.button("✏️ 编辑", key=f"edit_{conn_id}", use_container_width=True):
                st.session_state.edit_mode = True
                st.session_state.edit_id = conn_id
                st.rerun()
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("🧪 测试", key=f"test_{conn_id}", use_container_width=True):
                with st.spinner("测试中..."):
                    connector = DatabaseConnector(config)
                    success = connector.test_connection()
                    connector.disconnect()
                
                if success:
                    st.success("连接成功！")
                else:
                    st.error("连接失败！")
        
        with col_d:
            if st.button("🗑️ 删除", key=f"delete_{conn_id}", use_container_width=True):
                manager.delete_connection(conn_id)
                st.warning(f"已删除连接：{name}")
                st.rerun()
    
    st.markdown("---")


def get_db_type_icon(db_type: str) -> str:
    """获取数据库类型图标"""
    icons = {
        'mysql': '🐬',
        'postgresql': '🐘',
        'sqlite': '📁',
        'sqlserver': '🖥️'
    }
    return icons.get(db_type, '🔗')


def main():
    """主函数"""
    # 初始化连接管理器
    manager = init_connection_manager()
    
    # 检查编辑模式
    if not hasattr(st.session_state, 'edit_mode'):
        st.session_state.edit_mode = False
        st.session_state.edit_id = None
    
    # 侧边栏
    with st.sidebar:
        st.header("🔧 操作")
        
        if st.button("➕ 新建连接", use_container_width=True):
            st.session_state.edit_mode = False
            st.session_state.edit_id = None
        
        st.markdown("---")
        
        # 搜索
        search_query = st.text_input("🔍 搜索连接")
        if search_query:
            results = manager.search_connections(search_query)
            st.write(f"找到 {len(results)} 个连接")
            for conn in results:
                st.write(f"- {conn['name']} ({conn['db_type']})")
        
        st.markdown("---")
        
        # 导出导入
        st.subheader("📦 导入/导出")
        
        if st.button("📤 导出连接", use_container_width=True):
            from pathlib import Path
            output_path = Path.home() / f"db_connections_export_{st.session_state.get('export_counter', 0)}.json"
            manager.export_connections(str(output_path))
            st.success(f"已导出到 {output_path}")
            st.session_state.export_counter = st.session_state.get('export_counter', 0) + 1
        
        uploaded_file = st.file_uploader("📥 导入连接", type=['json'])
        if uploaded_file:
            import tempfile
            import json
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
                f.write(uploaded_file.getvalue())
                temp_path = f.name
            
            count = manager.import_connections(temp_path)
            st.success(f"导入了 {count} 个连接")
            Path(temp_path).unlink()  # 清理临时文件
        
        st.markdown("---")
        
        # 统计信息
        st.subheader("📊 统计")
        stats = manager.get_statistics()
        st.json(stats)
    
    # 主内容区
    if st.session_state.edit_mode:
        render_connection_form(manager, st.session_state.edit_id)
        st.markdown("---")
    
    render_connection_form(manager)
    
    st.markdown("---")
    
    render_connection_list(manager)


if __name__ == "__main__":
    main()
