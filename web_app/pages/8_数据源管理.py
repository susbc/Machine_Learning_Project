"""
数据源管理页面

功能:
- 数据库连接配置
- 连接测试
- 表列表展示
- SQL 查询编辑器
- 查询结果预览
- 查询保存/加载
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_state import init_session_state

# 初始化
init_session_state()

# 页面配置
st.set_page_config(
    page_title="数据源管理",
    page_icon="🗄️",
    layout="wide"
)

# 标题
st.title("🗄️ 数据源管理")
st.markdown("---")

# ============================================================================
# 导入数据库模块
# ============================================================================

@st.cache_resource
def get_database_connector(config: Dict[str, Any]):
    """获取数据库连接器 (缓存)"""
    from model_code.database import DatabaseConnector
    return DatabaseConnector(config)


# ============================================================================
# 侧边栏 - 数据库配置
# ============================================================================

with st.sidebar:
    st.header("🔌 数据库配置")
    
    # 数据库类型选择
    db_type = st.selectbox(
        "数据库类型",
        options=["mysql", "postgresql", "sqlite", "sqlserver"],
        format_func=lambda x: {
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'sqlite': 'SQLite',
            'sqlserver': 'SQL Server'
        }[x]
    )
    
    # 根据数据库类型显示不同配置
    config = {'db_type': db_type}
    
    if db_type != 'sqlite':
        config['host'] = st.text_input("主机地址", value="localhost")
        config['port'] = st.number_input("端口", value={
            'mysql': 3306,
            'postgresql': 5432,
            'sqlserver': 1433
        }.get(db_type, 3306))
        config['user'] = st.text_input("用户名", value="root")
        config['password'] = st.text_input("密码", type="password")
    
    config['database'] = st.text_input(
        "数据库名" if db_type != 'sqlite' else "数据库文件路径",
        value="test.db" if db_type == 'sqlite' else ""
    )
    
    # 连接测试
    st.markdown("---")
    
    if st.button("🔌 测试连接", use_container_width=True):
        try:
            db = get_database_connector(config)
            if db.test_connection():
                st.success("✅ 连接成功!")
                st.session_state.db_config = config
                st.session_state.db_connected = True
            else:
                st.error("❌ 连接失败")
                st.session_state.db_connected = False
        except Exception as e:
            st.error(f"❌ 连接错误：{str(e)}")
            st.session_state.db_connected = False
    
    # 显示连接状态
    if st.session_state.get('db_connected', False):
        st.success("🟢 已连接")
    else:
        st.info("⚪ 未连接")

# ============================================================================
# 主页面 - 选项卡布局
# ============================================================================

if not st.session_state.get('db_connected', False):
    st.info("👈 请在左侧配置并测试数据库连接")
    st.stop()

# 创建数据库连接器
db = get_database_connector(st.session_state.db_config)

# 创建选项卡
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 表列表",
    "🔍 查询编辑器",
    "💾 保存的查询",
    "📊 数据预览"
])

# ============================================================================
# 选项卡 1: 表列表
# ============================================================================

with tab1:
    st.header("📋 数据库表列表")
    
    # 获取表列表
    tables = db.list_tables()
    
    if tables:
        # 表选择器
        selected_table = st.selectbox(
            "选择表",
            options=tables,
            index=0 if tables else 0
        )
        
        # 显示表信息
        if selected_table:
            cols = st.columns(3)
            
            # 获取列信息
            columns = db.get_table_columns(selected_table)
            
            with cols[0]:
                st.metric("表名", selected_table)
            
            with cols[1]:
                st.metric("列数", len(columns))
            
            # 获取行数
            try:
                count_df = db.query(f"SELECT COUNT(*) as count FROM {selected_table}")
                row_count = count_df['count'].iloc[0] if len(count_df) > 0 else 0
                with cols[2]:
                    st.metric("行数", f"{row_count:,}")
            except:
                pass
            
            # 显示列详情
            st.subheader("列信息")
            if columns:
                col_data = []
                for col in columns:
                    col_data.append({
                        '列名': col.get('name', ''),
                        '类型': str(col.get('type', '')),
                        '可空': '是' if col.get('nullable', True) else '否',
                        '默认值': col.get('default')
                    })
                
                col_df = pd.DataFrame(col_data)
                st.dataframe(col_df, use_container_width=True)
            
            # 数据预览
            st.subheader("数据预览")
            preview_limit = st.slider("预览行数", 10, 500, 100)
            
            try:
                preview_df = db.read_table(selected_table, limit=preview_limit)
                st.dataframe(preview_df, use_container_width=True)
                
                # 导出按钮
                csv = preview_df.to_csv(index=False)
                st.download_button(
                    label="📥 导出为 CSV",
                    data=csv,
                    file_name=f"{selected_table}_preview.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"读取数据失败：{str(e)}")
    else:
        st.warning("未找到任何表")

# ============================================================================
# 选项卡 2: 查询编辑器
# ============================================================================

with tab2:
    st.header("🔍 SQL 查询编辑器")
    
    # 查询编辑器
    default_query = f"SELECT * FROM {tables[0] if tables else 'table_name'} LIMIT 100"
    
    sql_query = st.text_area(
        "SQL 查询",
        value=default_query,
        height=200,
        help="输入 SQL 查询语句"
    )
    
    # 查询参数
    with st.expander("⚙️ 查询参数"):
        use_params = st.checkbox("使用参数化查询")
        if use_params:
            params_json = st.text_area(
                "参数 (JSON 格式)",
                value='{}',
                height=100
            )
            try:
                import json
                params = json.loads(params_json)
            except:
                params = {}
                st.warning("参数格式错误，将使用空参数")
        else:
            params = None
    
    # 执行按钮
    col1, col2 = st.columns([1, 4])
    
    with col1:
        execute_btn = st.button("▶️ 执行查询", type="primary", use_container_width=True)
    
    with col2:
        timeout = st.number_input("查询超时 (秒)", min_value=1, max_value=300, value=30)
    
    if execute_btn:
        with st.spinner("执行查询中..."):
            try:
                # 执行查询
                result_df = db.query(sql_query, params=params)
                
                if len(result_df) > 0:
                    st.success(f"✅ 查询成功，返回 {len(result_df):,} 行")
                    
                    # 显示结果
                    st.dataframe(result_df, use_container_width=True)
                    
                    # 统计信息
                    st.subheader("结果统计")
                    stat_cols = st.columns(3)
                    
                    with stat_cols[0]:
                        st.metric("行数", len(result_df))
                    
                    with stat_cols[1]:
                        st.metric("列数", len(result_df.columns))
                    
                    with stat_cols[2]:
                        mem_usage = result_df.memory_usage(deep=True).sum() / 1024 / 1024
                        st.metric("内存占用", f"{mem_usage:.2f} MB")
                    
                    # 导出选项
                    st.subheader("导出结果")
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 导出 CSV",
                            data=csv,
                            file_name="query_result.csv",
                            mime="text/csv"
                        )
                    
                    with exp_col2:
                        try:
                            excel_buffer = result_df.to_excel(index=False, engine='openpyxl')
                            from io import BytesIO
                            output = BytesIO()
                            result_df.to_excel(output, index=False, engine='openpyxl')
                            st.download_button(
                                label="📥 导出 Excel",
                                data=output.getvalue(),
                                file_name="query_result.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.warning(f"Excel 导出失败：{str(e)}")
                    
                    # 保存到 session state
                    st.session_state.last_query_result = result_df
                    st.session_state.last_query = sql_query
                else:
                    st.info("ℹ️ 查询成功，但没有返回结果")
                
            except Exception as e:
                st.error(f"❌ 查询失败：{str(e)}")
                st.code(sql_query)

# ============================================================================
# 选项卡 3: 保存的查询
# ============================================================================

with tab3:
    st.header("💾 保存的查询")
    
    # 加载保存的查询
    saved_queries = db.load_saved_queries()
    
    if saved_queries:
        # 查询选择器
        query_names = list(saved_queries.keys())
        selected_query_name = st.selectbox("选择已保存的查询", options=query_names)
        
        if selected_query_name:
            query_info = saved_queries[selected_query_name]
            
            st.subheader(f"📝 {selected_query_name}")
            
            # 显示查询信息
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("数据库类型", query_info.get('db_type', 'unknown'))
            
            with col2:
                st.metric("创建时间", query_info.get('created_at', 'unknown'))
            
            # 查询描述
            if query_info.get('description'):
                st.info(f"📌 {query_info['description']}")
            
            # SQL 语句
            st.code(query_info['sql'], language='sql')
            
            # 操作按钮
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ 执行此查询", use_container_width=True):
                    try:
                        result_df = db.query(query_info['sql'])
                        st.session_state.last_query_result = result_df
                        st.success(f"✅ 执行成功，返回 {len(result_df)} 行")
                        st.switch_page("pages/8_数据源管理.py")
                    except Exception as e:
                        st.error(f"❌ 执行失败：{str(e)}")
            
            with col2:
                if st.button("📝 编辑查询", use_container_width=True):
                    st.session_state.edit_query_name = selected_query_name
                    st.session_state.edit_query_sql = query_info['sql']
                    st.rerun()
            
            with col3:
                if st.button("🗑️ 删除查询", use_container_width=True):
                    # 删除查询需要手动编辑 JSON 文件
                    st.warning("删除功能开发中...")
    
    else:
        st.info("暂无保存的查询")
    
    # 保存新查询
    st.markdown("---")
    st.subheader("💾 保存当前查询")
    
    if st.session_state.get('last_query'):
        save_name = st.text_input("查询名称")
        save_desc = st.text_input("查询描述")
        
        if st.button("保存查询"):
            if save_name:
                success = db.save_query(save_name, st.session_state.last_query, save_desc)
                if success:
                    st.success("✅ 查询已保存")
                    st.rerun()
                else:
                    st.error("❌ 保存失败")
            else:
                st.warning("请输入查询名称")
    else:
        st.info("先执行一个查询，然后可以保存")

# ============================================================================
# 选项卡 4: 数据预览
# ============================================================================

with tab4:
    st.header("📊 数据预览")
    
    if st.session_state.get('last_query_result') is not None:
        result_df = st.session_state.last_query_result
        
        st.info(f"上次查询：{st.session_state.get('last_query', 'N/A')}")
        
        # 显示数据
        st.dataframe(result_df, use_container_width=True)
        
        # 统计信息
        st.subheader("数据统计")
        
        # 数值列统计
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**数值列统计:**")
            st.dataframe(result_df[numeric_cols].describe(), use_container_width=True)
        
        # 缺失值统计
        st.write("**缺失值统计:**")
        missing_df = pd.DataFrame({
            '字段': result_df.columns,
            '缺失数': result_df.isnull().sum().values,
            '缺失率 (%)': (result_df.isnull().sum() / len(result_df) * 100).values
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.info("请先在【查询编辑器】执行一个查询")

# ============================================================================
# 底部 - 快捷操作
# ============================================================================

st.markdown("---")
st.subheader("⚡ 快捷操作")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 刷新表列表", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col2:
    if st.session_state.get('last_query_result') is not None:
        if st.button("📊 使用此数据训练", use_container_width=True):
            # 将查询结果设置为训练数据
            st.session_state.uploaded_data = st.session_state.last_query_result
            st.success("✅ 数据已加载，请前往模型训练页面")
            st.switch_page("pages/4_模型训练.py")

with col3:
    if st.button("📋 返回数据管理", use_container_width=True):
        st.switch_page("pages/1_数据管理.py")
