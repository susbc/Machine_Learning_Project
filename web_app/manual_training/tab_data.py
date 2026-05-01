"""
tab_data.py — 数据准备 Tab

合并自：
  - web_app/pages/1_数据管理.py（文件上传 + 预览 + 统计 + 导出）
  - web_app/pages/8_数据源管理.py（数据库连接 + SQL 查询）

注意：原 8_数据源管理.py 使用 with st.sidebar 写入侧边栏；本模块改为
顶部 expander，避免污染其它 Tab 的侧边栏。
"""

import streamlit as st


def render() -> None:
    """渲染数据准备 Tab。"""
    sub_file, sub_db = st.tabs(["📁 文件上传", "🗄️ 数据库源"])
    with sub_file:
        _render_file_upload()
    with sub_db:
        _render_db_source()


def _render_file_upload() -> None:
    """子分区：文件上传（来自原 1_数据管理.py）。"""
    from web_app.components.data_upload import render_data_upload_section
    from web_app.utils.file_handler import calculate_data_statistics

    st.header("📊 数据上传")
    uploaded_df = render_data_upload_section()

    if uploaded_df is not None:
        st.session_state.uploaded_data = uploaded_df
        st.session_state.data_statistics = calculate_data_statistics(uploaded_df)
        _render_data_subtabs(uploaded_df)
    elif st.session_state.get("uploaded_data") is not None:
        st.success("✅ 已加载之前上传的数据")
        df = st.session_state.uploaded_data
        _render_data_subtabs(df)
        if st.button("🗑️ 清除已加载数据", key="data_file_clear_btn"):
            st.session_state.uploaded_data = None
            st.session_state.data_statistics = None
            st.rerun()
    else:
        st.info("👆 请上传数据文件开始分析")
        st.markdown(
            """
        ### 📝 使用说明

        1. **支持格式**: CSV、Excel (.xlsx, .xls)
        2. **编码支持**: 自动检测 UTF-8、GBK 等编码
        3. **数据预览**: 上传后可编辑查看前 100 行
        4. **统计分析**: 自动计算缺失率、唯一值、分布等

        ### 💡 提示

        - 上传的数据会保存在当前会话中
        - 可在其它 Tab（字段配置、模型训练等）使用已上传的数据
        """
        )


def _render_data_subtabs(df) -> None:
    """数据预览 / 字段 / 统计 / 导出 四个二级 tab。"""
    from web_app.components.data_upload import (
        render_data_preview,
        render_data_statistics,
        render_field_names,
    )
    from web_app.utils.file_handler import (
        create_download_button,
        create_excel_download_button,
    )

    sub1, sub2, sub3, sub4 = st.tabs(
        ["📋 数据预览", "🏷️ 字段列表", "📈 统计信息", "💾 导出数据"]
    )
    with sub1:
        render_data_preview(df)
    with sub2:
        render_field_names(df)
    with sub3:
        render_data_statistics(df)
    with sub4:
        st.subheader("💾 导出数据")
        col1, col2 = st.columns(2)
        with col1:
            create_download_button(df, "exported_data.csv", "📥 下载 CSV 格式")
        with col2:
            create_excel_download_button(df, "exported_data.xlsx", "📥 下载 Excel 格式")
        st.info("💡 上传的数据将自动保存到会话状态，可在其它 Tab 使用")


@st.cache_resource
def _get_database_connector(config: dict):
    """获取数据库连接器（缓存）。延迟导入 model_code.database 避免文件上传场景白白加载。"""
    from model_code.database import DatabaseConnector

    return DatabaseConnector(config)


def _render_db_source() -> None:
    """子分区：数据库源（来自原 8_数据源管理.py）。

    DB 相关重依赖（model_code/database/*）只在本函数内 import，
    避免文件上传场景白白加载。
    """
    import pandas as pd

    st.header("🗄️ 数据库源")

    # ====== 数据库连接配置（原页面在 sidebar，移到顶部 expander） ======
    with st.expander("🔌 数据库连接配置", expanded=not st.session_state.get("db_connected", False)):
        db_type = st.selectbox(
            "数据库类型",
            options=["mysql", "postgresql", "sqlite", "sqlserver"],
            format_func=lambda x: {
                "mysql": "MySQL",
                "postgresql": "PostgreSQL",
                "sqlite": "SQLite",
                "sqlserver": "SQL Server",
            }[x],
            key="data_db_type",
        )

        config = {"db_type": db_type}

        if db_type != "sqlite":
            cc1, cc2 = st.columns(2)
            with cc1:
                config["host"] = st.text_input("主机地址", value="localhost", key="data_db_host")
                config["user"] = st.text_input("用户名", value="root", key="data_db_user")
            with cc2:
                config["port"] = st.number_input(
                    "端口",
                    value={"mysql": 3306, "postgresql": 5432, "sqlserver": 1433}.get(db_type, 3306),
                    key="data_db_port",
                )
                config["password"] = st.text_input("密码", type="password", key="data_db_password")

        config["database"] = st.text_input(
            "数据库名" if db_type != "sqlite" else "数据库文件路径",
            value="test.db" if db_type == "sqlite" else "",
            key="data_db_database",
        )

        cb1, cb2 = st.columns([1, 3])
        with cb1:
            test_btn = st.button("🔌 测试连接", use_container_width=True, key="data_db_test_btn")
        with cb2:
            if st.session_state.get("db_connected", False):
                st.success("🟢 已连接")
            else:
                st.info("⚪ 未连接")

        if test_btn:
            try:
                db = _get_database_connector(config)
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

    if not st.session_state.get("db_connected", False):
        st.info("👆 请在上方展开「数据库连接配置」并测试连接")
        return

    db = _get_database_connector(st.session_state.db_config)
    tables = db.list_tables()

    sub_tbl, sub_qe, sub_sq, sub_pv = st.tabs(
        ["📋 表列表", "🔍 查询编辑器", "💾 保存的查询", "📊 数据预览"]
    )

    # ===== 子分区 1：表列表 =====
    with sub_tbl:
        _render_db_tables(db, tables, pd)

    # ===== 子分区 2：查询编辑器 =====
    with sub_qe:
        _render_db_query_editor(db, tables, pd)

    # ===== 子分区 3：保存的查询 =====
    with sub_sq:
        _render_db_saved_queries(db)

    # ===== 子分区 4：数据预览 =====
    with sub_pv:
        _render_db_data_preview(pd)

    # ===== 底部快捷操作 =====
    st.markdown("---")
    st.subheader("⚡ 快捷操作")
    qa1, qa2 = st.columns(2)
    with qa1:
        if st.button("🔄 刷新表列表", use_container_width=True, key="data_db_refresh_btn"):
            st.cache_resource.clear()
            st.rerun()
    with qa2:
        if st.session_state.get("last_query_result") is not None:
            if st.button("📊 使用此数据训练", use_container_width=True, key="data_db_use_for_train_btn"):
                st.session_state.uploaded_data = st.session_state.last_query_result
                st.success("✅ 数据已加载，请切换到「4️⃣ 模型训练」Tab")


def _render_db_tables(db, tables, pd) -> None:
    st.subheader("📋 数据库表列表")

    if not tables:
        st.warning("未找到任何表")
        return

    selected_table = st.selectbox("选择表", options=tables, index=0, key="data_db_table_select")
    if not selected_table:
        return

    cols = st.columns(3)
    columns = db.get_table_columns(selected_table)
    with cols[0]:
        st.metric("表名", selected_table)
    with cols[1]:
        st.metric("列数", len(columns))
    try:
        count_df = db.query(f"SELECT COUNT(*) as count FROM {selected_table}")
        row_count = count_df["count"].iloc[0] if len(count_df) > 0 else 0
        with cols[2]:
            st.metric("行数", f"{row_count:,}")
    except Exception:
        pass

    st.markdown("**列信息**")
    if columns:
        col_data = [
            {
                "列名": c.get("name", ""),
                "类型": str(c.get("type", "")),
                "可空": "是" if c.get("nullable", True) else "否",
                "默认值": c.get("default"),
            }
            for c in columns
        ]
        st.dataframe(pd.DataFrame(col_data), use_container_width=True)

    st.markdown("**数据预览**")
    preview_limit = st.slider("预览行数", 10, 500, 100, key="data_db_preview_limit")
    try:
        preview_df = db.read_table(selected_table, limit=preview_limit)
        st.dataframe(preview_df, use_container_width=True)
        csv = preview_df.to_csv(index=False)
        st.download_button(
            label="📥 导出为 CSV",
            data=csv,
            file_name=f"{selected_table}_preview.csv",
            mime="text/csv",
            key="data_db_table_export_btn",
        )
    except Exception as e:
        st.error(f"读取数据失败：{str(e)}")


def _render_db_query_editor(db, tables, pd) -> None:
    import json
    from io import BytesIO

    st.subheader("🔍 SQL 查询编辑器")

    default_query = f"SELECT * FROM {tables[0] if tables else 'table_name'} LIMIT 100"
    sql_query = st.text_area(
        "SQL 查询",
        value=default_query,
        height=200,
        help="输入 SQL 查询语句",
        key="data_db_sql_input",
    )

    with st.expander("⚙️ 查询参数"):
        use_params = st.checkbox("使用参数化查询", key="data_db_use_params")
        if use_params:
            params_json = st.text_area("参数 (JSON 格式)", value="{}", height=100, key="data_db_params_json")
            try:
                params = json.loads(params_json)
            except Exception:
                params = {}
                st.warning("参数格式错误，将使用空参数")
        else:
            params = None

    col1, col2 = st.columns([1, 4])
    with col1:
        execute_btn = st.button("▶️ 执行查询", type="primary", use_container_width=True, key="data_db_execute_btn")
    with col2:
        st.number_input("查询超时 (秒)", min_value=1, max_value=300, value=30, key="data_db_timeout")

    if not execute_btn:
        return

    with st.spinner("执行查询中..."):
        try:
            result_df = db.query(sql_query, params=params)
            if len(result_df) == 0:
                st.info("ℹ️ 查询成功，但没有返回结果")
                return

            st.success(f"✅ 查询成功，返回 {len(result_df):,} 行")
            st.dataframe(result_df, use_container_width=True)

            st.markdown("**结果统计**")
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("行数", len(result_df))
            with stat_cols[1]:
                st.metric("列数", len(result_df.columns))
            with stat_cols[2]:
                mem_usage = result_df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("内存占用", f"{mem_usage:.2f} MB")

            st.markdown("**导出结果**")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 导出 CSV",
                    data=csv,
                    file_name="query_result.csv",
                    mime="text/csv",
                    key="data_db_query_export_csv_btn",
                )
            with exp_col2:
                try:
                    output = BytesIO()
                    result_df.to_excel(output, index=False, engine="openpyxl")
                    st.download_button(
                        label="📥 导出 Excel",
                        data=output.getvalue(),
                        file_name="query_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="data_db_query_export_xlsx_btn",
                    )
                except Exception as e:
                    st.warning(f"Excel 导出失败：{str(e)}")

            st.session_state.last_query_result = result_df
            st.session_state.last_query = sql_query
        except Exception as e:
            st.error(f"❌ 查询失败：{str(e)}")
            st.code(sql_query)


def _render_db_saved_queries(db) -> None:
    st.subheader("💾 保存的查询")

    saved_queries = db.load_saved_queries()
    if saved_queries:
        query_names = list(saved_queries.keys())
        selected_query_name = st.selectbox(
            "选择已保存的查询", options=query_names, key="data_db_saved_select"
        )

        if selected_query_name:
            query_info = saved_queries[selected_query_name]
            st.markdown(f"**📝 {selected_query_name}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("数据库类型", query_info.get("db_type", "unknown"))
            with col2:
                st.metric("创建时间", query_info.get("created_at", "unknown"))

            if query_info.get("description"):
                st.info(f"📌 {query_info['description']}")
            st.code(query_info["sql"], language="sql")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("▶️ 执行此查询", use_container_width=True, key="data_db_saved_execute_btn"):
                    try:
                        result_df = db.query(query_info["sql"])
                        st.session_state.last_query_result = result_df
                        st.session_state.last_query = query_info["sql"]
                        st.success(f"✅ 执行成功，返回 {len(result_df)} 行")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 执行失败：{str(e)}")
            with col2:
                if st.button("📝 编辑查询", use_container_width=True, key="data_db_saved_edit_btn"):
                    st.session_state.edit_query_name = selected_query_name
                    st.session_state.edit_query_sql = query_info["sql"]
                    st.rerun()
            with col3:
                if st.button("🗑️ 删除查询", use_container_width=True, key="data_db_saved_delete_btn"):
                    st.warning("删除功能开发中...")
    else:
        st.info("暂无保存的查询")

    st.markdown("---")
    st.markdown("**💾 保存当前查询**")

    if st.session_state.get("last_query"):
        save_name = st.text_input("查询名称", key="data_db_save_name")
        save_desc = st.text_input("查询描述", key="data_db_save_desc")
        if st.button("保存查询", key="data_db_save_btn"):
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


def _render_db_data_preview(pd) -> None:
    st.subheader("📊 数据预览")

    if st.session_state.get("last_query_result") is None:
        st.info("请先在【查询编辑器】执行一个查询")
        return

    result_df = st.session_state.last_query_result
    st.info(f"上次查询：{st.session_state.get('last_query', 'N/A')}")
    st.dataframe(result_df, use_container_width=True)

    st.markdown("**数据统计**")
    numeric_cols = result_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        st.write("**数值列统计:**")
        st.dataframe(result_df[numeric_cols].describe(), use_container_width=True)

    st.write("**缺失值统计:**")
    missing_df = pd.DataFrame(
        {
            "字段": result_df.columns,
            "缺失数": result_df.isnull().sum().values,
            "缺失率 (%)": (result_df.isnull().sum() / len(result_df) * 100).values,
        }
    )
    st.dataframe(missing_df, use_container_width=True)
