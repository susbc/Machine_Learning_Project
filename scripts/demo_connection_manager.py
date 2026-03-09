"""
数据库连接管理演示脚本

展示如何使用 ConnectionManager 管理数据库连接
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.database import ConnectionManager, DatabaseConnector


def demo_basic_usage():
    """基础使用演示"""
    print("=" * 60)
    print("数据库连接管理 - 基础使用演示")
    print("=" * 60)
    
    # 1. 创建连接管理器
    print("\n1️⃣  创建连接管理器...")
    manager = ConnectionManager()
    print(f"   ✅ 管理器已初始化，当前保存了 {len(manager)} 个连接")
    
    # 2. 添加 MySQL 连接
    print("\n2️⃣  添加 MySQL 连接...")
    mysql_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'mydb',
        'user': 'root',
        'password': 'your_password'
    }
    conn_id = manager.add_connection(
        name="本地 MySQL",
        config=mysql_config,
        description="本地开发环境数据库",
        tags=["开发", "本地", "MySQL"]
    )
    print(f"   ✅ 连接已保存，ID: {conn_id}")
    
    # 3. 添加 PostgreSQL 连接
    print("\n3️⃣  添加 PostgreSQL 连接...")
    pg_config = {
        'db_type': 'postgresql',
        'host': '192.168.1.100',
        'port': 5432,
        'database': 'production_db',
        'user': 'admin',
        'password': 'secret'
    }
    conn_id2 = manager.add_connection(
        name="生产 PostgreSQL",
        config=pg_config,
        description="生产环境数据库",
        tags=["生产", "远程", "PostgreSQL"]
    )
    print(f"   ✅ 连接已保存，ID: {conn_id2}")
    
    # 4. 查看所有连接
    print("\n4️⃣  查看所有连接...")
    connections = manager.get_all_connections()
    for conn in connections:
        status = "✅" if conn.get('is_active') else "⭕"
        print(f"   {status} {conn['name']} ({conn['db_type']}) - {conn.get('description', '')}")
    
    # 5. 激活连接
    print("\n5️⃣  激活 MySQL 连接...")
    manager.set_active_connection(conn_id)
    active = manager.get_active_connection()
    print(f"   ✅ 当前激活：{active['name']}")
    
    # 6. 搜索连接
    print("\n6️⃣  搜索包含'生产'的连接...")
    results = manager.search_connections("生产")
    for conn in results:
        print(f"   🔍 {conn['name']}")
    
    # 7. 获取统计信息
    print("\n7️⃣  统计信息...")
    stats = manager.get_statistics()
    print(f"   📊 总连接数：{stats['total_connections']}")
    print(f"   📊 当前激活：{stats['active_connection']}")
    print(f"   📊 按类型：{stats['by_type']}")
    
    print("\n" + "=" * 60)


def demo_test_connection():
    """测试连接演示"""
    print("\n" + "=" * 60)
    print("数据库连接测试演示")
    print("=" * 60)
    
    # SQLite 测试（不需要真实数据库）
    print("\n1️⃣  测试 SQLite 连接...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name
    
    sqlite_config = {
        'db_type': 'sqlite',
        'database': temp_db
    }
    
    connector = DatabaseConnector(sqlite_config)
    success = connector.test_connection()
    
    if success:
        print("   ✅ SQLite 连接测试成功！")
        
        # 执行简单查询
        df = connector.query("SELECT 1 as test")
        print(f"   📊 查询结果：{df.to_dict()}")
    else:
        print("   ❌ SQLite 连接测试失败")
    
    connector.disconnect()
    
    # 清理
    Path(temp_db).unlink()
    
    print("\n" + "=" * 60)


def demo_integration():
    """集成使用演示"""
    print("\n" + "=" * 60)
    print("与其他页面集成演示")
    print("=" * 60)
    
    print("""
在其他 Streamlit 页面中使用:

```python
import streamlit as st
from model_code.database import get_connection_manager, DatabaseConnector

# 获取连接管理器
manager = get_connection_manager()

# 获取当前激活的连接
active = manager.get_active_connection()

if active:
    st.success(f"已连接：{active['name']}")
    
    # 创建数据库连接器
    connector = DatabaseConnector(active['config'])
    
    # 读取数据
    if st.button("加载数据"):
        df = connector.read_table('my_table', limit=100)
        st.dataframe(df)
    
    connector.disconnect()
else:
    st.warning("请先在【数据库连接管理】页面激活连接")
    st.link_button("前往配置", "/11_数据库连接管理")
```
    """)
    
    print("=" * 60)


if __name__ == "__main__":
    demo_basic_usage()
    demo_test_connection()
    demo_integration()
    
    print("\n✅ 演示完成！")
    print("\n💡 提示:")
    print("   - 运行 Web UI: streamlit run web_app/app.py")
    print("   - 访问页面：11_数据库连接管理")
    print("   - 查看文档：CONNECTION_MANAGER_README.md")
