"""
测试数据库连接管理器
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.database.connection_manager import ConnectionManager
from model_code.database.db_connector import DatabaseConnector


def test_connection_manager():
    """测试连接管理器基本功能"""
    print("=" * 60)
    print("测试连接管理器")
    print("=" * 60)
    
    # 创建管理器（使用临时文件）
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    manager = ConnectionManager(temp_path)
    
    # 测试 1: 添加连接
    print("\n1️⃣  添加 MySQL 连接...")
    mysql_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'test_db',
        'user': 'root',
        'password': 'password123'
    }
    conn_id1 = manager.add_connection(
        "本地 MySQL",
        mysql_config,
        "本地开发数据库",
        ["开发", "本地"]
    )
    print(f"   ✅ 添加成功，ID: {conn_id1}")
    
    # 测试 2: 添加 PostgreSQL 连接
    print("\n2️⃣  添加 PostgreSQL 连接...")
    pg_config = {
        'db_type': 'postgresql',
        'host': '192.168.1.100',
        'port': 5432,
        'database': 'production',
        'user': 'admin',
        'password': 'secret'
    }
    conn_id2 = manager.add_connection(
        "生产 PostgreSQL",
        pg_config,
        "生产环境数据库",
        ["生产", "远程"]
    )
    print(f"   ✅ 添加成功，ID: {conn_id2}")
    
    # 测试 3: 添加 SQLite 连接
    print("\n3️⃣  添加 SQLite 连接...")
    sqlite_config = {
        'db_type': 'sqlite',
        'database': '/tmp/test.db'
    }
    conn_id3 = manager.add_connection(
        "测试 SQLite",
        sqlite_config,
        "临时测试数据库",
        ["测试"]
    )
    print(f"   ✅ 添加成功，ID: {conn_id3}")
    
    # 测试 4: 获取所有连接
    print("\n4️⃣  获取所有连接...")
    all_conns = manager.get_all_connections()
    print(f"   ✅ 共 {len(all_conns)} 个连接:")
    for conn in all_conns:
        print(f"      - {conn['name']} ({conn['db_type']})")
    
    # 测试 5: 设置激活连接
    print("\n5️⃣  激活 MySQL 连接...")
    manager.set_active_connection(conn_id1)
    active = manager.get_active_connection()
    print(f"   ✅ 当前激活：{active['name']}")
    
    # 测试 6: 搜索连接
    print("\n6️⃣  搜索包含'生产'的连接...")
    results = manager.search_connections("生产")
    print(f"   ✅ 找到 {len(results)} 个:")
    for conn in results:
        print(f"      - {conn['name']}")
    
    # 测试 7: 按类型筛选
    print("\n7️⃣  获取所有 MySQL 连接...")
    mysql_conns = manager.get_connections_by_type('mysql')
    print(f"   ✅ 找到 {len(mysql_conns)} 个 MySQL 连接")
    
    # 测试 8: 统计信息
    print("\n8️⃣  获取统计信息...")
    stats = manager.get_statistics()
    print(f"   ✅ 统计:")
    print(f"      - 总连接数：{stats['total_connections']}")
    print(f"      - 当前激活：{stats['active_connection']}")
    print(f"      - 按类型：{stats['by_type']}")
    
    # 测试 9: 更新连接
    print("\n9️⃣  更新连接描述...")
    manager.update_connection(conn_id1, description="更新后的描述")
    updated = manager.get_connection(conn_id1)
    print(f"   ✅ 新描述：{updated['description']}")
    
    # 测试 10: 删除连接
    print("\n🔟  删除 SQLite 连接...")
    manager.delete_connection(conn_id3)
    print(f"   ✅ 剩余 {len(manager)} 个连接")
    
    # 测试 11: 导出连接
    print("\n1️⃣1️⃣  导出连接到文件...")
    export_path = Path(temp_path).parent / "exported_connections.json"
    manager.export_connections(str(export_path))
    print(f"   ✅ 导出到 {export_path}")
    
    # 测试 12: 导入连接
    print("\n1️⃣2️⃣  从文件导入连接...")
    count = manager.import_connections(str(export_path), merge=True)
    print(f"   ✅ 导入了 {count} 个连接")
    
    # 清理
    Path(temp_path).unlink()
    if export_path.exists():
        export_path.unlink()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)


def test_db_connector():
    """测试数据库连接器（仅测试配置，不实际连接）"""
    print("\n" + "=" * 60)
    print("测试数据库连接器配置")
    print("=" * 60)
    
    # MySQL 配置
    print("\n1️⃣  MySQL 配置测试...")
    mysql_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'test',
        'user': 'root',
        'password': 'pass'
    }
    connector = DatabaseConnector(mysql_config)
    print(f"   ✅ 连接字符串：{connector.connection_string}")
    
    # PostgreSQL 配置
    print("\n2️⃣  PostgreSQL 配置测试...")
    pg_config = {
        'db_type': 'postgresql',
        'host': 'db.example.com',
        'port': 5432,
        'database': 'mydb',
        'user': 'user'
    }
    connector = DatabaseConnector(pg_config)
    print(f"   ✅ 连接字符串：{connector.connection_string}")
    
    # SQLite 配置
    print("\n3️⃣  SQLite 配置测试...")
    sqlite_config = {
        'db_type': 'sqlite',
        'database': '/path/to/database.db'
    }
    connector = DatabaseConnector(sqlite_config)
    print(f"   ✅ 连接字符串：{connector.connection_string}")
    
    print("\n" + "=" * 60)
    print("✅ 配置测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_connection_manager()
    test_db_connector()
