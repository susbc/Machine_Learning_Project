# 数据库连接管理功能

> **历史快照说明（2026-11-02 补注）**
>
> 本文是数据库连接管理功能上线时的说明文档，接口与页面形态可能已演进；
> 当前功能与启动方式请以 [README.md](README.md) 为准。

## 📋 功能概述

新增了完整的数据库连接管理功能，支持：

- ✅ 保存历史连接配置
- ✅ 测试数据库连接
- ✅ 灵活切换激活连接
- ✅ 连接搜索和筛选
- ✅ 导入/导出连接配置
- ✅ 使用统计和追踪

## 🗂️ 文件结构

```
Machine_Learning_Project/
├── model_code/
│   └── database/
│       ├── __init__.py              # 模块导出
│       ├── db_connector.py          # 数据库连接器（已有）
│       ├── query_builder.py         # 查询构建器（已有）
│       └── connection_manager.py    # 🆕 连接管理器
├── web_app/
│   └── pages/
│       └── 11_数据库连接管理.py     # 🆕 Web UI 页面
└── scripts/
    └── test_connection_manager.py   # 🆕 测试脚本
```

## 🚀 使用方法

### 1. Web UI 方式（推荐）

启动 Streamlit 应用后，访问 **11_数据库连接管理** 页面：

```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
streamlit run web_app/app.py
```

在页面中可以：
- 添加新数据库连接（MySQL/PostgreSQL/SQLite/SQL Server）
- 测试连接是否可用
- 查看历史连接列表
- 激活/切换连接
- 编辑或删除连接
- 导入/导出连接配置

### 2. 代码方式

```python
from model_code.database import ConnectionManager, DatabaseConnector

# 获取连接管理器
manager = ConnectionManager()

# 添加新连接
config = {
    'db_type': 'mysql',
    'host': 'localhost',
    'port': 3306,
    'database': 'mydb',
    'user': 'root',
    'password': 'password'
}
conn_id = manager.add_connection(
    name="本地 MySQL",
    config=config,
    description="开发数据库",
    tags=["开发", "本地"]
)

# 激活连接
manager.set_active_connection(conn_id)

# 获取当前激活的连接
active = manager.get_active_connection()
print(f"当前激活：{active['name']}")

# 使用连接器连接数据库
connector = DatabaseConnector(active['config'])
if connector.test_connection():
    print("连接成功！")
    # 执行查询
    df = connector.read_table('my_table', limit=100)
    connector.disconnect()
```

## 📊 功能特性

### 连接管理
- **添加连接**: 支持 MySQL、PostgreSQL、SQLite、SQL Server
- **测试连接**: 实时测试数据库连接是否可用
- **激活连接**: 设置当前使用的连接，其他页面可读取
- **编辑连接**: 修改已有连接的配置
- **删除连接**: 移除不需要的连接

### 搜索和筛选
- **关键词搜索**: 在名称、描述、标签中搜索
- **按类型筛选**: 查看特定类型的数据库连接

### 导入/导出
- **导出连接**: 将连接配置导出为 JSON 文件（密码隐藏）
- **导入连接**: 从 JSON 文件导入连接配置

### 统计信息
- 总连接数
- 当前激活连接
- 最常用连接
- 最近使用的连接
- 按类型分布

## 💾 存储位置

连接配置保存在：
```
Machine_Learning_Project/config/saved_connections.json
```

文件格式：
```json
{
  "connections": {
    "conn_id": {
      "id": "conn_id",
      "name": "连接名称",
      "config": { ... },
      "description": "描述",
      "tags": ["标签"],
      "db_type": "mysql",
      "created_at": "2026-03-09T16:00:00",
      "last_used": "2026-03-09T17:00:00",
      "use_count": 5,
      "is_active": true
    }
  },
  "current_connection": "conn_id",
  "last_updated": "2026-03-09T17:00:00"
}
```

## 🧪 运行测试

```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
python3 scripts/test_connection_manager.py
```

测试内容包括：
- 添加/删除连接
- 激活/切换连接
- 搜索和筛选
- 导入/导出
- 统计信息

## 🔒 安全说明

- 密码在导出时自动隐藏（显示为 `***`）
- 连接配置文件建议加入 `.gitignore`
- 生产环境建议使用环境变量存储敏感信息

## 📝 与其他页面集成

在其他页面中读取当前激活的连接：

```python
from model_code.database import get_connection_manager

manager = get_connection_manager()
active = manager.get_active_connection()

if active:
    config = active['config']
    # 使用 config 创建数据库连接
```

## 🎯 下一步优化

- [ ] 在其他页面集成连接选择器
- [ ] 添加连接池管理
- [ ] 支持连接分组/文件夹
- [ ] 添加连接健康检查定时任务
