# 新增功能说明

本文档说明 Machine_Learning_Project 的最新增强功能。

> **历史快照说明（2026-10-30 补注）**
>
> 尽管标题写作「最新」，本文实际定格于 2026-03-09，早已不代表现状。当前功能清单、
> 前端形态与启动方式请以 [README.md](README.md) 为准（主线入口已是 Vue + FastAPI，
> Streamlit 仅作回退）。
>
> 下文「4. UI/UX 优化」整节介绍的 `web_app/components/ui_enhancements.py`
> （`set_page_theme` / `show_toast` / `render_progress_bar` / `render_quick_actions`）
> 从未被任何模块导入，属于未接线代码，已于 2026-10-29 在提交 `71ffcf4` 中删除；
> 该节的使用示例已不可运行，仅作历史记录保留。

## 📋 目录

1. [数据库连接支持](#1-数据库连接支持)
2. [模型效果对比](#2-模型效果对比)
3. [自动迭代训练](#3-自动迭代训练)
4. [UI/UX 优化](#4-uiux-优化)
5. [数据生成器](#5-数据生成器)
6. [快速开始](#6-快速开始)

---

## 1. 数据库连接支持

### 功能特性

- ✅ 支持多种数据库：MySQL、PostgreSQL、SQLite、SQL Server
- ✅ 数据库连接配置管理
- ✅ 连接测试功能
- ✅ 表列表展示
- ✅ SQL 查询编辑器
- ✅ 查询结果预览
- ✅ 查询保存/加载
- ✅ SQL 注入防护（参数化查询）
- ✅ 大查询超时处理

### 文件结构

```
model_code/database/
├── __init__.py          # 模块导出
├── db_connector.py      # 数据库连接器
└── query_builder.py     # SQL 查询构建器

web_app/pages/
└── 8_数据源管理.py       # 数据源管理页面
```

### 使用示例

```python
from model_code.database import DatabaseConnector, QueryBuilder

# 连接 SQLite
config = {
    'db_type': 'sqlite',
    'database': 'path/to/database.db'
}

db = DatabaseConnector(config)
db.connect()

# 查询数据
df = db.query("SELECT * FROM users WHERE age > 18")

# 使用查询构建器（防 SQL 注入）
qb = QueryBuilder('users')
sql, params = qb.select('id', 'name').where('age', '>', 18).build()
df = db.query(sql, params)

# 保存查询
db.save_query('active_users', "SELECT * FROM users WHERE status='active'")

db.disconnect()
```

### Web UI 使用

1. 启动 Web 应用
2. 导航到 **🗄️ 数据源管理** 页面
3. 在左侧配置数据库连接
4. 点击"测试连接"
5. 浏览表或编写 SQL 查询

---

## 2. 模型效果对比

### 功能特性

- ✅ 选择多个已训练模型（最多 5 个）
- ✅ 对比指标：AUC、Precision、Recall、F1、准确率
- ✅ ROC 曲线对比图
- ✅ PR 曲线对比图
- ✅ 特征重要性对比
- ✅ 预测结果对比
- ✅ 导出对比报告（文本/CSV）

### 文件结构

```
web_app/
├── components/
│   └── model_comparison.py  # 模型对比组件
└── pages/
    └── 9_模型对比.py        # 模型对比页面
```

### 使用示例

```python
from web_app.components.model_comparison import compare_models

# 对比多个模型
results = compare_models(
    model_paths=['model1.joblib', 'model2.joblib', 'model3.joblib'],
    X_test=X_test,
    y_test=y_test,
    threshold=0.5
)

# 查看结果
for model_name, metrics in results.items():
    print(f"{model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
```

### Web UI 使用

1. 导航到 **⚖️ 模型对比** 页面
2. 选择要对比的模型（至少 2 个）
3. 设置分类阈值
4. 点击"开始对比"
5. 查看可视化对比结果
6. 导出对比报告

---

## 3. 自动迭代训练

### 功能特性

- ✅ 设置迭代次数/停止条件
- ✅ 自动超参数调整
- ✅ 自动特征选择
- ✅ 自动采样比例调整
- ✅ 迭代进度跟踪
- ✅ 最佳模型自动保存
- ✅ 迭代历史可视化
- ✅ 支持中断和恢复

### 文件结构

```
model_code/
└── auto_train.py          # 自动迭代训练模块

web_app/pages/
└── 10_自动迭代.py         # 自动迭代训练页面
```

### 使用示例

```python
from model_code.auto_train import AutoTrainer

# 配置训练
config = {
    'X': X,
    'y': y,
    'model_params': {'n_estimators': 200},
    'test_size': 0.3,
    'random_state': 42
}

# 创建训练器
trainer = AutoTrainer(
    base_config=config,
    max_iterations=10,
    early_stopping_rounds=3,
    min_improvement=0.001
)

# 运行训练
results = trainer.run()

# 查看历史
history_df = trainer.get_history_dataframe()
print(history_df)
```

### Web UI 使用

1. 导航到 **🔄 自动迭代** 页面
2. 配置迭代参数
3. 设置模型参数
4. 点击"开始自动迭代训练"
5. 实时查看训练进度
6. 训练完成后可查看历史趋势

---

## 4. UI/UX 优化

### 功能特性

- ✅ 全局主题切换（亮色/暗色）
- ✅ Toast 通知系统
- ✅ 页面加载进度条
- ✅ 数据刷新按钮
- ✅ 快捷操作菜单
- ✅ 响应式布局优化

### 文件结构

```
web_app/components/
└── ui_enhancements.py     # UI 增强组件
```

### 使用示例

```python
from web_app.components.ui_enhancements import (
    set_page_theme,
    show_toast,
    render_progress_bar,
    render_quick_actions
)

# 设置主题
set_page_theme('dark')

# 显示通知
show_toast("操作成功!", type='success')

# 进度条
render_progress_bar(0.75, text="处理中...")

# 快捷操作
render_quick_actions([
    {'label': '刷新', 'icon': '🔄', 'callback': refresh_data},
    {'label': '导出', 'icon': '📥', 'callback': export_data}
])
```

---

## 5. 数据生成器

### 功能特性

- ✅ 生成分类数据集
- ✅ 可配置样本数、特征数
- ✅ 可配置正负样本比例
- ✅ 添加噪声/异常值
- ✅ 添加缺失值
- ✅ 导出为 CSV/Excel
- ✅ 包含分类特征
- ✅ 包含 ID 列

### 文件结构

```
scripts/
└── generate_test_data.py  # 数据生成脚本
```

### 命令行使用

```bash
# 基本用法
python scripts/generate_test_data.py --samples 10000 --features 20

# 自定义配置
python scripts/generate_test_data.py \
    --samples 5000 \
    --features 15 \
    --positive-ratio 0.15 \
    --missing 0.1 \
    --outliers 0.05 \
    --output ./my_data

# 查看帮助
python scripts/generate_test_data.py --help
```

### Python API 使用

```python
from scripts.generate_test_data import generate_test_data

X, y, df = generate_test_data(
    n_samples=10000,
    n_features=20,
    positive_ratio=0.1,
    missing_rate=0.05,
    outlier_rate=0.02,
    output_dir='./generated_data'
)

print(f"生成 {len(df)} 行数据")
print(f"正样本比例：{df['target'].mean():.2%}")
```

---

## 6. 快速开始

### 安装依赖

```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
source .venv/bin/activate
pip install -r requirements.txt
```

### 生成测试数据

```bash
python scripts/generate_test_data.py --samples 10000 --features 20 --output ./generated_data
```

### 启动 Web 应用

```bash
# macOS/Linux
./run_web.sh

# 或直接运行
streamlit run web_app/app.py
```

### 测试流程

1. **生成测试数据**
   ```bash
   python scripts/generate_test_data.py --samples 10000 --features 20
   ```

2. **上传数据到 Web UI**
   - 启动 Web 应用
   - 前往 **📊 数据管理** 页面
   - 上传生成的 CSV 文件

3. **配置字段**
   - 前往 **⚙️ 配置管理** 页面
   - 设置 ID 字段、标签字段
   - 配置字段处理规则

4. **训练模型**
   - 前往 **🚀 模型训练** 页面
   - 配置参数并训练 3 个不同参数的模型

5. **对比模型**
   - 前往 **⚖️ 模型对比** 页面
   - 选择训练的 3 个模型
   - 查看对比结果

6. **自动迭代**
   - 前往 **🔄 自动迭代** 页面
   - 运行 5 次自动迭代
   - 查看训练历史

---

## 技术细节

### 数据库安全

- **SQL 注入防护**: 使用参数化查询
- **密码加密**: 连接字符串中的密码会被隐藏显示
- **连接池**: 使用 SQLAlchemy 连接池管理
- **超时处理**: 查询超时可配置

### 性能优化

- **向量化计算**: 使用 NumPy/Pandas 向量化操作
- **缓存**: Streamlit 缓存数据库连接
- **增量更新**: 模型对比结果增量显示

### 代码质量

- **类型注解**: 完整的类型提示
- **单元测试**: 包含核心模块测试
- **日志记录**: 详细的运行日志

---

## 更新日志

### v1.1.0 (2026-03-09)

**新增功能:**
- ✅ 数据库连接支持（MySQL/PostgreSQL/SQLite/SQL Server）
- ✅ 模型效果对比功能
- ✅ 自动迭代训练
- ✅ UI/UX 优化（主题切换、通知系统）
- ✅ 测试数据生成器

**改进:**
- ✅ 更新 requirements.txt 添加数据库依赖
- ✅ 更新主应用导航
- ✅ 添加单元测试

**文档:**
- ✅ 新增 NEW_FEATURES.md
- ✅ 更新代码注释

---

## 常见问题

### Q: 如何连接 MySQL 数据库？

```python
config = {
    'db_type': 'mysql',
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'your_password',
    'database': 'your_database'
}
```

### Q: 模型对比支持多少个模型？

最多支持 5 个模型同时对比，以保证可视化效果。

### Q: 自动迭代如何中断？

在 Web UI 中点击"中断训练"按钮，或在代码中调用 `trainer.interrupt()`。

### Q: 生成的数据如何用于训练？

生成的 CSV 文件可以直接在 **📊 数据管理** 页面上传，然后按正常流程配置和训练。

---

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
