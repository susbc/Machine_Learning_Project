# 手动训练页面合并设计

**日期**：2026-05-01
**作者**：renguanjie + Claude
**状态**：待审阅

## 背景与目标

`web_app/pages/` 下当前有 16 个页面，其中页面 1-9 共同构成"手动训练"工作流：数据上传 / 配置 / EDA / 训练 / 模型管理 / 预测 / 结果分析。每个页面是独立 Streamlit 脚本，侧边栏显得拥挤，跨页面跳转打断流程。

**目标**：将页面 1-9 合并为一个带 Tab 的页面，按业务流划分为 6 个 Tab，原页面文件移至备份目录。

**非目标**：

- 不动 14/15/16（Agent / AutoML / MLOps），它们独立保留
- 不动 11（DB 连接管理）、12（大模型配置）、13（AI 智能分析），它们是基础设施配置
- 不修改 `model_code/`、`core/`、`backend/`
- 不引入新单元测试框架

## §1 目录结构

```
web_app/
├── pages/
│   └── 4_📚_手动训练.py            ← 唯一入口（薄壳，~80 行）
│
├── manual_training/                ← 新增模块目录
│   ├── __init__.py
│   ├── tab_data.py                 ← 1_数据管理 + 8_数据源管理
│   ├── tab_config.py               ← 2_配置管理
│   ├── tab_eda.py                  ← 3_数据分析
│   ├── tab_train.py                ← 4_模型训练
│   ├── tab_model_mgmt.py           ← 5_模型管理 + 9_模型对比
│   └── tab_predict.py              ← 6_模型预测 + 7_结果分析
│
└── _legacy_pages/                  ← 备份（不在 pages/ 下，Streamlit 不索引）
    ├── 1_数据管理.py
    ├── 2_配置管理.py
    ├── 3_数据分析.py
    ├── 4_模型训练.py
    ├── 5_模型管理.py
    ├── 6_模型预测.py
    ├── 7_结果分析.py
    ├── 8_数据源管理.py
    └── 9_模型对比.py
```

**关键约束**：

- `set_page_config` 只在入口页调用一次
- `init_session_state()` 只在入口页调用一次
- 每个 `tab_*.py` 暴露统一接口 `def render() -> None`
- `tab_*.py` 不调用 `set_page_config` / `init_session_state` / `st.title` / `sys.path.insert`

## §2 接口契约 & 命名规范

### `tab_*.py` 必须遵守的契约

```python
"""
tab_<name>.py — 手动训练流程的 <名称> 子模块

来自原页面：web_app/pages/<原文件名>.py
"""
import streamlit as st

def render() -> None:
    """
    渲染本 tab 内容。
    前置条件：
      - 调用方已执行 init_session_state()
      - 调用方已设置 st.set_page_config()
    禁止：
      - 调用 st.set_page_config() / init_session_state()
      - 调用 st.title()
      - 操作 sys.path
    """
    ...
```

### 入口页 `pages/4_📚_手动训练.py` 骨架

```python
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from web_app.utils.session_state import init_session_state
from web_app.manual_training import (
    tab_data, tab_config, tab_eda,
    tab_train, tab_model_mgmt, tab_predict,
)

st.set_page_config(page_title="手动训练流程", page_icon="📚", layout="wide")
init_session_state()
st.title("📚 手动训练流程")

TAB_LABELS = [
    "1️⃣ 数据准备", "2️⃣ 字段配置", "3️⃣ 数据探索",
    "4️⃣ 模型训练", "5️⃣ 模型管理与对比", "6️⃣ 预测与结果",
]
tabs = st.tabs(TAB_LABELS)

with tabs[0]: tab_data.render()
with tabs[1]: tab_config.render()
with tabs[2]: tab_eda.render()
with tabs[3]: tab_train.render()
with tabs[4]: tab_model_mgmt.render()
with tabs[5]: tab_predict.render()
```

### 多源合并 tab 的内部子分区

合并多个原页面的 tab，内部用二级 `st.tabs` 区分：

| Tab 模块 | 内部子分区 |
|---|---|
| `tab_data.py` | `st.tabs(["📁 文件上传", "🗄️ 数据库源"])` |
| `tab_model_mgmt.py` | `st.tabs(["📦 模型列表", "⚖️ 模型对比"])` |
| `tab_predict.py` | `st.tabs(["🔮 预测", "📋 结果分析"])` |

### Streamlit Key 命名规范

每个 tab 模块所有 widget key 必须加前缀 `<tabname>_`：

- `tab_data.py` → `data_uploader`, `data_preview_limit`
- `tab_train.py` → `train_n_estimators`, `train_run_button`
- `tab_predict.py` → `predict_threshold`, `predict_run_button`

## §3 数据流 / Session State 契约

### 不变量

- ✅ Session state 的 key 名称、类型完全保持原样（参考 `web_app/utils/session_state.py:43-123`）
- ✅ 14/15/16 号页面继续工作，因为它们各自调用 `init_session_state()`
- ✅ 文件路径常量（`project_root`, `data_dir`, `model_dir`, `config_dir`, `output_dir`）保持原样

### Tab 间依赖与守卫

每个 tab 顶部应保留原页面已有的状态守卫：

```python
def render() -> None:
    if st.session_state.get('uploaded_data') is None:
        st.warning("请先在「数据准备」Tab 上传数据")
        return
    # ...
```

迁移时仅把"请先在 X 页面..."提示语改成"请先在 X Tab..."。

### 新增便利

`web_app/manual_training/__init__.py` 暴露 tab 列表，便于入口页一行 import：

```python
from web_app.manual_training import (
    tab_data, tab_config, tab_eda,
    tab_train, tab_model_mgmt, tab_predict,
)
```

## §4 迁移策略（机械式 6 步）

### 准备阶段

**Step 0 — 搭骨架**：

1. 创建 `web_app/manual_training/__init__.py`
2. 创建 `web_app/_legacy_pages/`
3. 创建入口页 `web_app/pages/4_📚_手动训练.py`，6 个空 tab + `st.info("待实现")`
4. 验证：`streamlit run web_app/app.py`，新页面在侧边栏出现，6 个空 tab 能切换

### 迁移阶段（按顺序逐 tab 执行）

| 步 | 目标 tab | 来源页面 |
|---|---|---|
| Step 1 | `tab_data.py` | `1_数据管理.py` + `8_数据源管理.py` |
| Step 2 | `tab_config.py` | `2_配置管理.py` |
| Step 3 | `tab_eda.py` | `3_数据分析.py` |
| Step 4 | `tab_train.py` | `4_模型训练.py` |
| Step 5 | `tab_model_mgmt.py` | `5_模型管理.py` + `9_模型对比.py` |
| Step 6 | `tab_predict.py` | `6_模型预测.py` + `7_结果分析.py` |

### 单 tab 迁移的"四步动作"

1. **复制**：原页面整体复制到新 `tab_xxx.py`
2. **改造**（机械操作）：
   - 删除 `st.set_page_config(...)`
   - 删除 `init_session_state()` 调用
   - 删除文件顶部 `sys.path.insert(...)`
   - 把所有顶层代码包进 `def render() -> None:`
   - `st.title(...)` 改为 `st.header(...)` 或删除
   - 所有 `st.<widget>(key=...)` 加前缀 `<tab>_`
   - 多源 tab：在 render() 内用二级 `st.tabs([...])` 包住两段，每段加独立 key 前缀
3. **备份**：`mv web_app/pages/X_xxx.py web_app/_legacy_pages/`
4. **验证**：`streamlit run web_app/app.py`，进入「📚 手动训练」→ 对应 tab，手动操作完整流程对照原页面行为

### 收尾

5. 写一行说明 `web_app/_legacy_pages/README.md`（备份用途）
6. 提交：建议每完成一个 tab 提一次 commit

## §5 风险与回滚

### 已识别风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | Streamlit widget key 冲突 | 强制规范：每个 widget 显式加 `key="<tab>_xxx"` |
| R2 | `st.rerun()` 跨 tab 副作用 | 迁移时审查所有 `st.rerun()` |
| R3 | 多源 tab 内部 widget 同名 | 二级 `st.tabs` 隔离 + key 前缀 `data_file_*`, `data_db_*` |
| R4 | 8 号页面 DB 依赖较重 | tab_data.py 延迟导入：DB 部分 import 放进二级 tab `with` 块内 |
| R5 | 文件名带 emoji 跨平台 | 已有同款命名（14/15/16），沿用 |
| R6 | 14/15/16 号页面读取 session_state | 不变，但需冒烟验证 |
| R7 | 导入路径错误 | 入口页统一处理 sys.path，tab 模块不操作 |

### 回滚

完全可逆：

```bash
# 单步回滚
mv web_app/_legacy_pages/4_模型训练.py web_app/pages/

# 全量回滚
mv web_app/_legacy_pages/*.py web_app/pages/
rm -rf web_app/manual_training/
rm web_app/pages/4_📚_手动训练.py
```

### YAGNI 边界（明确不做）

- ❌ 不抽取公共组件基类 / Mixin
- ❌ 不引入路由框架 / 状态机库
- ❌ 不动 `web_app/components/` 和 `web_app/utils/`
- ❌ 不动 `model_code/`、`core/`、`backend/`
- ❌ 不重命名 session_state key
- ❌ 不写新单元测试

## §6 测试与验收

### 静态校验（每个 tab 完成时）

```bash
# 1. 导入正确性
python -c "from web_app.manual_training import tab_data, tab_config, tab_eda, tab_train, tab_model_mgmt, tab_predict"

# 2. 无禁用调用残留
grep -rn "st.set_page_config\|init_session_state\|sys.path.insert" web_app/manual_training/
# 期望：完全无输出
```

### 手动冒烟清单（最终验收）

`streamlit run web_app/app.py` → 「📚 手动训练」：

| Tab | 冒烟动作 | 期望 |
|---|---|---|
| 1️⃣ 数据准备 | 上传 CSV → 预览 | `session_state.uploaded_data` 写入 |
| 1️⃣ 数据准备 | 「🗄️ 数据库源」加载已存连接 | 与 8 号页面一致 |
| 2️⃣ 字段配置 | 编辑 field_config → 保存 | xlsx 写入磁盘 |
| 3️⃣ 数据探索 | 绘制分布 / 相关性 | 图表正常 |
| 4️⃣ 模型训练 | 配置参数 → 训练 | `trained_models` 更新 |
| 5️⃣ 模型管理与对比 | 「📦 模型列表」 | 与 5 号页面一致 |
| 5️⃣ 模型管理与对比 | 「⚖️ 模型对比」 | 指标图正常 |
| 6️⃣ 预测与结果 | 「🔮 预测」 | `predict_results` 写入 |
| 6️⃣ 预测与结果 | 「📋 结果分析」调阈值 | 与 7 号页面一致 |

### 回归测试

对剩余入口冒烟一次：`10_自动迭代`、`11_数据库连接管理`、`12_大模型配置管理`、`13_AI_智能分析`、`14_Agent工作流`、`15_AutoML 训练`、`16_MLOps 监控`。重点验证 14/15/16 仍能正确读取 `session_state.uploaded_data` 与 `trained_models`。

### 验收标准（Done 的定义）

1. ✅ 6 个 tab 全部迁移完成，原 9 页面已移至 `web_app/_legacy_pages/`
2. ✅ 静态校验全部通过
3. ✅ 手动冒烟 9 条 + 回归 7 条全部通过
4. ✅ 入口页 `pages/4_📚_手动训练.py` ≤ 100 行
5. ✅ 每个 `tab_*.py` 各自只有一个 `def render()` 顶层函数
6. ✅ 所有改动已 commit
