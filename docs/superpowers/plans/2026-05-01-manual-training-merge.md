# 手动训练页面合并 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `web_app/pages/` 下的页面 1-9 合并为单个带 6 个 Tab 的入口页 `pages/4_📚_手动训练.py`，原 9 页面移至 `web_app/_legacy_pages/`。

**Architecture:** 创建 `web_app/manual_training/` 模块，每个 tab 一个文件（`tab_data.py` / `tab_config.py` / `tab_eda.py` / `tab_train.py` / `tab_model_mgmt.py` / `tab_predict.py`），各暴露 `render()` 函数。入口页统一 `set_page_config` + `init_session_state` + `st.tabs(...)`。每个 tab 内 widget key 加 `<tab>_` 前缀避免冲突；多源合并 tab 内部用二级 `st.tabs` 隔离子分区。

**Tech Stack:** Streamlit 1.x（多页面 + Tab）、Python 3.12、joblib、LightGBM/sklearn（仅 tab_train 用到）、pandas/numpy/plotly。

**参考 spec：** `docs/superpowers/specs/2026-05-01-manual-training-merge-design.md`

---

## 文件结构

会创建：
- `web_app/manual_training/__init__.py` — 模块入口，re-export 6 个 tab 子模块
- `web_app/manual_training/tab_data.py` — 数据准备（合并原 1+8）
- `web_app/manual_training/tab_config.py` — 字段配置（原 2）
- `web_app/manual_training/tab_eda.py` — 数据探索（原 3）
- `web_app/manual_training/tab_train.py` — 模型训练（原 4）
- `web_app/manual_training/tab_model_mgmt.py` — 模型管理与对比（原 5+9）
- `web_app/manual_training/tab_predict.py` — 预测与结果（原 6+7）
- `web_app/pages/4_📚_手动训练.py` — 唯一入口（替换原 4_模型训练.py 的位置）
- `web_app/_legacy_pages/README.md` — 备份目录说明

会移动（不修改）：
- `web_app/pages/1_数据管理.py` → `web_app/_legacy_pages/1_数据管理.py`
- `web_app/pages/2_配置管理.py` → `web_app/_legacy_pages/2_配置管理.py`
- `web_app/pages/3_数据分析.py` → `web_app/_legacy_pages/3_数据分析.py`
- `web_app/pages/4_模型训练.py` → `web_app/_legacy_pages/4_模型训练.py`
- `web_app/pages/5_模型管理.py` → `web_app/_legacy_pages/5_模型管理.py`
- `web_app/pages/6_模型预测.py` → `web_app/_legacy_pages/6_模型预测.py`
- `web_app/pages/7_结果分析.py` → `web_app/_legacy_pages/7_结果分析.py`
- `web_app/pages/8_数据源管理.py` → `web_app/_legacy_pages/8_数据源管理.py`
- `web_app/pages/9_模型对比.py` → `web_app/_legacy_pages/9_模型对比.py`

---

## 通用迁移规则（每个 tab 任务都遵守）

每次将一个老页面 `web_app/pages/X_xxx.py` 改造为 `web_app/manual_training/tab_xxx.py` 时，**机械应用**以下转换规则：

| 老页面里的模式 | 新 tab 中的对应处理 |
|---|---|
| `import sys` + `sys.path.insert(...)` | **删除**（入口页统一处理） |
| `from web_app.utils.session_state import init_session_state, ...` | 保留**仅**用到的函数（如 `add_training_log`），删除 `init_session_state` |
| `init_session_state()` 调用 | **删除** |
| `st.set_page_config(...)` 调用 | **删除** |
| `st.title("XXX")` | 改为 `st.header("XXX")` 或**删除**（入口页有总标题） |
| 顶层代码（非函数包裹） | 全部缩进进 `def render() -> None:` |
| `st.button("XXX")` 无 key | 改为 `st.button("XXX", key="<tab>_xxx_btn")` |
| `st.selectbox(..., key="abc")` | 改为 `st.selectbox(..., key="<tab>_abc")` |
| `st.switch_page("pages/Y_xxx.py")` | **删除整个跳转按钮 + warning 块**，改为 `st.warning("请先在「<目标 Tab>」Tab 上完成 ...")` 后 `return` |
| `st.stop()` | 改为 `return`（因为现在身处函数中） |

**key 前缀规则：**
- `tab_data.py` → `data_*`（文件上传子区用 `data_file_*`，DB 子区用 `data_db_*`）
- `tab_config.py` → `config_*`
- `tab_eda.py` → `eda_*`
- `tab_train.py` → `train_*`
- `tab_model_mgmt.py` → `mm_*`（列表子区用 `mm_list_*`，对比子区用 `mm_cmp_*`）
- `tab_predict.py` → `pred_*`（预测子区用 `pred_run_*`，结果子区用 `pred_res_*`）

**多源合并 tab 模板：**

```python
def render() -> None:
    sub_a, sub_b = st.tabs(["📁 子分区 A", "🗄️ 子分区 B"])
    with sub_a:
        _render_sub_a()
    with sub_b:
        _render_sub_b()
```

---

## Task 1: 搭骨架（创建模块目录 + 备份目录 + 空入口页）

**Files:**
- Create: `web_app/manual_training/__init__.py`
- Create: `web_app/_legacy_pages/README.md`
- Create: `web_app/pages/4_📚_手动训练.py`

- [ ] **Step 1.1: 创建空的 `manual_training/__init__.py`**

写入 `web_app/manual_training/__init__.py`：

```python
"""
手动训练流程模块。

本目录下每个 tab_*.py 文件对应一个 Tab，暴露统一的 render() 函数。
入口页：web_app/pages/4_📚_手动训练.py
"""
```

- [ ] **Step 1.2: 创建备份目录 README**

写入 `web_app/_legacy_pages/README.md`：

```markdown
# Legacy Pages 备份

此目录保留 2026-05-01 合并前的 9 个手动训练页面源码。

合并后入口：`web_app/pages/4_📚_手动训练.py`
合并后模块：`web_app/manual_training/`
设计文档：`docs/superpowers/specs/2026-05-01-manual-training-merge-design.md`

> 此目录在 `web_app/` 下但**不在** `pages/` 下，因此 Streamlit 不会将其纳入侧边栏。
```

- [ ] **Step 1.3: 创建最小可跑的入口页骨架（6 个空 tab）**

写入 `web_app/pages/4_📚_手动训练.py`：

```python
"""
手动训练流程入口页。

合并自原 pages/1_数据管理.py、2_配置管理.py、3_数据分析.py、
4_模型训练.py、5_模型管理.py、6_模型预测.py、7_结果分析.py、
8_数据源管理.py、9_模型对比.py。

设计文档：docs/superpowers/specs/2026-05-01-manual-training-merge-design.md
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 入口页统一处理 sys.path（tab 模块禁止再做 sys.path.insert）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web_app.utils.session_state import init_session_state

# 页面配置（整个页面只能调用一次）
st.set_page_config(
    page_title="手动训练流程",
    page_icon="📚",
    layout="wide",
)

# 初始化所有 session state（所有 tab 共享）
init_session_state()

st.title("📚 手动训练流程")
st.caption(
    "数据准备 → 字段配置 → 数据探索 → 模型训练 → 模型管理与对比 → 预测与结果"
)
st.markdown("---")

TAB_LABELS = [
    "1️⃣ 数据准备",
    "2️⃣ 字段配置",
    "3️⃣ 数据探索",
    "4️⃣ 模型训练",
    "5️⃣ 模型管理与对比",
    "6️⃣ 预测与结果",
]

tab_data_ui, tab_config_ui, tab_eda_ui, tab_train_ui, tab_model_ui, tab_predict_ui = st.tabs(TAB_LABELS)

with tab_data_ui:
    st.info("待实现：数据准备 Tab（Task 2）")
with tab_config_ui:
    st.info("待实现：字段配置 Tab（Task 3）")
with tab_eda_ui:
    st.info("待实现：数据探索 Tab（Task 4）")
with tab_train_ui:
    st.info("待实现：模型训练 Tab（Task 5）")
with tab_model_ui:
    st.info("待实现：模型管理与对比 Tab（Task 6）")
with tab_predict_ui:
    st.info("待实现：预测与结果 Tab（Task 7）")
```

- [ ] **Step 1.4: 验证骨架可加载**

Run（在项目根目录）：

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.utils.session_state import init_session_state; print('ok')"
```

Expected: 输出 `ok`，无 ImportError。

启动 Streamlit 验证（手动）：

```bash
streamlit run web_app/app.py --server.port 8501
```

Expected：浏览器侧边栏出现「4 📚 手动训练」入口；点击进入后显示 6 个 Tab，每个 Tab 显示「待实现」蓝色提示信息；其它原有页面（1-9 还在原位）也都能正常打开。手动确认后 Ctrl-C 关闭。

- [ ] **Step 1.5: Commit**

```bash
git add web_app/manual_training/__init__.py web_app/_legacy_pages/README.md "web_app/pages/4_📚_手动训练.py"
git commit -m "$(cat <<'EOF'
feat(web_app): 添加手动训练合并入口骨架

新增 web_app/manual_training/ 模块目录与 _legacy_pages/ 备份目录占位，
新建 pages/4_📚_手动训练.py 作为合并后的统一入口（6 个空 Tab）。
后续按 spec 逐 Tab 迁移原 1-9 号页面。

Refs: docs/superpowers/specs/2026-05-01-manual-training-merge-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: tab_data — 合并 1_数据管理 + 8_数据源管理

**Files:**
- Create: `web_app/manual_training/tab_data.py`
- Modify: `web_app/pages/4_📚_手动训练.py`（替换 tab_data_ui 的占位）
- Move: `web_app/pages/1_数据管理.py` → `web_app/_legacy_pages/1_数据管理.py`
- Move: `web_app/pages/8_数据源管理.py` → `web_app/_legacy_pages/8_数据源管理.py`

- [ ] **Step 2.1: 创建 tab_data.py 骨架**

写入 `web_app/manual_training/tab_data.py`（先写最小骨架，然后填充）：

```python
"""
tab_data.py — 数据准备 Tab

合并自：
  - web_app/pages/1_数据管理.py（文件上传 + 预览 + 统计 + 导出）
  - web_app/pages/8_数据源管理.py（数据库连接 + SQL 查询）
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
    import pandas as pd
    from web_app.components.data_upload import (
        render_data_upload_section,
        render_data_preview,
        render_data_statistics,
        render_field_names,
    )
    from web_app.utils.file_handler import (
        calculate_data_statistics,
        create_download_button,
        create_excel_download_button,
    )

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
        st.header("💾 导出数据")
        col1, col2 = st.columns(2)
        with col1:
            create_download_button(df, "exported_data.csv", "📥 下载 CSV 格式")
        with col2:
            create_excel_download_button(df, "exported_data.xlsx", "📥 下载 Excel 格式")
        st.info("💡 上传的数据将自动保存到会话状态，可在其它 Tab 使用")


def _render_db_source() -> None:
    """子分区：数据库源（来自原 8_数据源管理.py）。

    重要：DB 相关重依赖（model_code/database/*）只在本函数内 import，
    避免文件上传场景白白加载。
    """
    st.header("🗄️ 数据库源")
    # 把原 web_app/pages/8_数据源管理.py 的 body 内容（去掉
    # set_page_config / init_session_state / sys.path / st.title）整段粘进来。
    # 所有 widget key 加前缀 data_db_*。
    # st.switch_page 调用全部删除，改为 st.warning + return。
    # 详见 Step 2.3。
    raise NotImplementedError("placeholder—Step 2.3 fills in")
```

- [ ] **Step 2.2: 填充 _render_file_upload（如已写完跳过）**

Step 2.1 已写完。

- [ ] **Step 2.3: 填充 _render_db_source（拷贝 8_数据源管理.py 的内容并改造）**

打开 `web_app/pages/8_数据源管理.py`，把它的**主体逻辑**（从 `st.title` 之后到文件末尾的所有 streamlit 调用）整段拷贝进 `_render_db_source()` 函数内。在拷贝时按通用迁移规则做改造：

  1. 不要拷贝 `import sys`/`sys.path.insert`/`init_session_state()`/`st.set_page_config(...)`/顶层 `st.title(...)` 这些行
  2. `st.title("🗄️ 数据源管理")` 已在函数内有 `st.header("🗄️ 数据库源")` 替代，删除
  3. 所有 `st.button/selectbox/text_input/checkbox/...` 加 `key="data_db_xxx"`
  4. 删除 `raise NotImplementedError(...)`
  5. 如有 `st.switch_page("pages/Y_xxx.py")` 调用 → 改成 `st.warning("请先在「<目标>」Tab ...")` 然后 `return`

注意：所有 `from model_code.database.*` 等重依赖 import 留在本函数内（**不**搬到模块顶部），保持延迟加载。

- [ ] **Step 2.4: 把入口页中 tab_data_ui 的占位改成调用 tab_data.render()**

修改 `web_app/pages/4_📚_手动训练.py`：

把这段：
```python
with tab_data_ui:
    st.info("待实现：数据准备 Tab（Task 2）")
```

改为：
```python
with tab_data_ui:
    from web_app.manual_training import tab_data
    tab_data.render()
```

import 放在 `with` 块内是为了在切换 tab 之前不去触发 DB 等重依赖。

- [ ] **Step 2.5: 静态校验**

Run：

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_data; assert callable(tab_data.render); print('ok')"
```

Expected: 输出 `ok`，无 ImportError。

Run：

```bash
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert" web_app/manual_training/tab_data.py
```

Expected: 无输出（无任何禁用调用残留）。

- [ ] **Step 2.6: 移动原 1_数据管理.py 与 8_数据源管理.py 到备份目录**

Run：

```bash
mv web_app/pages/1_数据管理.py web_app/_legacy_pages/1_数据管理.py
mv web_app/pages/8_数据源管理.py web_app/_legacy_pages/8_数据源管理.py
```

Expected: 两个文件已移动；`ls web_app/pages/` 不再包含它们；`ls web_app/_legacy_pages/` 包含它们。

- [ ] **Step 2.7: 手动冒烟**

Run：

```bash
streamlit run web_app/app.py --server.port 8501
```

手动测试：
1. 侧边栏不再有「1 数据管理」「8 数据源管理」
2. 进入「📚 手动训练」→「1️⃣ 数据准备」
3. 切换「📁 文件上传」子 tab → 上传一个小 CSV → 预览 / 字段 / 统计 / 导出 四个二级 tab 都能切换、内容正常
4. 切换「🗄️ 数据库源」子 tab → 与原 8 号页面行为一致（不需要真的连数据库，只看 UI 是否渲染、加 / 测试连接的按钮是否点击响应）
5. 在其它 Tab（如「2️⃣ 字段配置」点占位 info、「14_🤖_Agent工作流」等）确认 `st.session_state.uploaded_data` 仍是刚上传的数据

Expected：行为与原 1 号、8 号页面一致；没有 widget key 报错（红色 stack）。

- [ ] **Step 2.8: Commit**

```bash
git add web_app/manual_training/tab_data.py "web_app/pages/4_📚_手动训练.py"
git mv -f web_app/_legacy_pages/1_数据管理.py 2>/dev/null || true  # 已通过 mv 移动，git 自动检测
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 合并数据管理 + 数据源管理为 tab_data

新增 web_app/manual_training/tab_data.py，二级 tab 隔离文件上传与
数据库源；原 pages/1_数据管理.py 与 pages/8_数据源管理.py 移至
web_app/_legacy_pages/ 备份。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: tab_config — 迁移 2_配置管理

**Files:**
- Create: `web_app/manual_training/tab_config.py`
- Modify: `web_app/pages/4_📚_手动训练.py`
- Move: `web_app/pages/2_配置管理.py` → `web_app/_legacy_pages/2_配置管理.py`

- [ ] **Step 3.1: 创建 tab_config.py 并粘贴改造后的内容**

写入 `web_app/manual_training/tab_config.py`：

```python
"""
tab_config.py — 字段配置 Tab

合并自：web_app/pages/2_配置管理.py
"""

import streamlit as st


def render() -> None:
    """渲染字段配置 Tab。"""
    import pandas as pd
    import numpy as np
    from web_app.components.config_editor import (
        render_config_editor,
        render_config_actions,
        render_field_type_distribution,
        render_config_summary,
        auto_infer_config,
    )
    from web_app.utils.file_handler import (
        read_file_from_path,
        save_config_to_excel,
    )

    st.header("⚙️ 字段配置")

    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        st.info("💡 字段配置需要基于已上传的数据生成")
        return

    if st.session_state.get("field_config") is None:
        config_path = st.session_state.config_dir / "field_config.xlsx"
        if config_path.exists():
            st.session_state.field_config = read_file_from_path(str(config_path))
            st.session_state.config_file_path = str(config_path)
            st.success(f"✅ 已加载现有配置：{config_path}")
        else:
            st.session_state.field_config = auto_infer_config(
                st.session_state.uploaded_data
            )
            st.info("✨ 已根据数据自动生成配置，您可以进行调整")

    config_df = st.session_state.field_config

    if st.session_state.get("config_file_path"):
        st.caption(f"📁 当前配置文件：{st.session_state.config_file_path}")

    st.subheader("📝 配置编辑")
    edited_config = render_config_editor(config_df)

    if not edited_config.equals(config_df):
        st.session_state.field_config = edited_config
        st.session_state.config_modified = True
        st.info("✏️ 配置已修改，记得保存！")

    st.markdown("---")
    render_config_actions(edited_config, st.session_state.get("config_file_path"))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        render_field_type_distribution(edited_config)
    with col2:
        render_config_summary(edited_config)

    with st.expander("📖 配置说明"):
        st.markdown(
            """
        ### 字段配置说明

        #### field_type (字段类型)
        - **num**: 数值型字段，用于模型特征
        - **str**: 分类型字段，需要编码处理
        - **key**: 标识字段（如用户 ID），不参与训练
        - **label**: 目标变量（标签），用于监督学习

        #### outlier_value (异常值处理)
        - **3std**: 使用 3 倍标准差裁剪
        - **5std**: 使用 5 倍标准差裁剪
        - **(min,max)**: 使用固定范围裁剪，如 (0,100)
        - **空**: 不处理异常值

        #### missing_value (缺失值填充)
        - **mean**: 使用均值填充
        - **median**: 使用中位数填充
        - **mode**: 使用众数填充
        - **0/-1**: 使用固定值填充
        - **空**: 不填充

        #### ecode_type (编码方式)
        - **one-hot**: 独热编码（适合无序分类）
        - **ordinal**: 顺序编码（适合有序分类）
        - **空**: 不编码（数值型字段）

        #### standardize_type (标准化)
        - **z-score**: 标准差标准化（均值 0，方差 1）
        - **min-max**: 最小 - 最大标准化（缩放到 0-1）
        - **robust**: 稳健标准化（使用四分位数）
        - **空**: 不标准化
        """
        )
```

> 注意：原页面里调用了 `render_config_editor` / `render_config_actions` 等组件，这些组件内部可能已经使用了 widget key（看 `web_app/components/config_editor.py`）。本 tab 不重命名它们的 key（不动 components/），仅在本文件内新增的 widget 加前缀 `config_`。本 tab 当前没有直接调用 `st.button/selectbox` 等带 key 的 widget，故不需额外加 key。

- [ ] **Step 3.2: 替换入口页占位**

修改 `web_app/pages/4_📚_手动训练.py`，把：
```python
with tab_config_ui:
    st.info("待实现：字段配置 Tab（Task 3）")
```

改为：
```python
with tab_config_ui:
    from web_app.manual_training import tab_config
    tab_config.render()
```

- [ ] **Step 3.3: 静态校验**

Run：

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_config; assert callable(tab_config.render); print('ok')"
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert" web_app/manual_training/tab_config.py
```

Expected: 第一行输出 `ok`；第二行无输出。

- [ ] **Step 3.4: 移动原 2_配置管理.py 到备份**

```bash
mv web_app/pages/2_配置管理.py web_app/_legacy_pages/2_配置管理.py
```

- [ ] **Step 3.5: 手动冒烟**

```bash
streamlit run web_app/app.py --server.port 8501
```

测试：
1. 进入「📚 手动训练」
2. 先到「1️⃣ 数据准备」上传 CSV
3. 切到「2️⃣ 字段配置」
4. 验证：自动推断生成配置 → 编辑某字段类型 → 保存配置 → field_type 分布饼图 + 摘要正常
5. 不上传数据直接进入「2️⃣ 字段配置」应显示 `请先在「1️⃣ 数据准备」Tab 上传数据`

Expected: 与原 2 号页面行为一致。

- [ ] **Step 3.6: Commit**

```bash
git add web_app/manual_training/tab_config.py "web_app/pages/4_📚_手动训练.py"
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 迁移配置管理为 tab_config

新增 web_app/manual_training/tab_config.py，原 pages/2_配置管理.py
移至 web_app/_legacy_pages/。跨页面跳转改为同页面 Tab 内的提示。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: tab_eda — 迁移 3_数据分析

**Files:**
- Create: `web_app/manual_training/tab_eda.py`
- Modify: `web_app/pages/4_📚_手动训练.py`
- Move: `web_app/pages/3_数据分析.py` → `web_app/_legacy_pages/3_数据分析.py`

- [ ] **Step 4.1: 创建 tab_eda.py**

写入 `web_app/manual_training/tab_eda.py`：

```python
"""
tab_eda.py — 数据探索 Tab

合并自：web_app/pages/3_数据分析.py
"""

import streamlit as st


def render() -> None:
    """渲染数据探索 Tab。"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from web_app.components.charts import (
        render_distribution_chart,
        render_correlation_heatmap,
        render_missing_value_heatmap,
        render_outlier_boxplot,
    )

    st.header("📈 数据探索")

    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        return

    df = st.session_state.uploaded_data

    sub_dist, sub_corr, sub_outlier, sub_report = st.tabs(
        ["📊 分布分析", "🔥 相关性分析", "⚠️ 异常值检测", "📋 生成报告"]
    )

    with sub_dist:
        _render_distribution(df, render_distribution_chart)

    with sub_corr:
        _render_correlation(df, render_correlation_heatmap, render_missing_value_heatmap)

    with sub_outlier:
        _render_outlier(df, render_outlier_boxplot)

    with sub_report:
        _render_report(df)


def _render_distribution(df, render_distribution_chart) -> None:
    import numpy as np

    st.subheader("📊 字段分布分析")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**数值字段分布**")
        if numeric_cols:
            selected_num_col = st.selectbox(
                "选择数值字段", numeric_cols, key="eda_dist_num"
            )
            if selected_num_col:
                try:
                    fig = render_distribution_chart(df, selected_num_col)
                    st.plotly_chart(fig, use_container_width=True, key=f"eda_plot_num_{selected_num_col}")
                    col_series = df[selected_num_col].dropna()
                    if len(col_series) > 0:
                        st.caption(
                            f"均值：{col_series.mean():.4f} | 中位数：{col_series.median():.4f} | "
                            f"标准差：{col_series.std():.4f} | 最小值：{col_series.min():.4f} | "
                            f"最大值：{col_series.max():.4f}"
                        )
                except Exception as e:
                    st.warning(f"⚠️ 无法绘制 {selected_num_col} 的分布图：{str(e)}")
        else:
            st.info("无数值字段")

    with col2:
        st.markdown("**分类字段分布**")
        if category_cols:
            selected_cat_col = st.selectbox(
                "选择分类字段", category_cols, key="eda_dist_cat"
            )
            if selected_cat_col:
                fig = render_distribution_chart(df, selected_cat_col)
                st.plotly_chart(fig, use_container_width=True, key=f"eda_plot_cat_{selected_cat_col}")
                value_counts = df[selected_cat_col].value_counts().head(10)
                st.caption(f"Top 类别：{value_counts.to_dict()}")
        else:
            st.info("无分类字段")

    st.markdown("---")
    st.markdown("**📊 所有数值字段分布**")
    if numeric_cols:
        show_all = st.checkbox(
            "显示所有数值字段分布（可能较慢）", key="eda_show_all_dist"
        )
        if show_all:
            for col in numeric_cols[:10]:
                fig = render_distribution_chart(df, col)
                st.plotly_chart(fig, use_container_width=True, key=f"eda_dist_all_{col}")


def _render_correlation(df, render_correlation_heatmap, render_missing_value_heatmap) -> None:
    import numpy as np
    import pandas as pd

    st.subheader("🔥 相关性分析")

    st.markdown("**特征相关性热力图**")
    fig = render_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="eda_corr_heatmap")

        st.markdown("**📊 高相关性特征对**")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.7:
                    high_corr_pairs.append({
                        "特征 1": corr_matrix.columns[i],
                        "特征 2": corr_matrix.columns[j],
                        "相关系数": corr_matrix.iloc[i, j],
                    })
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("相关系数", ascending=False)
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            st.info("💡 高相关性特征可能存在多重共线性，建议考虑移除其中一个")
        else:
            st.success("✅ 未发现高相关性特征对（相关系数 > 0.7）")
    else:
        st.warning("⚠️ 数值字段不足，无法计算相关性")

    st.markdown("---")
    st.markdown("**🔥 缺失值分布热力图**")
    fig = render_missing_value_heatmap(df)
    st.plotly_chart(fig, use_container_width=True, key="eda_missing_heatmap")


def _render_outlier(df, render_outlier_boxplot) -> None:
    import numpy as np
    import pandas as pd

    st.subheader("⚠️ 异常值检测")
    st.markdown("**箱线图检测异常值**")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ 无数值字段")
        return

    selected_cols = st.multiselect(
        "选择要分析的字段", numeric_cols, default=numeric_cols[:5], key="eda_outlier_cols"
    )
    if not selected_cols:
        return

    fig = render_outlier_boxplot(df, selected_cols)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"eda_boxplot_{'_'.join(selected_cols[:3])}",
    )

    st.markdown("**📊 异常值统计**")
    outlier_stats = []
    for col in selected_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_stats.append({
            "字段": col,
            "异常值数量": len(outliers),
            "异常值比例 (%)": round(len(outliers) / len(df) * 100, 2),
            "下界": round(lower, 4),
            "上界": round(upper, 4),
        })
    st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True, hide_index=True)


def _render_report(df) -> None:
    """生成 HTML 数据分析报告。"""
    import numpy as np
    from datetime import datetime

    st.subheader("📋 生成分析报告")
    st.markdown(
        """
    ### 分析报告内容

    - 数据概览（行数、列数、内存占用）
    - 字段类型分布
    - 缺失值分析
    - 唯一值分析
    - 数值字段统计
    - 相关性分析
    - 异常值检测
    """
    )

    st.markdown("**📝 报告选项**")
    col1, col2 = st.columns(2)
    with col1:
        include_stats = st.checkbox("包含统计信息", value=True, key="eda_rep_stats")
        include_missing = st.checkbox("包含缺失值分析", value=True, key="eda_rep_missing")
        include_correlation = st.checkbox("包含相关性分析", value=True, key="eda_rep_corr")
    with col2:
        include_outliers = st.checkbox("包含异常值检测", value=True, key="eda_rep_outliers")
        include_charts = st.checkbox("包含图表", value=True, key="eda_rep_charts")

    st.markdown("---")
    if not st.button("📄 生成 HTML 报告", type="primary", key="eda_rep_generate_btn"):
        return

    report_html = _build_report_html(df, include_stats, include_missing)
    st.download_button(
        label="📥 下载 HTML 报告",
        data=report_html,
        file_name=f"数据分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        key="eda_rep_download_btn",
    )
    st.success("✅ 报告生成成功！")


def _build_report_html(df, include_stats: bool, include_missing: bool) -> str:
    """构造 HTML 报告字符串。"""
    import numpy as np
    from datetime import datetime

    head = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>机器学习数据分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #ecf0f1; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>📊 机器学习数据分析报告</h1>
        <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>1. 数据概览</h2>
        <div class="metric"><div class="metric-value">{len(df):,}</div><div class="metric-label">总行数</div></div>
        <div class="metric"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">总列数</div></div>
        <div class="metric"><div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div><div class="metric-label">内存占用</div></div>

        <h2>2. 字段列表</h2>
        <table>
            <tr><th>序号</th><th>字段名</th><th>数据类型</th><th>非空值</th><th>唯一值</th></tr>
    """
    for i, col in enumerate(df.columns):
        head += (
            f"<tr><td>{i + 1}</td><td>{col}</td><td>{df[col].dtype}</td>"
            f"<td>{int(df[col].notna().sum()):,}</td>"
            f"<td>{int(df[col].nunique()):,}</td></tr>"
        )
    head += "</table>"

    if include_stats:
        head += """
        <h2>3. 数值字段统计</h2>
        <table><tr><th>字段</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            head += (
                f"<tr><td>{col}</td><td>{df[col].mean():.4f}</td><td>{df[col].median():.4f}</td>"
                f"<td>{df[col].std():.4f}</td><td>{df[col].min():.4f}</td>"
                f"<td>{df[col].max():.4f}</td></tr>"
            )
        head += "</table>"

    if include_missing:
        head += """
        <h2>4. 缺失值分析</h2>
        <table><tr><th>字段</th><th>缺失数量</th><th>缺失率 (%)</th></tr>
        """
        missing_summary = df.isna().sum()
        for col in df.columns:
            missing_count = int(missing_summary[col])
            missing_rate = missing_count / len(df) * 100
            if missing_count > 0:
                head += (
                    f"<tr><td>{col}</td><td>{missing_count:,}</td>"
                    f"<td>{missing_rate:.2f}%</td></tr>"
                )
        head += "</table>"

    head += "</body></html>"
    return head
```

- [ ] **Step 4.2: 替换入口页占位**

修改 `web_app/pages/4_📚_手动训练.py`，把 `with tab_eda_ui:` 块改成：

```python
with tab_eda_ui:
    from web_app.manual_training import tab_eda
    tab_eda.render()
```

- [ ] **Step 4.3: 静态校验**

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_eda; assert callable(tab_eda.render); print('ok')"
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert" web_app/manual_training/tab_eda.py
```

Expected: 第一行 `ok`，第二行无输出。

- [ ] **Step 4.4: 移动原 3_数据分析.py**

```bash
mv web_app/pages/3_数据分析.py web_app/_legacy_pages/3_数据分析.py
```

- [ ] **Step 4.5: 手动冒烟**

```bash
streamlit run web_app/app.py --server.port 8501
```

测试：进入「📚 手动训练」→ 上传 CSV →「3️⃣ 数据探索」→ 4 个二级 tab 切换：分布图、相关性热力图、异常值箱线图、HTML 报告下载。

Expected: 与原 3 号页面行为一致。

- [ ] **Step 4.6: Commit**

```bash
git add web_app/manual_training/tab_eda.py "web_app/pages/4_📚_手动训练.py"
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 迁移数据分析为 tab_eda

新增 web_app/manual_training/tab_eda.py，分布 / 相关性 / 异常值 / HTML
报告四子分区；原 pages/3_数据分析.py 移至 web_app/_legacy_pages/。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: tab_train — 迁移 4_模型训练

**Files:**
- Create: `web_app/manual_training/tab_train.py`
- Modify: `web_app/pages/4_📚_手动训练.py`
- Move: `web_app/pages/4_模型训练.py` → `web_app/_legacy_pages/4_模型训练.py`

> 注意：原 4_模型训练.py 共 639 行，是体量最大的页面。本任务把它**整体**迁移到 `tab_train.py`，仅做"通用迁移规则"列出的机械改造。**不**做任何业务重构、参数调优、行为变更。

- [ ] **Step 5.1: 创建 tab_train.py 头部**

写入 `web_app/manual_training/tab_train.py` 的 docstring 和顶部 import：

```python
"""
tab_train.py — 模型训练 Tab

迁移自：web_app/pages/4_模型训练.py（639 行）
本次仅做机械迁移：
  - 删除 set_page_config / init_session_state / sys.path / st.title
  - 包进 def render()
  - widget key 加 train_ 前缀
  - 跨页面跳转改为同页面 Tab 提示
不做行为/算法/参数变更。
"""

import streamlit as st
```

- [ ] **Step 5.2: 写 render() 函数主体**

把 `web_app/_legacy_pages/4_模型训练.py`（仍在原位 `web_app/pages/4_模型训练.py`，等下再移走）从行 50 (`# ======== 检查前置条件 ========`) 到文件末尾的代码**整段缩进进 `def render() -> None:`**，并按下表替换：

| 原代码 | 新代码 |
|---|---|
| `if st.session_state.get('uploaded_data') is None:` ... `st.switch_page("pages/1_数据管理.py")` ... `st.stop()` | `if st.session_state.get('uploaded_data') is None:`<br>`    st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")`<br>`    return` |
| `if st.session_state.get('field_config') is None:` ... `st.switch_page("pages/2_配置管理.py")` ... `st.stop()` | `if st.session_state.get('field_config') is None:`<br>`    st.warning("⚠️ 请先在「2️⃣ 字段配置」Tab 配置字段")`<br>`    return` |
| `start_training = st.button("🚀 开始训练", type="primary", use_container_width=True, disabled=...)` | 末尾加 `key="train_start_btn"` |
| `if st.button("🗑️ 清除训练结果", use_container_width=True, disabled=...)` | 末尾加 `key="train_clear_btn"` |
| `st.download_button(label="📥 下载预测结果 (CSV)", ...)` | 末尾加 `key="train_dl_predictions_btn"` |
| `st.download_button(label="📥 下载特征重要性 (CSV)", ...)` | 末尾加 `key="train_dl_importance_btn"` |
| `st.plotly_chart(fig, use_container_width=True)` （ROC/PR/CM/特征重要性 4 个） | 加 `key="train_plot_roc"` / `train_plot_pr` / `train_plot_cm` / `train_plot_fi` |
| `st.tabs([...])`（结果区域那 4 个）变量名 `tab1, tab2, tab3, tab4` | 改为 `train_res_t1, train_res_t2, train_res_t3, train_res_t4`（避免和外层 tab 变量名混淆） |
| 文件顶部 `st.title("🚀 模型训练")` 行 | 改为函数体首行 `st.header("🚀 模型训练")` |

> **slider/number_input/checkbox 的 widget**：原页面这些控件无 key，Streamlit 会按位置自动生成 key。本 tab 内**新增**显式 key，全部加 `train_` 前缀，例如：
>   - `n_estimators = st.slider("n_estimators (树的数量)", ..., key="train_n_estimators")`
>   - `learning_rate = st.slider(..., key="train_learning_rate")`
>   - 所有 9 个 LightGBM 参数 slider + 3 个训练配置（test_size/cv_folds/random_state） + use_grid_search 复选框 + n_iter slider 都加 `train_` 前缀

- [ ] **Step 5.3: 替换入口页占位**

修改 `web_app/pages/4_📚_手动训练.py`：

```python
with tab_train_ui:
    from web_app.manual_training import tab_train
    tab_train.render()
```

- [ ] **Step 5.4: 静态校验**

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_train; assert callable(tab_train.render); print('ok')"
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert\|st.stop()\|st.switch_page" web_app/manual_training/tab_train.py
```

Expected: 第一行 `ok`，第二行无输出。

- [ ] **Step 5.5: 移动原 4_模型训练.py（注意目标文件名要错开）**

注意：在 Task 1 中已经创建了 `web_app/pages/4_📚_手动训练.py`（带 emoji），原 `web_app/pages/4_模型训练.py`（不带 emoji）是另一个文件。两者并存不冲突。这一步移走原文件：

```bash
mv web_app/pages/4_模型训练.py web_app/_legacy_pages/4_模型训练.py
```

Expected: 移动成功。`ls web_app/pages/` 中只有 `4_📚_手动训练.py`（带 emoji），不再有 `4_模型训练.py`。

- [ ] **Step 5.6: 手动冒烟**

```bash
streamlit run web_app/app.py --server.port 8501
```

测试：上传 CSV →「2️⃣ 字段配置」生成配置 →「4️⃣ 模型训练」→
1. 9 个 LightGBM 参数 slider 拖动正常
2. 测试集比例 / CV 折数 / 随机种子可调
3. 启用网格搜索 + 5 次迭代 → 点击「🚀 开始训练」
4. 训练完成后看到 4 个评估图（ROC/PR/CM/特征重要性）
5. 看到分类报告 dataframe
6. 两个下载按钮可点击
7. 点击「🗑️ 清除训练结果」能清空

Expected: 与原 4 号页面行为一致；训练完成后 `st.session_state.trained_models` 写入新模型；`saved/` 或 `model_code/saved/` 目录有 `model_*.joblib` 与 `train_summary_*.joblib`。

- [ ] **Step 5.7: Commit**

```bash
git add web_app/manual_training/tab_train.py "web_app/pages/4_📚_手动训练.py"
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 迁移模型训练为 tab_train

新增 web_app/manual_training/tab_train.py（机械迁移自 639 行老页面）；
所有 widget 显式 key 加 train_ 前缀；跨页面跳转改为同页面 Tab 提示。
原 pages/4_模型训练.py 移至 web_app/_legacy_pages/。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: tab_model_mgmt — 合并 5_模型管理 + 9_模型对比

**Files:**
- Create: `web_app/manual_training/tab_model_mgmt.py`
- Modify: `web_app/pages/4_📚_手动训练.py`
- Move: `web_app/pages/5_模型管理.py` → `web_app/_legacy_pages/5_模型管理.py`
- Move: `web_app/pages/9_模型对比.py` → `web_app/_legacy_pages/9_模型对比.py`

- [ ] **Step 6.1: 创建 tab_model_mgmt.py 骨架**

写入 `web_app/manual_training/tab_model_mgmt.py`：

```python
"""
tab_model_mgmt.py — 模型管理与对比 Tab

合并自：
  - web_app/pages/5_模型管理.py（已训练模型列表 / 详情 / 删除 / 导出）
  - web_app/pages/9_模型对比.py（多模型并排评估）
"""

import streamlit as st


def render() -> None:
    """渲染模型管理与对比 Tab。"""
    sub_list, sub_cmp = st.tabs(["📦 模型列表", "⚖️ 模型对比"])
    with sub_list:
        _render_model_list()
    with sub_cmp:
        _render_model_compare()


def _render_model_list() -> None:
    """子分区：模型列表（来自原 5_模型管理.py）。"""
    # 把原 web_app/pages/5_模型管理.py 主体（去掉 set_page_config /
    # init_session_state / sys.path / 顶层 st.title）整段粘进来。
    # 所有 widget key 加前缀 mm_list_。
    # 详见 Step 6.2。
    raise NotImplementedError("Step 6.2 fills in")


def _render_model_compare() -> None:
    """子分区：模型对比（来自原 9_模型对比.py）。"""
    # 把原 web_app/pages/9_模型对比.py 主体（去掉 set_page_config /
    # init_session_state / sys.path / 顶层 st.title）整段粘进来。
    # 所有 widget key 加前缀 mm_cmp_。
    # 详见 Step 6.3。
    raise NotImplementedError("Step 6.3 fills in")
```

- [ ] **Step 6.2: 填充 _render_model_list**

打开 `web_app/pages/5_模型管理.py`（仍在原位），把 `st.title("📦 模型管理")` 之后的全部代码整段缩进进 `_render_model_list()`，按通用规则改造：

  1. 不要拷贝 set_page_config / init_session_state / sys.path / 顶层 st.title
  2. 函数首行替换为 `st.subheader("📦 已训练模型")`
  3. 所有 widget 加 `key="mm_list_xxx"` 前缀（每个 selectbox / button / download_button）
  4. 任何 `st.switch_page` 删除整块改为 `st.warning("请先在 ... Tab")` + `return`

- [ ] **Step 6.3: 填充 _render_model_compare**

打开 `web_app/pages/9_模型对比.py`，把 `st.title("⚖️ 模型对比")` 之后的全部代码整段缩进进 `_render_model_compare()`，按通用规则改造：

  1. 不要拷贝 set_page_config / init_session_state / sys.path / 顶层 st.title
  2. 函数首行替换为 `st.subheader("⚖️ 模型对比")`
  3. 所有 widget 加 `key="mm_cmp_xxx"` 前缀
  4. 任何 `st.plotly_chart(fig, ...)` 加 `key="mm_cmp_chart_<语义>"`
  5. `from sklearn.model_selection import train_test_split` 等重 import 留在函数体内

- [ ] **Step 6.4: 替换入口页占位**

修改 `web_app/pages/4_📚_手动训练.py`：

```python
with tab_model_ui:
    from web_app.manual_training import tab_model_mgmt
    tab_model_mgmt.render()
```

- [ ] **Step 6.5: 静态校验**

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_model_mgmt; assert callable(tab_model_mgmt.render); print('ok')"
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert\|NotImplementedError" web_app/manual_training/tab_model_mgmt.py
```

Expected: 第一行 `ok`，第二行无输出（NotImplementedError 已在 6.2/6.3 删除）。

- [ ] **Step 6.6: 移动原 5 与 9 号页面**

```bash
mv web_app/pages/5_模型管理.py web_app/_legacy_pages/5_模型管理.py
mv web_app/pages/9_模型对比.py web_app/_legacy_pages/9_模型对比.py
```

- [ ] **Step 6.7: 手动冒烟**

```bash
streamlit run web_app/app.py --server.port 8501
```

测试：在「4️⃣ 模型训练」训练 2 个模型 →「5️⃣ 模型管理与对比」→
1. 「📦 模型列表」能看到刚训练的模型，能查看详情、能删除
2. 「⚖️ 模型对比」能选 2 个模型并排对比 AUC/F1 等指标
3. 切到 14_Agent 工作流确认 `st.session_state.trained_models` 仍可读

Expected: 与原 5 号、9 号页面行为一致。

- [ ] **Step 6.8: Commit**

```bash
git add web_app/manual_training/tab_model_mgmt.py "web_app/pages/4_📚_手动训练.py"
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 合并模型管理 + 模型对比为 tab_model_mgmt

新增 web_app/manual_training/tab_model_mgmt.py，二级 tab 隔离列表与对比；
原 pages/5_模型管理.py 与 pages/9_模型对比.py 移至 web_app/_legacy_pages/。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: tab_predict — 合并 6_模型预测 + 7_结果分析

**Files:**
- Create: `web_app/manual_training/tab_predict.py`
- Modify: `web_app/pages/4_📚_手动训练.py`
- Move: `web_app/pages/6_模型预测.py` → `web_app/_legacy_pages/6_模型预测.py`
- Move: `web_app/pages/7_结果分析.py` → `web_app/_legacy_pages/7_结果分析.py`

- [ ] **Step 7.1: 创建 tab_predict.py 骨架**

写入 `web_app/manual_training/tab_predict.py`：

```python
"""
tab_predict.py — 预测与结果 Tab

合并自：
  - web_app/pages/6_模型预测.py（选模型 / 上传数据 / 跑预测 / 导出）
  - web_app/pages/7_结果分析.py（预测结果分布 / 阈值分析）
"""

import streamlit as st


def render() -> None:
    """渲染预测与结果 Tab。"""
    sub_run, sub_res = st.tabs(["🔮 预测", "📋 结果分析"])
    with sub_run:
        _render_run_predict()
    with sub_res:
        _render_result_analysis()


def _render_run_predict() -> None:
    """子分区：预测控制（来自原 6_模型预测.py）。"""
    raise NotImplementedError("Step 7.2 fills in")


def _render_result_analysis() -> None:
    """子分区：结果分析（来自原 7_结果分析.py）。"""
    raise NotImplementedError("Step 7.3 fills in")
```

- [ ] **Step 7.2: 填充 _render_run_predict**

打开 `web_app/pages/6_模型预测.py`，把 `st.title("🔮 模型预测")` 之后的全部代码整段缩进进 `_render_run_predict()`，按通用规则改造：

  1. 不要拷贝 set_page_config / init_session_state / sys.path / 顶层 st.title
  2. 函数首行 `st.subheader("🔮 模型预测")`
  3. 所有 widget 加 `key="pred_run_xxx"` 前缀
  4. 任何 `st.switch_page("pages/X_xxx.py")` → `st.warning("请先在「<目标 Tab>」Tab ...") + return`

- [ ] **Step 7.3: 填充 _render_result_analysis**

打开 `web_app/pages/7_结果分析.py`，把 `st.title("📋 结果分析")` 之后的全部代码整段缩进进 `_render_result_analysis()`，按通用规则改造：

  1. 不要拷贝 set_page_config / init_session_state / sys.path / 顶层 st.title
  2. 函数首行 `st.subheader("📋 结果分析")`
  3. 注意原 7 号页面顶部 import 是 `from utils.session_state import ...`（**没有** `web_app.` 前缀），是个 bug。本次迁移**不**直接 import session_state（render 内不需要），如果 import `from web_app.components.charts import ...` 等，统一写完整路径
  4. 所有 widget 加 `key="pred_res_xxx"` 前缀
  5. 任何 `st.plotly_chart(fig, ...)` 加 `key="pred_res_chart_<语义>"`

- [ ] **Step 7.4: 替换入口页占位**

修改 `web_app/pages/4_📚_手动训练.py`：

```python
with tab_predict_ui:
    from web_app.manual_training import tab_predict
    tab_predict.render()
```

- [ ] **Step 7.5: 静态校验**

```bash
python -c "import sys; sys.path.insert(0, '.'); from web_app.manual_training import tab_predict; assert callable(tab_predict.render); print('ok')"
grep -n "st.set_page_config\|init_session_state(\|sys.path.insert\|NotImplementedError\|^from utils\." web_app/manual_training/tab_predict.py
```

Expected: 第一行 `ok`，第二行无输出（包含 7 号页面的 `from utils.` 拼写残留检查）。

- [ ] **Step 7.6: 移动原 6 与 7 号页面**

```bash
mv web_app/pages/6_模型预测.py web_app/_legacy_pages/6_模型预测.py
mv web_app/pages/7_结果分析.py web_app/_legacy_pages/7_结果分析.py
```

- [ ] **Step 7.7: 手动冒烟**

```bash
streamlit run web_app/app.py --server.port 8501
```

测试：用「1️⃣ 数据准备」上传含相同 schema 的预测数据 →「6️⃣ 预测与结果」→
1. 「🔮 预测」选中之前训练的模型 → 加载预测数据 → 跑预测 → `predict_results` 写入 session
2. 「📋 结果分析」看到预测概率分布、阈值滑动器、TP/FP/TN/FN 统计

Expected: 与原 6 号、7 号页面行为一致。

- [ ] **Step 7.8: Commit**

```bash
git add web_app/manual_training/tab_predict.py "web_app/pages/4_📚_手动训练.py"
git add -A web_app/_legacy_pages/ web_app/pages/
git commit -m "$(cat <<'EOF'
feat(web_app): 合并模型预测 + 结果分析为 tab_predict

新增 web_app/manual_training/tab_predict.py，二级 tab 隔离预测控制与
结果分析；原 pages/6_模型预测.py 与 pages/7_结果分析.py 移至
web_app/_legacy_pages/。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: 全量验收（静态检查 + 端到端冒烟 + 回归）

**Files:** 无新建/修改

- [ ] **Step 8.1: 全量静态校验**

```bash
python -c "import sys; sys.path.insert(0, '.'); \
from web_app.manual_training import tab_data, tab_config, tab_eda, tab_train, tab_model_mgmt, tab_predict; \
for m in [tab_data, tab_config, tab_eda, tab_train, tab_model_mgmt, tab_predict]: \
    assert callable(m.render), f'{m.__name__}.render not callable'; \
print('all 6 modules OK')"
```

Expected: 输出 `all 6 modules OK`。

```bash
grep -rn "st.set_page_config\|init_session_state(\|sys.path.insert\|st.stop()" web_app/manual_training/
```

Expected: 无输出（所有 6 个 tab 模块零禁用调用残留）。

```bash
ls web_app/pages/
```

Expected：除了 `4_📚_手动训练.py` 外，剩下的页面应只有：

```
4_📚_手动训练.py
10_自动迭代.py
11_数据库连接管理.py
12_大模型配置管理.py
13_AI_智能分析.py
14_🤖_Agent工作流.py
15_📊_AutoML 训练.py
16_📈_MLOps 监控.py
__init__.py
```

```bash
ls web_app/_legacy_pages/
```

Expected：

```
1_数据管理.py
2_配置管理.py
3_数据分析.py
4_模型训练.py
5_模型管理.py
6_模型预测.py
7_结果分析.py
8_数据源管理.py
9_模型对比.py
README.md
```

- [ ] **Step 8.2: 入口页行数校验**

```bash
wc -l "web_app/pages/4_📚_手动训练.py"
```

Expected: 行数 ≤ 100（spec §6 验收标准）。如超过，删除冗余空行/注释直至 ≤ 100。

- [ ] **Step 8.3: 端到端手动冒烟（按 spec §6 清单）**

```bash
streamlit run web_app/app.py --server.port 8501
```

进入「📚 手动训练」，逐 Tab 验证（每条对照原页面行为）：

| Tab | 动作 | 验收 |
|---|---|---|
| 1️⃣ 数据准备 → 📁 文件上传 | 上传 sample CSV | 预览/字段/统计/导出四子 tab 正常；session.uploaded_data 写入 |
| 1️⃣ 数据准备 → 🗄️ 数据库源 | 加载已存连接 | UI 与原 8 号页面一致 |
| 2️⃣ 字段配置 | 编辑字段 → 保存 | xlsx 写入磁盘；`config_modified=True` |
| 3️⃣ 数据探索 | 4 子 tab 切换 + HTML 报告下载 | 图表渲染、HTML 文件下载成功 |
| 4️⃣ 模型训练 | 调参 → 训练 → 看结果 | trained_models 增加；4 子图表正常 |
| 5️⃣ 模型管理与对比 → 📦 模型列表 | 看模型、查看详情 | 列表正常 |
| 5️⃣ 模型管理与对比 → ⚖️ 模型对比 | 选 2 个模型 → 对比 | 指标对比图正常 |
| 6️⃣ 预测与结果 → 🔮 预测 | 选模型 + 上传数据 + 跑 | predict_results 写入 |
| 6️⃣ 预测与结果 → 📋 结果分析 | 调阈值 → 看分布 | 与原 7 号页面一致 |

- [ ] **Step 8.4: 回归测试（其它入口）**

在同一 streamlit 会话，依次进入：

- `10_自动迭代.py`
- `11_数据库连接管理.py`
- `12_大模型配置管理.py`
- `13_AI_智能分析.py`
- `14_🤖_Agent工作流.py`
- `15_📊_AutoML 训练.py`
- `16_📈_MLOps 监控.py`

每个页面：点击主按钮 / 看主图，确认无报错；尤其 14/15/16 号页面应能读到 `st.session_state.uploaded_data`、`st.session_state.field_config`、`st.session_state.trained_models`（这些是 Task 2-7 共同写入的状态）。

Expected: 7 个回归页面无报错；session_state 跨 tab 共享正常。

- [ ] **Step 8.5: 最终 commit（如有任何修补）**

如果 8.1-8.4 中发现并修补了小问题：

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(web_app): 手动训练合并验收修补

通过端到端冒烟与回归测试发现的小问题集中修补。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

如果 8.1-8.4 全部一次通过、无修补，跳过此步。

---

## Task 9: 文档收尾

**Files:**
- Modify: `web_app/_legacy_pages/README.md`（补充入口与回滚指引）

- [ ] **Step 9.1: 完善备份目录 README**

把 `web_app/_legacy_pages/README.md` 改为：

```markdown
# Legacy Pages 备份

此目录保留 2026-05-01 合并前的 9 个手动训练页面源码：

- `1_数据管理.py` → 合并入 `web_app/manual_training/tab_data.py`（文件上传子区）
- `2_配置管理.py` → `web_app/manual_training/tab_config.py`
- `3_数据分析.py` → `web_app/manual_training/tab_eda.py`
- `4_模型训练.py` → `web_app/manual_training/tab_train.py`
- `5_模型管理.py` → `web_app/manual_training/tab_model_mgmt.py`（模型列表子区）
- `6_模型预测.py` → `web_app/manual_training/tab_predict.py`（预测控制子区）
- `7_结果分析.py` → `web_app/manual_training/tab_predict.py`（结果分析子区）
- `8_数据源管理.py` → `web_app/manual_training/tab_data.py`（数据库源子区）
- `9_模型对比.py` → `web_app/manual_training/tab_model_mgmt.py`（模型对比子区）

合并后入口：`web_app/pages/4_📚_手动训练.py`
设计文档：`docs/superpowers/specs/2026-05-01-manual-training-merge-design.md`
实施计划：`docs/superpowers/plans/2026-05-01-manual-training-merge.md`

> 此目录在 `web_app/` 下但**不在** `pages/` 下，因此 Streamlit 不会将其纳入侧边栏。

## 回滚方法

### 单文件回滚（恢复某个原页面）

```bash
mv web_app/_legacy_pages/X_xxx.py web_app/pages/
```

### 全量回滚（撤销整个合并）

```bash
mv web_app/_legacy_pages/[0-9]*_*.py web_app/pages/
rm -rf web_app/manual_training/
rm "web_app/pages/4_📚_手动训练.py"
```

由于本次合并保持了 `session_state` 契约不变，回滚后行为与合并前完全一致。
```

- [ ] **Step 9.2: Commit**

```bash
git add web_app/_legacy_pages/README.md
git commit -m "$(cat <<'EOF'
docs(web_app): 补充手动训练合并 _legacy_pages README

罗列每个原页面对应的新 tab 模块去向，补充单文件 / 全量回滚命令。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## 完成标志

全部 Task 1-9 完成后：

✅ `web_app/pages/` 仅剩 1 个手动训练入口 `4_📚_手动训练.py` + 7 个未受影响的入口（10/11/12/13/14/15/16 + `__init__.py`）
✅ `web_app/manual_training/` 含 6 个 `tab_*.py` + `__init__.py`
✅ `web_app/_legacy_pages/` 含 9 个原页面备份 + `README.md`
✅ 静态校验：6 个 tab 模块导入正常，零禁用调用
✅ 端到端冒烟：spec §6 9 条全过
✅ 回归测试：14/15/16 + 10/11/12/13 全部正常工作
✅ 入口页 ≤ 100 行
✅ 9 次 git commit（每个 Task 一次，Task 8 可选）
