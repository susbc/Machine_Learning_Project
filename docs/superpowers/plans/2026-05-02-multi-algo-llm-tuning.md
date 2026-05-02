# 多算法 + LLM 调参 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将手动训练 Tab 从写死 LightGBM 升级为 5 种算法可选 + LLM 一键推荐参数。

**Architecture:** 新增 `param_specs.py` 定义 UI 参数元数据，新增 `llm_tuning.py` 做 LLM 调用+鲁棒解析，增量修改 `IntelligentAnalyzer` 增加 `recommend_hyperparameters()` 方法，改造 `tab_train.py` 替换写死逻辑。

**Tech Stack:** Streamlit, Python dataclasses, ModelFactory, IntelligentAnalyzer, LLMManager, json, re

---

### Task 1: 创建 `param_specs.py` — UI 参数元数据

**Files:**
- Create: `web_app/manual_training/param_specs.py`

- [ ] **Step 1: 实现 ParamSpec dataclass 和 get_specs 函数**

```python
"""
param_specs.py — 每算法的 UI 参数元数据。

定义 ParamSpec dataclass + get_specs(algorithm) 返回某算法在 UI 中暴露的参数列表。
供 tab_train.py 动态渲染 widgets，供 llm_tuning.py 做越界截断。
"""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    widget: Literal["slider", "number_input", "select", "checkbox"]
    default: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step: Optional[Any] = None
    fmt: Optional[str] = None
    options: Optional[List[Any]] = None
    help: Optional[str] = None


_ALGO_SPECS: dict[str, list[ParamSpec]] = {
    "lightgbm": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量，越多精度越高但越慢"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率，越小越稳定但需要更多树"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 6, 3, 15, 1, help="限制树深度，防止过拟合"),
        ParamSpec("num_leaves", "num_leaves (叶子节点数)", "slider", 50, 10, 200, 10, help="LightGBM 核心参数，控制模型复杂度"),
        ParamSpec("min_child_samples", "min_child_samples (叶子最小样本)", "slider", 20, 5, 100, 5, help="叶子节点最少样本数"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树使用的样本比例"),
        ParamSpec("colsample_bytree", "colsample_bytree (特征采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树使用的特征比例"),
        ParamSpec("reg_alpha", "reg_alpha (L1 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L1 正则化系数"),
        ParamSpec("reg_lambda", "reg_lambda (L2 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L2 正则化系数"),
    ],
    "xgboost": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 6, 3, 15, 1, help="树的最大深度"),
        ParamSpec("min_child_weight", "min_child_weight (叶子最小权重)", "slider", 1, 1, 20, 1, help="叶子节点最小权重和"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="样本采样比例"),
        ParamSpec("colsample_bytree", "colsample_bytree (特征采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="特征采样比例"),
        ParamSpec("reg_alpha", "reg_alpha (L1 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L1 正则化"),
        ParamSpec("reg_lambda", "reg_lambda (L2 正则)", "slider", 0.1, 0.0, 10.0, 0.1, "%.1f", help="L2 正则化"),
    ],
    "catboost": [
        ParamSpec("iterations", "iterations (迭代次数)", "slider", 200, 50, 1000, 50, help=" boosting 迭代次数"),
        ParamSpec("learning_rate", "learning_rate (学习率)", "slider", 0.1, 0.001, 0.3, 0.01, "%.3f", help="学习率"),
        ParamSpec("depth", "depth (树深度)", "slider", 6, 3, 15, 1, help="树深度"),
        ParamSpec("l2_leaf_reg", "l2_leaf_reg (L2 正则)", "slider", 3.0, 0.1, 20.0, 0.5, "%.1f", help="L2 正则化系数"),
        ParamSpec("border_count", "border_count (分箱数)", "slider", 256, 32, 512, 32, help="数值特征分箱数量"),
        ParamSpec("subsample", "subsample (样本采样比例)", "slider", 0.9, 0.5, 1.0, 0.05, "%.2f", help="每棵树样本采样比例"),
    ],
    "random_forest": [
        ParamSpec("n_estimators", "n_estimators (树的数量)", "slider", 200, 50, 1000, 50, help="树的数量"),
        ParamSpec("max_depth", "max_depth (树最大深度)", "slider", 10, 3, 30, 1, help="树最大深度，None 表示不限制"),
        ParamSpec("min_samples_split", "min_samples_split (最小分割样本)", "slider", 2, 2, 20, 1, help="内部节点再分割所需最小样本数"),
        ParamSpec("min_samples_leaf", "min_samples_leaf (叶子最小样本)", "slider", 1, 1, 10, 1, help="叶子节点最少样本数"),
        ParamSpec("max_features", "max_features (最大特征数)", "select", "sqrt", options=["sqrt", "log2", None], help="每次分割考虑的最大特征数"),
    ],
    "logistic_regression": [
        ParamSpec("C", "C (正则化强度倒数)", "slider", 1.0, 0.001, 100.0, 0.1, "%.3f", help="正则化强度的倒数，越小正则化越强"),
        ParamSpec("penalty", "penalty (正则化类型)", "select", "l2", options=["l1", "l2"], help="正则化类型"),
        ParamSpec("solver", "solver (优化算法)", "select", "liblinear", options=["liblinear", "saga"], help="优化算法，l1 正则需 liblinear 或 saga"),
        ParamSpec("max_iter", "max_iter (最大迭代次数)", "slider", 1000, 100, 5000, 100, help="优化器最大迭代次数"),
    ],
}


def get_specs(algorithm: str) -> list[ParamSpec]:
    """返回某算法在 UI 中暴露的参数列表。"""
    specs = _ALGO_SPECS.get(algorithm)
    if specs is None:
        raise ValueError(f"不支持的算法：{algorithm}。支持的算法：{list(_ALGO_SPECS.keys())}")
    return specs


def get_supported_algorithms() -> list[str]:
    """返回所有支持的算法标识。"""
    return list(_ALGO_SPECS.keys())
```

- [ ] **Step 2: 静态校验**

Run:
```bash
python -c "
from web_app.manual_training.param_specs import get_specs, get_supported_algorithms
for a in get_supported_algorithms():
    specs = get_specs(a)
    print(f'{a}: {len(specs)} params')
print('OK')
"
```

Expected output:
```
lightgbm: 9 params
xgboost: 8 params
catboost: 6 params
random_forest: 5 params
logistic_regression: 4 params
OK
```

- [ ] **Step 3: Commit**

```bash
git add web_app/manual_training/param_specs.py
git commit -m "feat: 新增 param_specs.py 定义 5 种算法 UI 参数元数据

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
"
```

---

### Task 2: 为 `IntelligentAnalyzer` 新增 `recommend_hyperparameters()` 方法

**Files:**
- Modify: `model_code/llm/intelligent_analyzer.py`

- [ ] **Step 1: 在 `IntelligentAnalyzer` 类末尾（`save_analysis` 方法之后）新增 `recommend_hyperparameters` 方法**

在 `save_analysis` 方法之后、单例模式之前（约第 578 行 `class` 结束处）添加：

```python
    def recommend_hyperparameters(
        self,
        df: 'pd.DataFrame',
        target_col: str,
        model_type: str,
        eda_summary: Optional[Dict] = None,
        last_training: Optional[Dict] = None,
        business_context: str = "",
    ) -> str:
        """
        调用 LLM 推荐超参数。

        复用已有的 _get_hyperparameter_tuning_prompt 模板，
        在末尾追加结构化 JSON 输出指令。

        Args:
            df: 完整数据框
            target_col: 目标列名
            model_type: 算法标识 (lightgbm / xgboost / ...)
            eda_summary: EDA 摘要字典（可选）
            last_training: 上次训练结果（可选）
            business_context: 业务场景描述（可选）

        Returns:
            LLM 原文（包含 JSON + 解释文本）。
        """
        if pd is None:
            raise RuntimeError("需要安装 pandas")

        n_samples, n_features = df.shape
        n_numeric = len(df.select_dtypes(include=['int64', 'float64']).columns)
        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_rate = float(df.isna().sum().sum()) / (n_samples * n_features)

        # 类别不平衡
        if df[target_col].nunique() == 2:
            class_dist = df[target_col].value_counts(normalize=True)
            imbalance_ratio = float(class_dist.min() / class_dist.max())
        else:
            imbalance_ratio = 1.0

        # 高相关对（取前 5）
        high_corr_pairs = []
        if eda_summary and "high_corr_pairs" in eda_summary:
            high_corr_pairs = eda_summary["high_corr_pairs"][:5]

        # 异常值率（仅数值列）
        outlier_summary = {}
        if eda_summary and "outlier_rates" in eda_summary:
            outlier_summary = eda_summary["outlier_rates"]

        # 上次训练结果
        last_params = last_training.get("model_params", {}) if last_training else {}
        last_auc = last_training.get("test_auc") if last_training else None

        # 使用已有的 hyperparameter_tuning 模板
        current_params_str = json.dumps(last_params, indent=2, ensure_ascii=False) if last_params else f"用户将为 {model_type} 手动设置参数"

        history_text = f"上次训练 AUC: {last_auc:.4f}" if last_auc else "首次训练"

        prompt = self.prompts['hyperparameter_tuning'].format(
            model_type=model_type,
            current_params=current_params_str,
            history_records=history_text,
            performance_history="",
            max_train_time="未限制",
            compute_resources="单 CPU",
        )

        # 追加上下文 + JSON 输出指令
        json_instruction = f"""

## 当前数据画像
- 样本量：{n_samples}
- 特征数：{n_features}（数值：{n_numeric}，类别：{n_categorical}）
- 缺失率：{missing_rate:.2%}
- 类别不平衡比（少数/多数）：{imbalance_ratio:.4f}
- 高相关特征对：{high_corr_pairs if high_corr_pairs else '无'}
- 异常值概况：{outlier_summary if outlier_summary else '无'}
- 目标列：{target_col}
- 算法：{model_type}

## 输出要求
请在回答末尾输出一个完整的 JSON 代码块（用 ```json 包围），格式如下：
```json
{{
  "params": {{
    "n_estimators": 300,
    "learning_rate": 0.05
  }},
  "reasoning": "一段简短的理由，说明为什么推荐这些参数"
}}
```
JSON 中只包含适用于 {model_type} 的参数。参数值必须在 UI 控件的合理范围内。
"""

        messages = [{"role": "user", "content": prompt + json_instruction}]
        return self.llm_manager.chat(messages, temperature=0.3)
```

- [ ] **Step 2: 静态校验**

Run:
```bash
python -c "
from model_code.llm.intelligent_analyzer import IntelligentAnalyzer
assert hasattr(IntelligentAnalyzer, 'recommend_hyperparameters')
print('OK: recommend_hyperparameters method exists')
"
```

- [ ] **Step 3: Commit**

```bash
git add model_code/llm/intelligent_analyzer.py
git commit -m "feat: IntelligentAnalyzer 新增 recommend_hyperparameters() 方法

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
"
```

---

### Task 3: 创建 `llm_tuning.py` — LLM 调参薄壳 + 鲁棒解析

**Files:**
- Create: `web_app/manual_training/llm_tuning.py`

- [ ] **Step 1: 实现完整 llm_tuning.py**

```python
"""
llm_tuning.py — LLM 一键调参薄壳。

职责：收集上下文 → 调 IntelligentAnalyzer → 鲁棒解析 JSON → 截断越界值 → 返回结构化结果。
不依赖 streamlit（纯 Python），UI 层由 tab_train.py 负责。
"""

import json
import re
from typing import Any, Dict, List, Optional

from web_app.manual_training.param_specs import get_specs


class LLMTuningError(Exception):
    """LLM 调参异常。"""
    pass


def _build_context(
    df, target_col: str, eda_summary: Optional[Dict], last_training: Optional[Dict]
) -> Dict[str, Any]:
    """构建 EDA 上下文字典。"""
    import pandas as pd

    n_samples = len(df)
    n_features = len(df.columns) - 1  # 减去 target
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    n_numeric = len([c for c in numeric_cols if c != target_col])
    n_categorical = len([c for c in categorical_cols if c != target_col])

    # 缺失率（排除 target）
    feature_df = df[[c for c in df.columns if c != target_col]]
    missing_rate = float(feature_df.isna().sum().sum()) / (n_samples * n_features) if n_features > 0 else 0.0

    # 类别不平衡
    if df[target_col].nunique() == 2:
        counts = df[target_col].value_counts()
        imbalance_ratio = float(counts.min() / counts.max())
    else:
        imbalance_ratio = 1.0

    high_correlations = eda_summary.get("high_corr_pairs", [])[:5] if eda_summary else []
    outlier_summary = eda_summary.get("outlier_rates", {}) if eda_summary else {}

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "imbalance_ratio": imbalance_ratio,
        "missing_rate": missing_rate,
        "high_correlations": high_correlations,
        "outlier_summary": outlier_summary,
        "last_training": last_training,
    }


def _extract_json(text: str) -> str:
    """从 LLM 响应中提取 JSON 字符串。

    三层退化：
    1. 优先正则 r'```json\s*(.*?)\s*```' (DOTALL)
    2. 退化到 r'```\s*(\{.*?\})\s*```'
    3. 退化到首个完整大括号 \\{[^{}]*\"params\"[^{}]*\\}
    全失败则抛异常。
    """
    # Level 1: ```json ... ```
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Level 2: ``` { ... } ```
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Level 3: 首个包含 "params" 的大括号块
    m = re.search(r"\{[^{}]*\"params\"[^{}]*\}", text, re.DOTALL)
    if m:
        return m.group(0)

    raise LLMTuningError("LLM 返回非 JSON 格式")


def _clip_and_coerce(params: Dict[str, Any], algorithm: str) -> tuple[Dict[str, Any], List[str]]:
    """按 param_specs 过滤 + 截断 + 类型转换。

    Returns:
        (clipped_params, warnings)
    """
    specs = get_specs(algorithm)
    spec_map = {s.name: s for s in specs}
    clipped = {}
    warnings = []

    for key, value in params.items():
        if key not in spec_map:
            continue  # 过滤掉 UI 不认识的参数

        spec = spec_map[key]

        # 类型转换
        if spec.widget == "select":
            if value not in spec.options:
                warnings.append(f"{key} 值 '{value}' 不在选项中，已跳过")
                continue
            clipped[key] = value
            continue

        # 数值型参数
        try:
            if isinstance(spec.default, int):
                value = int(value)
            elif isinstance(spec.default, float):
                value = float(value)
        except (ValueError, TypeError):
            warnings.append(f"{key} 类型转换失败（'{value}'），已跳过")
            continue

        # 越界截断
        if spec.min_value is not None and value < spec.min_value:
            warnings.append(f"{key}={value} 低于下限 {spec.min_value}，已截到 {spec.min_value}")
            value = spec.min_value if isinstance(value, int) else float(spec.min_value)
        if spec.max_value is not None and value > spec.max_value:
            warnings.append(f"{key}={value} 超出上限 {spec.max_value}，已截到 {spec.max_value}")
            value = spec.max_value if isinstance(value, int) else float(spec.max_value)

        clipped[key] = value

    return clipped, warnings


def recommend_params(
    df,
    target_col: str,
    model_type: str,
    eda_summary: Optional[Dict] = None,
    last_training: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    LLM 推荐参数入口。

    Returns:
        {
            "params": {"n_estimators": 300, ...},
            "reasoning": "...",
            "raw": "...",
            "warnings": ["learning_rate 超出 UI 上限，已截到 0.3"],
        }

    Raises:
        LLMTuningError: LLM 不可用 / 无法解析 JSON / 解析后 params 为空
    """
    from model_code.llm.intelligent_analyzer import get_intelligent_analyzer

    analyzer = get_intelligent_analyzer()

    # 1. 调 LLM
    raw_response = analyzer.recommend_hyperparameters(
        df=df,
        target_col=target_col,
        model_type=model_type,
        eda_summary=eda_summary,
        last_training=last_training,
    )

    # 2. 鲁棒解析
    json_str = _extract_json(raw_response)
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise LLMTuningError(f"JSON 解析失败：{e}")

    # 3. 取 params
    params = payload.get("params")
    if not params or not isinstance(params, dict):
        raise LLMTuningError("LLM 未给出可应用的参数")

    reasoning = payload.get("reasoning", "")

    # 4. 过滤 + 截断
    clipped, warn_list = _clip_and_coerce(params, model_type)

    if not clipped:
        raise LLMTuningError("LLM 未给出可应用的参数")

    return {
        "params": clipped,
        "reasoning": reasoning,
        "raw": raw_response,
        "warnings": warn_list,
    }
```

- [ ] **Step 2: 静态校验**

Run:
```bash
python -c "
from web_app.manual_training.llm_tuning import LLMTuningError, _extract_json, _clip_and_coerce
print('OK: llm_tuning imports work')
"
```

- [ ] **Step 3: Commit**

```bash
git add web_app/manual_training/llm_tuning.py
git commit -m "feat: 新增 llm_tuning.py LLM 调参薄壳 + 鲁棒 JSON 解析

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
"
```

---

### Task 4: 重构 `tab_train.py` — 算法下拉 + 动态参数 widgets

**Files:**
- Modify: `web_app/manual_training/tab_train.py`

This is the largest task. I'll break it into clear steps.

- [ ] **Step 1: 在 render() 函数顶部添加 LLM apply 前置处理**

在 `st.header("🚀 模型训练")` 之前，加入：

```python
    # LLM 推荐参数「应用」前置：必须在 widget 渲染之前 pop
    if "train_llm_pending_apply" in st.session_state:
        pending = st.session_state.pop("train_llm_pending_apply")
        for k, v in pending.items():
            st.session_state[f"train_{k}"] = v
```

- [ ] **Step 2: 替换写死的 LightGBM 参数区域**

将第 44-136 行（`st.subheader("🎯 模型参数配置")` 到 9 个 slider 的结束）全部替换为：

```python
    st.subheader("🎯 算法选择")

    from web_app.manual_training.param_specs import get_specs, get_supported_algorithms
    from model_code.model_factory import ModelFactory
    from model_code.auto_selector import AutoModelSelector

    algo_options = get_supported_algorithms()
    algo_labels = [ModelFactory.get_display_name(a) for a in algo_options]

    current_algo_idx = st.session_state.get("train_algorithm_idx", 0)
    if current_algo_idx >= len(algo_options):
        current_algo_idx = 0

    selected_algo_display = st.selectbox(
        "选择算法",
        options=algo_labels,
        index=current_algo_idx,
        key="train_algo_selectbox",
    )
    selected_algo = algo_options[algo_labels.index(selected_algo_display)]
    st.session_state.train_algorithm_idx = algo_labels.index(selected_algo_display)
    st.session_state.train_algorithm = selected_algo

    # 智能推荐按钮
    if st.button("🧠 智能推荐算法", use_container_width=True, key="train_algo_recommend_btn"):
        with st.spinner("正在分析数据并推荐算法..."):
            from web_app.utils.session_state import add_training_log
            label_col = st.session_state.field_config[
                st.session_state.field_config["field_type"] == "label"
            ]["field_name"].values[0]
            feature_cols_raw = [
                c for c in df.columns if c != label_col
            ]
            feature_df = df[feature_cols_raw]
            rec = AutoModelSelector.recommend(feature_df, df[label_col])
            st.session_state.train_algo_recommendation = rec
            st.session_state.train_algo_show_recommendation = True

    if st.session_state.get("train_algo_show_recommendation"):
        rec = st.session_state.train_algo_recommendation
        with st.expander("🧠 算法推荐结果", expanded=True):
            st.info(f"**推荐算法**: {ModelFactory.get_display_name(rec['recommended'])}\n\n**理由**: {rec['reason']}")
            if rec.get("alternatives"):
                st.caption(f"备选: {', '.join(ModelFactory.get_display_name(a) for a in rec['alternatives'][:3])}")
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                if st.button("✅ 采纳推荐", use_container_width=True, key="train_algo_adopt_btn"):
                    recommended_idx = algo_options.index(rec["recommended"])
                    st.session_state.train_algorithm_idx = recommended_idx
                    st.session_state.train_algo_show_recommendation = False
                    st.rerun()
            with col_rec2:
                if st.button("❌ 关闭", use_container_width=True, key="train_algo_close_btn"):
                    st.session_state.train_algo_show_recommendation = False
                    st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 模型参数配置")

    with st.expander(f"{ModelFactory.get_display_name(selected_algo)} 参数设置", expanded=True):
        specs = get_specs(selected_algo)
        collected_params = {}
        cols = st.columns(3)
        for i, spec in enumerate(specs):
            with cols[i % 3]:
                widget_key = f"train_{spec.name}"
                current = st.session_state.model_params.get(spec.name, spec.default)
                if spec.widget == "slider":
                    v = st.slider(
                        spec.label,
                        min_value=spec.min_value,
                        max_value=spec.max_value,
                        value=current,
                        step=spec.step,
                        format=spec.fmt,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "select":
                    idx = spec.options.index(current) if current in spec.options else 0
                    v = st.selectbox(
                        spec.label,
                        options=spec.options,
                        index=idx,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "number_input":
                    v = st.number_input(
                        spec.label,
                        value=current,
                        min_value=spec.min_value or 0,
                        max_value=spec.max_value,
                        step=spec.step,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "checkbox":
                    v = st.checkbox(
                        spec.label,
                        value=current,
                        help=spec.help,
                        key=widget_key,
                    )
                collected_params[spec.name] = v
```

- [ ] **Step 3: 替换旧的 `st.session_state.model_params` 赋值**

将原来第 189-199 行（`st.session_state.model_params = {"n_estimators": ...}`）替换为：

```python
    st.session_state.model_params = collected_params
```

（注意：`collected_params` 是上面动态渲染循环中收集的变量，它在同一作用域内）

- [ ] **Step 4: 在训练配置区块后、训练控制区块前，插入 LLM 调参 UI 区块**

在 `use_grid_search` / `n_iter` 控件之后（原 `st.session_state.model_params = ...` 赋值之后），`🚀 训练控制` 之前，插入：

```python
    # ====== LLM 一键调参区块 ======
    st.markdown("---")
    st.subheader("🤖 LLM 一键调参")

    from web_app.manual_training.llm_tuning import recommend_params, LLMTuningError

    llm_state = st.session_state.get("train_llm_state", "idle")
    llm_rec = st.session_state.get("train_llm_recommendation")

    if llm_state == "idle" or llm_state == "error":
        if st.button("🤖 LLM 推荐参数", use_container_width=True, key="train_llm_request_btn"):
            # 检查 LLM 配置
            from model_code.llm.llm_connector import get_llm_manager
            llm_mgr = get_llm_manager()
            if not llm_mgr.providers:
                st.warning("⚠️ 请先在「12_大模型配置管理」页面配置 LLM 提供商")
            else:
                st.session_state.train_llm_state = "loading"
                st.rerun()

    if llm_state == "loading":
        with st.spinner("正在调用 LLM 分析并推荐参数..."):
            try:
                label_col = st.session_state.field_config[
                    st.session_state.field_config["field_type"] == "label"
                ]["field_name"].values[0]
                result = recommend_params(
                    df=df,
                    target_col=label_col,
                    model_type=selected_algo,
                    eda_summary=st.session_state.get("eda_summary_cache"),
                    last_training=st.session_state.get("training_results"),
                )
                st.session_state.train_llm_recommendation = result
                st.session_state.train_llm_state = "has_recommendation"
            except LLMTuningError as e:
                st.session_state.train_llm_error = str(e)
                st.session_state.train_llm_raw_backup = ""
                st.session_state.train_llm_state = "error"
            except Exception as e:
                st.session_state.train_llm_error = f"调用 LLM 失败：{e}"
                st.session_state.train_llm_state = "error"
            st.rerun()

    if llm_state == "has_recommendation" and llm_rec:
        with st.expander("🤖 LLM 推荐结果", expanded=True):
            st.info(f"**推荐理由**\n\n{llm_rec['reasoning']}")
            st.code(json.dumps(llm_rec["params"], indent=2, ensure_ascii=False), language="json")
            if llm_rec.get("warnings"):
                for w in llm_rec["warnings"]:
                    st.warning(f"⚠️ {w}")
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                if st.button("✅ 应用到参数控件", use_container_width=True, key="train_llm_apply_btn"):
                    st.session_state.train_llm_pending_apply = llm_rec["params"]
                    st.session_state.train_llm_state = "idle"
                    st.session_state.train_llm_recommendation = None
                    st.rerun()
            with col_llm2:
                if st.button("🗑️ 清除推荐", use_container_width=True, key="train_llm_clear_btn"):
                    st.session_state.train_llm_state = "idle"
                    st.session_state.train_llm_recommendation = None
                    st.rerun()
```

- [ ] **Step 5: 替换训练执行中的 LightGBM 写死逻辑**

将第 314-408 行（`try: import lightgbm as lgb ...` 到 `model.fit(X_train, y_train)` 整个 LightGBM + RF 分支）替换为：

```python
            add_training_log(f"\n使用 {ModelFactory.get_display_name(selected_algo)} 模型训练...")

            try:
                base_model = ModelFactory.build(
                    algorithm=selected_algo,
                    task_type="binary",
                    params=st.session_state.model_params,
                )
            except ImportError as e:
                add_training_log(f"\n❌ {selected_algo} 未安装：{e}")
                st.error(f"{ModelFactory.get_display_name(selected_algo)} 未安装")
                if selected_algo == "xgboost":
                    st.code("pip install xgboost")
                elif selected_algo == "catboost":
                    st.code("pip install catboost")
                st.session_state.training_in_progress = False
                return
            except Exception as e:
                add_training_log(f"\n❌ 模型构建失败：{e}")
                st.error(f"模型构建失败：{e}")
                st.session_state.training_in_progress = False
                return

            if use_grid_search and n_iter > 1:
                add_training_log(f"\n执行随机网格搜索 (n_iter={n_iter}, cv={cv_folds})...")
                param_grid = ModelFactory.get_param_grid(selected_algo)
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=0,
                    random_state=random_state,
                )
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                add_training_log(f"最佳参数：{random_search.best_params_}")
                st.session_state.cv_results = random_search.cv_results_
            else:
                add_training_log("\n使用固定参数训练...")
                model = base_model
                model.fit(X_train, y_train)
```

- [ ] **Step 6: 模型文件名加上算法标识**

将第 448-451 行的模型保存路径：
```python
            model_path = (
                st.session_state.model_dir
                / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
```
改为：
```python
            model_path = (
                st.session_state.model_dir
                / f"model_{selected_algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
```

- [ ] **Step 7: training_results 新增 algorithm 字段**

在第 431-446 行的 `st.session_state.training_results = {...}` 字典中，加入 `"algorithm": selected_algo`：

```python
            st.session_state.training_results = {
                "model": model,
                "algorithm": selected_algo,
                "model_params": st.session_state.model_params,
                ...  # 其余不变
```

- [ ] **Step 8: 在文件顶部 import json（用于 LLM 推荐结果的 JSON 显示）**

在 `import streamlit as st` 后添加 `import json`。

- [ ] **Step 9: 静态校验**

Run:
```bash
python -c "
import ast
with open('web_app/manual_training/tab_train.py') as f:
    tree = ast.parse(f.read())
print(f'tab_train.py: {len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef])} functions, syntax OK')
"
```

- [ ] **Step 10: Commit**

```bash
git add web_app/manual_training/tab_train.py
git commit -m "feat: tab_train 重构为多算法 + LLM 调参

- 算法下拉选择 5 种算法
- 动态参数 widgets 基于 param_specs
- LLM 一键调参区块
- 使用 ModelFactory.build() 替代写死 LightGBM
- 模型文件名增加算法标识

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
"
```

---

### Task 5: 编写 `tests/test_llm_tuning.py` — 单元测试

**Files:**
- Create: `tests/test_llm_tuning.py`

- [ ] **Step 1: 实现 6 个单元测试**

```python
"""
tests/test_llm_tuning.py — LLM 调参鲁棒解析单元测试。

不写 LLM 调用本身的集成测试（依赖外部 API key）。
只测试 _extract_json, _clip_and_coerce 的纯函数行为。
"""

import pytest
from web_app.manual_training.llm_tuning import LLMTuningError, _extract_json, _clip_and_coerce


def test_parse_clean_json_block():
    """完美 ```json``` 块 → 解析成功。"""
    text = """
Here is my analysis...

```json
{
  "params": {"n_estimators": 300, "learning_rate": 0.05},
  "reasoning": "Based on the data characteristics..."
}
```

Hope this helps!
"""
    result = _extract_json(text)
    assert '"n_estimators"' in result
    assert '"learning_rate"' in result


def test_parse_dirty_response():
    """LLM 在 JSON 前后有解释文本 → 仍能提取。"""
    text = """
好的，让我来分析一下。基于数据特征，我推荐以下参数：

首先，learning_rate 应该降低到 0.05 左右。
其次，n_estimators 可以增加到 300。

```json
{
  "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8},
  "reasoning": "降低学习率增加树数量是常见的调参策略"
}
```

以上就是我的建议，你可以根据实际情况微调。
"""
    result = _extract_json(text)
    assert '"n_estimators": 300' in result


def test_parse_no_json():
    """全是文本无 JSON → 抛 LLMTuningError。"""
    text = """
我认为应该把学习率调到 0.01，树的数量 200 就够了。
max_depth 设为 6 比较合适。
"""
    with pytest.raises(LLMTuningError, match="非 JSON"):
        _extract_json(text)


def test_clip_out_of_range():
    """learning_rate=2.0 → 截到 0.3。"""
    params = {"learning_rate": 2.0, "n_estimators": 200}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert clipped["learning_rate"] == 0.3
    assert any("超出上限" in w for w in warnings)


def test_filter_unknown_params():
    """返回 gamma=0.1 但 LightGBM 不暴露 → 过滤。"""
    params = {"gamma": 0.1, "n_estimators": 200}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert "gamma" not in clipped
    assert "n_estimators" in clipped


def test_string_value_coercion():
    """"300" → int(300)。"""
    params = {"n_estimators": "300", "learning_rate": "0.05"}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert clipped["n_estimators"] == 300
    assert isinstance(clipped["n_estimators"], int)
    assert clipped["learning_rate"] == 0.05
    assert isinstance(clipped["learning_rate"], float)
```

- [ ] **Step 2: 运行测试**

Run:
```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project && python -m pytest tests/test_llm_tuning.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_llm_tuning.py
git commit -m "test: 新增 llm_tuning 6 个单元测试

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
"
```

---

### Task 6: 手动冒烟测试 + 最终验证

**Files:** No code changes.

- [ ] **Step 1: 启动 Streamlit 应用**

Run:
```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project && streamlit run web_app/app.py
```

- [ ] **Step 2: 手动冒烟清单验证**

在浏览器中执行以下检查：

| ID | 操作 | 期望 |
|---|---|---|
| A1 | 进入 4_手动训练 → 4_模型训练 Tab，算法下拉切换 5 种 | widgets 变化；公共参数值保留 |
| A2 | 5 种算法各跑一次训练 | 训练成功；模型文件名含 `<algo>` |
| A3 | 启用网格搜索 | 用 `ModelFactory.get_param_grid` 不报错 |
| A4 | 卸载 xgboost 后选 XGB | 友好报错 + pip 命令提示 |
| B1 | 没配 LLM 点推荐 | 警告提示去 12 号页面 |
| B2 | 配好后点推荐 | spinner → JSON + reasoning |
| B3 | 点「应用」 | widgets 值变化 |
| C1 | 回归 tab_data | 不变 |
| C2 | 回归 tab_eda | EDA cache 命中 |
| C3 | 回归 14/15/16 号页面 | 不报错 |

- [ ] **Step 3: 确认验收标准**

1. ✅ 5 种算法全部能选择并训练
2. ✅ 切换算法后 widgets 动态切换，公共参数继承
3. ✅ LLM 推荐端到端跑通（如有 LLM 配置）
4. ✅ 6 个单元测试全过
5. ✅ A1-A4 + C1-C3 手动冒烟全过
6. ✅ `tab_train.py` ≤ 600 行（运行 `wc -l web_app/manual_training/tab_train.py` 确认）

---