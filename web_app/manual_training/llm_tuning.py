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
    1. 优先正则 r'```json\\s*(.*?)\\s*```' (DOTALL)
    2. 退化到 r'```\\s*(\\{.*?\\})\\s*```'
    3. 退化到首个包含 "params" 的大括号块
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

    # Level 3: 首个包含 "params" 的大括号块（贪婪匹配，支持嵌套）
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if '"params"' in candidate:
                        return candidate
                    break
        start = text.find("{", start + 1)

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
            warnings.append(f"{key}={value} 低于下限 {spec.min_value}，已截到下限")
            value = type(spec.default)(spec.min_value)
        if spec.max_value is not None and value > spec.max_value:
            warnings.append(f"{key}={value} 超出上限 {spec.max_value}，已截到上限")
            value = type(spec.default)(spec.max_value)

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
            "param_reasons": {"n_estimators": "...", ...},
            "data_analysis": "...",
            "strategy": "...",
            "expected_improvement": "...",
            "raw": "...",
            "warnings": [...],
        }

    Raises:
        LLMTuningError: LLM 不可用 / 无法解析 JSON / 解析后 params 为空
    """
    from model_code.llm.intelligent_analyzer import get_intelligent_analyzer

    analyzer = get_intelligent_analyzer()

    # 构造 param_ranges 给 LLM 参考
    specs = get_specs(model_type)
    param_ranges = {}
    for spec in specs:
        info = {"default": spec.default}
        if spec.options:
            info["options"] = spec.options
        else:
            info["min"] = spec.min_value
            info["max"] = spec.max_value
        param_ranges[spec.name] = info

    # 1. 调 LLM
    raw_response = analyzer.recommend_hyperparameters(
        df=df,
        target_col=target_col,
        model_type=model_type,
        eda_summary=eda_summary,
        last_training=last_training,
        param_ranges=param_ranges,
    )

    if not raw_response or not raw_response.strip():
        raise LLMTuningError("LLM 返回空内容，请检查模型配置或网络")

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

    # 4. 过滤 + 截断
    clipped, warn_list = _clip_and_coerce(params, model_type)
    if not clipped:
        raise LLMTuningError("LLM 给出的参数全部不可用（不在 UI 暴露范围内）")

    # 5. param_reasons 过滤（只保留 clipped 中存在的参数）
    raw_reasons = payload.get("param_reasons") or {}
    if not isinstance(raw_reasons, dict):
        raw_reasons = {}
    param_reasons = {k: raw_reasons[k] for k in clipped if k in raw_reasons}

    return {
        "params": clipped,
        "param_reasons": param_reasons,
        "data_analysis": payload.get("data_analysis", ""),
        "strategy": payload.get("strategy", ""),
        "expected_improvement": payload.get("expected_improvement", ""),
        "raw": raw_response,
        "warnings": warn_list,
    }
