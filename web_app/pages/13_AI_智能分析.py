"""
AI 智能分析页面

面向当前项目真实流程：
- 数据集分析 + 模型推荐
- 训练结果诊断
- 超参数调优建议
- 预测结果分析
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_code.llm import get_intelligent_analyzer, get_llm_manager
from web_app.utils.session_state import init_session_state


st.set_page_config(
    page_title="AI 智能分析",
    page_icon="🧠",
    layout="wide",
)

init_session_state()


def init_analyzer():
    """初始化分析器和 LLM 管理器。"""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = get_intelligent_analyzer()
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = get_llm_manager()
    return st.session_state.analyzer


def _active_llm():
    manager = st.session_state.llm_manager
    return manager.get_active_provider()


def _ensure_active_llm() -> bool:
    if _active_llm() is None:
        st.warning("请先在「大模型配置管理」中配置并激活一个可用的大模型。")
        return False
    return True


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _format_optional_metric(value, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "未提供"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "未提供"


def _run_llm_task(label: str, callback):
    if not _ensure_active_llm():
        return None

    try:
        with st.spinner(f"AI 正在{label}..."):
            result = callback()
        st.success("分析完成")
        return result
    except Exception as e:
        st.error(f"{label}失败：{type(e).__name__}: {e}")
        with st.expander("查看错误详情", expanded=False):
            st.code(traceback.format_exc(), language="text")
        return None


def _get_uploaded_data() -> Optional[pd.DataFrame]:
    df = st.session_state.get("uploaded_data")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _detect_target_col(df: pd.DataFrame) -> str:
    config_df = st.session_state.get("field_config")
    if isinstance(config_df, pd.DataFrame) and "field_type" in config_df.columns:
        label_rows = config_df[config_df["field_type"] == "label"]
        if not label_rows.empty:
            candidate = label_rows.iloc[0]["field_name"]
            if candidate in df.columns:
                return candidate

    common_names = {"label", "target", "y", "class", "is_fraud", "标签", "目标", "是否违约"}
    for col in df.columns:
        if str(col).strip().lower() in common_names:
            return col
    return df.columns[-1]


def _data_quality_snapshot(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    feature_df = df.drop(columns=[target_col], errors="ignore")
    return {
        "rows": len(df),
        "features": feature_df.shape[1],
        "missing_rate": float(df.isna().sum().sum()) / max(df.size, 1),
        "duplicate_rate": float(df.duplicated().mean()) if len(df) else 0.0,
        "target_unique": int(df[target_col].nunique(dropna=True)),
        "numeric_features": len(feature_df.select_dtypes(include=["number"]).columns),
        "categorical_features": len(feature_df.select_dtypes(include=["object", "category", "bool"]).columns),
    }


def _save_analysis_button(key: str, analysis_type: str, content_key: str, metadata: Dict[str, Any], analyzer) -> None:
    disabled = not st.session_state.get(content_key)
    if st.button("保存报告", key=key, use_container_width=True, disabled=disabled):
        filepath = analyzer.save_analysis(
            analysis_type,
            st.session_state[content_key],
            metadata,
        )
        st.success(f"报告已保存：{filepath}")


def _training_context_from_session() -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[str, float]]]]:
    results = st.session_state.get("training_results")
    if not isinstance(results, dict):
        return None

    params = dict(results.get("model_params") or {})
    params.setdefault("model_type", results.get("algorithm", "Unknown"))

    metrics = {
        "train_auc": results.get("train_auc"),
        "val_auc": results.get("val_auc", results.get("test_auc")),
        "train_f1": results.get("train_f1"),
        "val_f1": results.get("val_f1", results.get("best_f1")),
        "train_size": results.get("train_size", 0),
        "val_size": results.get("test_size", 0),
        "n_features": len(results.get("feature_names", [])),
    }

    feature_names = list(results.get("feature_names") or [])
    importances = results.get("feature_importances")
    feature_importance: List[Tuple[str, float]] = []
    if feature_names and importances is not None:
        values = np.asarray(importances, dtype=float).reshape(-1)
        feature_importance = sorted(
            zip(feature_names, values),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:10]

    return metrics, params, feature_importance


def _history_from_cv_results() -> List[Dict[str, Any]]:
    cv_results = st.session_state.get("cv_results")
    if not isinstance(cv_results, dict):
        return []

    params = cv_results.get("params") or []
    scores = cv_results.get("mean_test_score")
    if not params or scores is None:
        return []

    scores_array = np.asarray(scores, dtype=float)
    ranked_idx = np.argsort(np.nan_to_num(scores_array, nan=-np.inf))[::-1][:8]
    history = []
    for idx in ranked_idx:
        history.append({
            "params": params[int(idx)],
            "auc": float(scores_array[int(idx)]),
        })
    return history


def _prediction_series_from_session() -> Tuple[Optional[pd.Series], str]:
    predict_results = st.session_state.get("predict_results")
    if isinstance(predict_results, pd.DataFrame) and "预测概率" in predict_results.columns:
        values = pd.to_numeric(predict_results["预测概率"], errors="coerce").dropna()
        if not values.empty:
            return values, "当前预测结果"

    training_results = st.session_state.get("training_results")
    if isinstance(training_results, dict) and training_results.get("y_pred_proba") is not None:
        proba = np.asarray(training_results["y_pred_proba"])
        if proba.ndim == 2 and proba.shape[1] >= 2:
            proba = proba[:, 1] if proba.shape[1] == 2 else np.max(proba, axis=1)
        values = pd.Series(np.asarray(proba).reshape(-1)).dropna()
        if not values.empty:
            return values, "训练测试集预测概率"

    return None, ""


def render_data_analysis_tab(analyzer) -> None:
    st.header("数据集智能分析")

    df = _get_uploaded_data()
    if df is None:
        st.warning("请先在「手动训练流程 / 数据准备」中加载数据。")
        return

    detected_target = _detect_target_col(df)
    target_col = st.selectbox(
        "目标变量",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(detected_target),
        key="ai_data_target_col",
    )

    snapshot = _data_quality_snapshot(df, target_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", f"{snapshot['rows']:,}")
    c2.metric("特征数", snapshot["features"])
    c3.metric("缺失率", f"{snapshot['missing_rate']:.2%}")
    c4.metric("目标唯一值", snapshot["target_unique"])

    with st.expander("数据预览", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)

    st.subheader("业务与模型偏好")
    business_context = st.text_area(
        "业务场景",
        placeholder="例如：金融风控二分类，希望更关注召回率，同时需要模型解释性。",
        height=90,
        key="ai_business_context",
    )

    col_req1, col_req2, col_req3 = st.columns(3)
    with col_req1:
        speed_requirement = st.selectbox("预测速度要求", ["中", "高", "低"], key="ai_speed_req")
    with col_req2:
        interpretability_requirement = st.selectbox("可解释性要求", ["中", "高", "低"], key="ai_interp_req")
    with col_req3:
        accuracy_requirement = st.selectbox("准确率要求", ["高", "中", "极高"], key="ai_acc_req")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("分析数据集", type="primary", use_container_width=True, key="ai_analyze_dataset_btn"):
            result = _run_llm_task(
                "分析数据集",
                lambda: analyzer.analyze_dataset(df, target_col, business_context),
            )
            if result:
                st.session_state.data_analysis = result
    with col2:
        if st.button("推荐模型", use_container_width=True, key="ai_recommend_model_btn"):
            requirements = {
                "speed_requirement": speed_requirement,
                "interpretability_requirement": interpretability_requirement,
                "accuracy_requirement": accuracy_requirement,
            }
            result = _run_llm_task(
                "推荐模型",
                lambda: analyzer.recommend_model(df, target_col, business_context, requirements),
            )
            if result:
                st.session_state.model_recommendation = result
    with col3:
        _save_analysis_button(
            "ai_data_report_btn",
            "data_analysis",
            "data_analysis",
            {"dataset_shape": df.shape, "target": target_col, "snapshot": snapshot},
            analyzer,
        )

    if st.session_state.get("data_analysis"):
        st.markdown("---")
        st.subheader("数据分析结果")
        st.markdown(st.session_state.data_analysis)

    if st.session_state.get("model_recommendation"):
        st.markdown("---")
        st.subheader("模型推荐结果")
        st.markdown(st.session_state.model_recommendation)


def render_training_analysis_tab(analyzer) -> None:
    st.header("训练结果智能分析")

    context = _training_context_from_session()
    manual_mode = context is None

    if context is None:
        st.info("未检测到训练结果。完成手动训练或 AutoML 后，这里会自动读取真实结果。")
        use_manual = st.checkbox("手动录入指标进行临时分析", value=False, key="ai_train_manual_mode")
        if not use_manual:
            return

        col1, col2 = st.columns(2)
        with col1:
            train_auc = st.number_input("训练集 AUC（可选）", 0.0, 1.0, value=0.0, step=0.01)
            train_f1 = st.number_input("训练集 F1（可选）", 0.0, 1.0, value=0.0, step=0.01)
            train_size = st.number_input("训练样本数", min_value=0, value=0, step=100)
        with col2:
            val_auc = st.number_input("验证/测试集 AUC", 0.0, 1.0, value=0.85, step=0.01)
            val_f1 = st.number_input("验证/测试集 F1", 0.0, 1.0, value=0.80, step=0.01)
            val_size = st.number_input("验证/测试样本数", min_value=0, value=0, step=100)

        metrics = {
            "train_auc": train_auc or None,
            "val_auc": val_auc,
            "train_f1": train_f1 or None,
            "val_f1": val_f1,
            "train_size": train_size,
            "val_size": val_size,
            "n_features": st.number_input("特征数量", min_value=0, value=0),
        }
        params = {"model_type": st.text_input("模型类型", value="LightGBM")}
        feature_importance = []
    else:
        metrics, params, feature_importance = context
        st.success("已读取当前训练结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("测试 ROC-AUC", _format_optional_metric(metrics.get("val_auc")))
        c2.metric("测试 F1", _format_optional_metric(metrics.get("val_f1")))
        c3.metric("训练样本", f"{metrics.get('train_size', 0):,}")
        c4.metric("特征数", metrics.get("n_features", 0))

        with st.expander("当前模型参数", expanded=False):
            st.json(params)
        if feature_importance:
            with st.expander("Top 特征重要性", expanded=False):
                st.dataframe(
                    pd.DataFrame(feature_importance, columns=["特征", "重要性"]),
                    use_container_width=True,
                    hide_index=True,
                )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("分析训练结果", type="primary", use_container_width=True, key="ai_train_analyze_btn"):
            result = _run_llm_task(
                "分析训练结果",
                lambda: analyzer.analyze_training_results(metrics, params, feature_importance),
            )
            if result:
                st.session_state.training_analysis = result
    with col2:
        _save_analysis_button(
            "ai_train_report_btn",
            "training_analysis",
            "training_analysis",
            {"metrics": metrics, "params": params, "source": "manual_input" if manual_mode else "training_results"},
            analyzer,
        )

    if st.session_state.get("training_analysis"):
        st.markdown("---")
        st.subheader("训练结果分析")
        st.markdown(st.session_state.training_analysis)


def render_hyperparameter_tab(analyzer) -> None:
    st.header("超参数智能调优")

    context = _training_context_from_session()
    params = context[1] if context else dict(st.session_state.get("model_params", {}))
    if "model_type" not in params and st.session_state.get("training_results"):
        params["model_type"] = st.session_state.training_results.get("algorithm", "Unknown")

    current_params = st.text_area(
        "当前参数（JSON）",
        value=json.dumps(
            params or {"model_type": "LightGBM"},
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        height=180,
        key="ai_tuning_params_json",
    )

    auto_history = _history_from_cv_results()
    if auto_history:
        st.caption(f"已从最近一次随机搜索读取 {len(auto_history)} 条调参记录。")
        with st.expander("自动读取的调参记录", expanded=False):
            st.dataframe(pd.DataFrame(auto_history), use_container_width=True, hide_index=True)

    history_text = st.text_area(
        "补充历史记录（可选，每行格式：param=value,param2=value → AUC: 0.87）",
        height=120,
        key="ai_tuning_history_text",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_time = st.selectbox("最大训练时间", ["30 分钟", "10 分钟", "1 小时", "2 小时"], key="ai_tuning_time")
    with col2:
        compute = st.selectbox("计算资源", ["单 CPU", "多 CPU", "GPU", "多 GPU"], key="ai_tuning_compute")

    if st.button("获取调参建议", type="primary", use_container_width=True, key="ai_tuning_btn"):
        try:
            parsed_params = json.loads(current_params)
        except json.JSONDecodeError as e:
            st.error(f"参数 JSON 格式错误：{e}")
            return

        history = list(auto_history)
        for line in history_text.strip().splitlines():
            if "→" not in line or "AUC" not in line:
                continue
            try:
                params_part, score_part = line.split("→", 1)
                auc = float(score_part.split(":")[-1].strip())
                history.append({"params": params_part.strip(), "auc": auc})
            except Exception:
                continue

        constraints = {"max_train_time": max_time, "compute_resources": compute}
        result = _run_llm_task(
            "生成调参建议",
            lambda: analyzer.suggest_hyperparameter_tuning(parsed_params, history, constraints),
        )
        if result:
            st.session_state.tuning_analysis = result

    if st.session_state.get("tuning_analysis"):
        st.markdown("---")
        st.subheader("调参建议")
        st.markdown(st.session_state.tuning_analysis)


def render_prediction_analysis_tab(analyzer) -> None:
    st.header("预测结果智能分析")

    predictions, source = _prediction_series_from_session()
    if predictions is None:
        st.info("暂无预测结果。完成预测后会自动读取 `predict_results`；也可先训练模型，用测试集预测概率做临时分析。")
        return

    threshold = st.slider(
        "分类阈值",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("prediction_threshold", 0.5)),
        step=0.05,
        key="ai_prediction_threshold",
    )

    positive = int((predictions >= threshold).sum())
    high_conf_pos = int((predictions >= 0.8).sum())
    high_conf_neg = int((predictions <= 0.2).sum())
    low_conf = int(((predictions >= 0.4) & (predictions <= 0.6)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("来源", source)
    c2.metric("样本数", f"{len(predictions):,}")
    c3.metric("预测正例", f"{positive:,} ({positive / len(predictions):.1%})")
    c4.metric("平均概率", f"{predictions.mean():.4f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("高置信正例", high_conf_pos)
    c6.metric("低置信区间", low_conf)
    c7.metric("高置信负例", high_conf_neg)

    hist, edges = np.histogram(predictions.clip(0, 1), bins=20, range=(0, 1))
    hist_df = pd.DataFrame({
        "概率区间": [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(hist))],
        "样本数": hist,
    }).set_index("概率区间")
    st.bar_chart(hist_df, use_container_width=True)

    if st.button("分析预测结果", type="primary", use_container_width=True, key="ai_pred_analyze_btn"):
        result = _run_llm_task(
            "分析预测结果",
            lambda: analyzer.analyze_prediction_results(predictions, threshold=threshold),
        )
        if result:
            st.session_state.prediction_analysis = result

    if st.session_state.get("prediction_analysis"):
        st.markdown("---")
        st.subheader("预测结果分析")
        st.markdown(st.session_state.prediction_analysis)


def main() -> None:
    st.title("🧠 AI 智能分析")
    st.caption("读取当前数据、训练结果和预测结果，调用大模型给出可执行的建模建议。")

    analyzer = init_analyzer()
    active = _active_llm()
    if active:
        st.success(f"当前大模型：{active.name}")
    else:
        st.warning("当前没有激活的大模型。页面可查看上下文，但分析按钮需要可用 LLM。")

    tabs = st.tabs(["数据集分析", "训练结果分析", "超参数调优", "预测结果分析"])
    with tabs[0]:
        render_data_analysis_tab(analyzer)
    with tabs[1]:
        render_training_analysis_tab(analyzer)
    with tabs[2]:
        render_hyperparameter_tab(analyzer)
    with tabs[3]:
        render_prediction_analysis_tab(analyzer)

    with st.sidebar:
        st.header("上下文状态")
        st.write(f"数据：{'已加载' if _get_uploaded_data() is not None else '未加载'}")
        st.write(f"训练结果：{'已读取' if st.session_state.get('training_results') is not None else '暂无'}")
        st.write(f"预测结果：{'已读取' if st.session_state.get('predict_results') is not None else '暂无'}")
        st.markdown("---")
        st.header("说明")
        st.markdown(
            """
            - 数据集分析读取当前上传数据和字段配置。
            - 训练分析优先读取手动训练/AutoML 结果。
            - 调参建议会复用最近一次随机搜索记录。
            - 预测分析优先读取预测页结果。
            """
        )


if __name__ == "__main__":
    main()
