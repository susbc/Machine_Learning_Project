"""
Agentic AutoML 训练页。

目标：复用手动训练的完整链路，但把字段配置、EDA、算法选择、LLM 调参、
参数搜索、模型评估与保存全部自动化。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web_app.utils.session_state import init_session_state


st.set_page_config(
    page_title="AutoML 训练 - v3.0.0",
    page_icon="📊",
    layout="wide",
)

init_session_state()


ALGORITHM_ALIASES = {
    "lightgbm": ["lightgbm", "lgbm", "lgb", "轻量梯度提升", "光梯度"],
    "xgboost": ["xgboost", "xgb"],
    "catboost": ["catboost", "cat"],
    "random_forest": ["random forest", "random_forest", "随机森林"],
    "logistic_regression": ["logistic regression", "logistic_regression", "逻辑回归", "lr"],
}


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def _detect_target_col(df: pd.DataFrame) -> str:
    """优先沿用配置中的 label，否则用常见命名，最后回退到最后一列。"""
    config_df = st.session_state.get("field_config")
    if config_df is not None and "field_type" in config_df.columns:
        label_rows = config_df[config_df["field_type"] == "label"]
        if len(label_rows) > 0:
            candidate = label_rows.iloc[0]["field_name"]
            if candidate in df.columns:
                return candidate

    label_names = {"label", "target", "y", "class", "is_fraud", "是否违约", "标签", "目标"}
    for col in df.columns:
        if str(col).strip().lower() in label_names:
            return col

    return df.columns[-1]


def _detect_key_col(df: pd.DataFrame, target_col: str) -> Optional[str]:
    patterns = ("id", "_id", "uuid", "key", "编号", "序号", "订单号", "用户号")
    for col in df.columns:
        if col == target_col:
            continue
        name = str(col).strip().lower()
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.85 and any(p in name for p in patterns):
            return col
    return None


def _build_auto_config(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    from web_app.components.config_editor import auto_infer_config

    config = auto_infer_config(df).copy()
    key_col = _detect_key_col(df, target_col)

    for idx, row in config.iterrows():
        field = row["field_name"]
        if field == target_col:
            config.loc[idx, ["field_type", "outlier_value", "missing_value", "ecode_type", "standardize_type"]] = [
                "label",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        elif key_col and field == key_col:
            config.loc[idx, ["field_type", "outlier_value", "missing_value", "ecode_type", "standardize_type"]] = [
                "key",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        elif row["field_type"] == "str":
            # ConfigDrivenPreprocessor 当前对 one-hot 没有稳定 transform 状态，
            # 自动流程使用 ordinal，保证训练/预测一致。
            config.loc[idx, "ecode_type"] = "ordinal"

    return config


def _build_eda_summary(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    feature_df = df.drop(columns=[target_col], errors="ignore")

    high_corr_pairs: List[Dict[str, Any]] = []
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(value) >= 0.75:
                    high_corr_pairs.append({
                        "feature_1": left,
                        "feature_2": right,
                        "corr": round(float(value), 4),
                    })

    outlier_rates: Dict[str, float] = {}
    for col in numeric_df.columns:
        if col == target_col:
            continue
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_rates[col] = round(float(((numeric_df[col] < lower) | (numeric_df[col] > upper)).mean()), 4)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "target": target_col,
        "n_features": max(len(df.columns) - 1, 0),
        "n_numeric": len(feature_df.select_dtypes(include=[np.number]).columns),
        "n_categorical": len(feature_df.select_dtypes(include=["object", "category"]).columns),
        "missing_rate": round(float(feature_df.isna().sum().sum()) / max(feature_df.size, 1), 4),
        "target_unique": int(df[target_col].nunique(dropna=True)),
        "high_corr_pairs": high_corr_pairs[:12],
        "outlier_rates": dict(sorted(outlier_rates.items(), key=lambda x: x[1], reverse=True)[:12]),
    }


def _llm_available() -> bool:
    try:
        from model_code.llm.llm_connector import get_llm_manager

        manager = get_llm_manager()
        return bool(manager.providers and manager.get_active_provider())
    except Exception:
        return False


def _parse_algorithms_from_text(text: str, supported: List[str]) -> List[str]:
    lowered = text.lower()
    ranked: List[str] = []
    for algo, aliases in ALGORITHM_ALIASES.items():
        if algo not in supported:
            continue
        if any(alias.lower() in lowered for alias in aliases):
            ranked.append(algo)
    return ranked


def _recommend_algorithms_with_llm(
    df: pd.DataFrame,
    target_col: str,
    heuristic_candidates: List[str],
    business_context: str,
    use_llm: bool,
) -> Tuple[List[str], str]:
    if not use_llm:
        return heuristic_candidates, "已关闭 LLM，使用启发式算法推荐。"

    if not _llm_available():
        return heuristic_candidates, "未检测到可用 LLM，使用启发式算法推荐。"

    try:
        from model_code.llm.intelligent_analyzer import get_intelligent_analyzer
        from web_app.manual_training.param_specs import get_supported_algorithms

        raw = get_intelligent_analyzer().recommend_model(
            df=df,
            target_col=target_col,
            business_context=business_context,
            requirements={
                "speed_requirement": "中等",
                "interpretability_requirement": "中等",
                "accuracy_requirement": "高",
            },
        )
        parsed = _parse_algorithms_from_text(raw, get_supported_algorithms())
        if parsed:
            merged = parsed + [a for a in heuristic_candidates if a not in parsed]
            return merged, raw
        return heuristic_candidates, raw
    except Exception as e:
        return heuristic_candidates, f"LLM 算法推荐失败，已回退启发式推荐：{e}"


def _recommend_params(
    df: pd.DataFrame,
    target_col: str,
    algorithm: str,
    eda_summary: Dict[str, Any],
    use_llm: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not use_llm or not _llm_available():
        return {}, {"source": "default", "message": "未使用 LLM 参数推荐。"}

    try:
        from web_app.manual_training.llm_tuning import recommend_params

        result = recommend_params(
            df=df,
            target_col=target_col,
            model_type=algorithm,
            eda_summary=eda_summary,
            last_training=st.session_state.get("training_results"),
        )
        return result["params"], {"source": "llm", **result}
    except Exception as e:
        return {}, {"source": "fallback", "message": f"LLM 参数推荐失败，已使用默认搜索空间：{e}"}


def _default_params_from_specs(algorithm: str) -> Dict[str, Any]:
    from web_app.manual_training.param_specs import get_specs

    return {spec.name: spec.default for spec in get_specs(algorithm)}


def _runtime_params(algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(params)
    if algorithm in {"lightgbm", "xgboost", "random_forest", "logistic_regression"}:
        runtime["n_jobs"] = 1
    if algorithm == "catboost":
        runtime["thread_count"] = 1
        runtime["verbose"] = False
    return runtime


def _values_around(value: Any, min_value: Any, max_value: Any, step: Any, is_int: bool) -> List[Any]:
    if value is None:
        return []
    if is_int:
        center = int(value)
        step_value = max(int(step or 1), 1)
        radius = max(step_value, int(abs(center) * 0.35))
        low = max(int(min_value), center - radius)
        high = min(int(max_value), center + radius)
        values = list(range(low, high + 1, step_value))
        return sorted(set(values or [center]))

    center = float(value)
    low = max(float(min_value), center * 0.65 if center > 0 else float(min_value))
    high = min(float(max_value), center * 1.35 if center > 0 else float(max_value))
    return [round(float(v), 6) for v in np.linspace(low, high, 7)]


def _build_param_distribution(algorithm: str, llm_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    from web_app.manual_training.param_specs import get_specs

    distribution: Dict[str, List[Any]] = {}
    defaults = _default_params_from_specs(algorithm)

    for spec in get_specs(algorithm):
        center = llm_params.get(spec.name, defaults.get(spec.name))
        if spec.widget == "select":
            options = list(spec.options or [])
            if center in options:
                distribution[spec.name] = [center] + [v for v in options if v != center]
            elif options:
                distribution[spec.name] = options
            continue

        if spec.min_value is None or spec.max_value is None:
            distribution[spec.name] = [center]
            continue

        distribution[spec.name] = _values_around(
            center,
            spec.min_value,
            spec.max_value,
            spec.step,
            isinstance(spec.default, int) and not isinstance(spec.default, bool),
        )

    return {k: v for k, v in distribution.items() if v}


def _prepare_data(
    df: pd.DataFrame,
    config_df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from model_code.utils import ConfigDrivenPreprocessor

    key_rows = config_df[config_df["field_type"] == "key"]
    key_col = key_rows.iloc[0]["field_name"] if len(key_rows) > 0 else None
    feature_cols = [c for c in df.columns if c != target_col and c != key_col]

    if len(feature_cols) == 0:
        raise ValueError("未找到可训练的特征列")

    y_raw = df[target_col]
    if y_raw.isna().any():
        keep_mask = y_raw.notna()
        df = df.loc[keep_mask].reset_index(drop=True)
        y_raw = df[target_col]

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), name=target_col)
    n_classes = int(y.nunique())
    if n_classes < 2:
        raise ValueError("目标变量至少需要 2 个类别")

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    X_train_raw = df.loc[train_idx, feature_cols]
    X_test_raw = df.loc[test_idx, feature_cols]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    preprocessor = ConfigDrivenPreprocessor(config_df)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 模型只接受数值矩阵；兜底把残留 object 转为数值。
    for frame in (X_train, X_test):
        for col in frame.columns:
            if not np.issubdtype(frame[col].dtype, np.number):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.fillna(0, inplace=True)

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder, n_classes


def _metric_bundle(model, X_test, y_test, n_classes: int) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "y_pred": y_pred,
        "y_pred_proba": None,
        "best_threshold": 0.5,
        "best_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    if proba is not None and n_classes == 2:
        y_score = proba[:, 1]
        metrics["y_pred_proba"] = y_score
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        except Exception:
            metrics["roc_auc"] = metrics["accuracy"]

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        if len(thresholds) > 0:
            best_idx = int(np.argmax(f1_scores[:-1]))
            metrics["best_threshold"] = float(thresholds[best_idx])
            metrics["best_f1"] = float(f1_scores[best_idx])
            metrics["y_pred"] = (y_score >= metrics["best_threshold"]).astype(int)
    elif proba is not None:
        metrics["y_pred_proba"] = proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = metrics["accuracy"]
    else:
        metrics["roc_auc"] = metrics["accuracy"]

    return metrics


def _feature_importance(model, feature_names: List[str]) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        values = np.zeros(len(feature_names), dtype=float)

    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))
    return values


def _train_candidate(
    algorithm: str,
    X_train,
    X_test,
    y_train,
    y_test,
    n_classes: int,
    cv_folds: int,
    n_iter: int,
    random_state: int,
    llm_params: Dict[str, Any],
) -> Dict[str, Any]:
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from model_code.model_factory import ModelFactory

    task_type = "binary" if n_classes == 2 else "multiclass"
    base_params = _runtime_params(algorithm, {})
    if task_type == "multiclass":
        if algorithm == "lightgbm":
            base_params["num_classes"] = n_classes
        elif algorithm == "xgboost":
            base_params["num_class"] = n_classes
        elif algorithm == "catboost":
            base_params["classes_count"] = n_classes

    base_model = ModelFactory.build(algorithm, task_type=task_type, params=base_params)
    class_min = int(pd.Series(y_train).value_counts().min())
    scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    param_distribution = _build_param_distribution(algorithm, llm_params)

    started = time.time()
    if n_iter > 1 and param_distribution and class_min >= 2:
        safe_cv = max(2, min(cv_folds, class_min))
        cv = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distribution,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            random_state=random_state,
            error_score=0.0,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = dict(search.best_params_)
        cv_score = float(search.best_score_)
        cv_results = search.cv_results_
    else:
        fixed_params = _runtime_params(algorithm, {**_default_params_from_specs(algorithm), **llm_params})
        if task_type == "multiclass":
            fixed_params.update({k: v for k, v in base_params.items() if k in {"num_classes", "num_class", "classes_count"}})
        model = ModelFactory.build(algorithm, task_type=task_type, params=fixed_params)
        model.fit(X_train, y_train)
        best_params = fixed_params
        cv_score = np.nan
        cv_results = None

    metrics = _metric_bundle(model, X_test, y_test, n_classes)
    return {
        "algorithm": algorithm,
        "model": model,
        "best_params": best_params,
        "cv_score": cv_score,
        "metrics": metrics,
        "fit_seconds": round(time.time() - started, 2),
        "cv_results": cv_results,
    }


def _display_results(results: Dict[str, Any]) -> None:
    from sklearn.metrics import classification_report
    from model_code.model_factory import ModelFactory
    from web_app.components.charts import (
        render_confusion_matrix,
        render_feature_importance,
        render_precision_recall_curve,
        render_roc_curve,
    )

    st.markdown("---")
    st.subheader("📊 AutoML 结果")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最佳模型", ModelFactory.get_display_name(results["algorithm"]))
    col2.metric("ROC-AUC", f"{results['test_auc']:.4f}")
    col3.metric("最佳 F1", f"{results['best_f1']:.4f}")
    col4.metric("最佳阈值", f"{results['best_threshold']:.4f}")

    leaderboard = st.session_state.get("automl_leaderboard")
    if leaderboard:
        st.markdown("**🏁 候选模型排行**")
        st.dataframe(pd.DataFrame(leaderboard), use_container_width=True, hide_index=True)

    st.markdown("**🧠 自动决策摘要**")
    st.json({
        "algorithm_reason": st.session_state.get("automl_algorithm_reason", ""),
        "llm_param_notes": st.session_state.get("automl_param_notes", {}),
        "best_params": results.get("model_params", {}),
    })

    tabs = st.tabs(["ROC 曲线", "PR 曲线", "混淆矩阵", "特征重要性", "分类报告"])
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_score = results["y_pred_proba"]

    with tabs[0]:
        if np.ndim(y_score) == 1:
            st.plotly_chart(render_roc_curve(y_test, y_score), use_container_width=True)
        else:
            st.info("多分类任务暂不绘制二分类 ROC 曲线，已在指标中使用 OVR ROC-AUC。")
    with tabs[1]:
        if np.ndim(y_score) == 1:
            st.plotly_chart(render_precision_recall_curve(y_test, y_score), use_container_width=True)
        else:
            st.info("多分类任务暂不绘制二分类 PR 曲线。")
    with tabs[2]:
        st.plotly_chart(render_confusion_matrix(y_test, y_pred), use_container_width=True)
    with tabs[3]:
        st.plotly_chart(
            render_feature_importance(results["feature_names"], results["feature_importances"]),
            use_container_width=True,
        )
    with tabs[4]:
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    st.success(f"✅ 模型已保存：{results['model_path']}")
    st.caption(f"训练摘要：{results['summary_path']}")


def _run_automl(
    df: pd.DataFrame,
    target_col: str,
    max_candidates: int,
    n_iter: int,
    cv_folds: int,
    test_size: float,
    random_state: int,
    use_llm: bool,
    business_context: str,
) -> Dict[str, Any]:
    from model_code.auto_selector import AutoModelSelector
    from model_code.model_factory import ModelFactory

    logs: List[str] = []

    def log(message: str) -> None:
        logs.append(message)
        st.session_state.automl_logs = logs.copy()

    config_df = _build_auto_config(df, target_col)
    st.session_state.field_config = config_df
    st.session_state.uploaded_data = df

    eda_summary = _build_eda_summary(df, target_col)
    st.session_state.eda_summary_cache = eda_summary

    key_rows = config_df[config_df["field_type"] == "key"]
    key_col = key_rows.iloc[0]["field_name"] if len(key_rows) > 0 else None
    feature_df = df[[c for c in df.columns if c not in [target_col, key_col]]]

    log("1/6 自动字段配置完成")
    heuristic = AutoModelSelector.recommend(feature_df, df[target_col])
    heuristic_candidates = heuristic["candidates"]
    algorithm_candidates, algorithm_reason = _recommend_algorithms_with_llm(
        df,
        target_col,
        heuristic_candidates,
        business_context,
        use_llm,
    )
    algorithm_candidates = algorithm_candidates[:max_candidates]
    st.session_state.automl_algorithm_reason = algorithm_reason
    log(f"2/6 候选模型：{', '.join(ModelFactory.get_display_name(a) for a in algorithm_candidates)}")

    X_train, X_test, y_train, y_test, preprocessor, label_encoder, n_classes = _prepare_data(
        df,
        config_df,
        target_col,
        test_size,
        random_state,
    )
    log(f"3/6 预处理完成：训练集 {X_train.shape[0]} 行，测试集 {X_test.shape[0]} 行")

    trial_results = []
    param_notes: Dict[str, Any] = {}
    progress = st.progress(0)
    status = st.empty()
    log_box = st.empty()

    for i, algorithm in enumerate(algorithm_candidates, start=1):
        status.info(f"正在训练 {ModelFactory.get_display_name(algorithm)} ({i}/{len(algorithm_candidates)})")
        llm_params, note = _recommend_params(df, target_col, algorithm, eda_summary, use_llm)
        param_notes[algorithm] = note
        if llm_params:
            log(f"4/6 LLM 为 {ModelFactory.get_display_name(algorithm)} 推荐参数：{llm_params}")
        else:
            log(f"4/6 {ModelFactory.get_display_name(algorithm)} 使用默认搜索空间")

        try:
            result = _train_candidate(
                algorithm=algorithm,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                n_classes=n_classes,
                cv_folds=cv_folds,
                n_iter=n_iter,
                random_state=random_state,
                llm_params=llm_params,
            )
            trial_results.append(result)
            score = result["metrics"]["roc_auc"]
            log(f"5/6 {ModelFactory.get_display_name(algorithm)} 完成，测试 ROC-AUC={score:.4f}")
        except Exception as e:
            log(f"{ModelFactory.get_display_name(algorithm)} 训练失败：{type(e).__name__}: {e}")

        progress.progress(i / len(algorithm_candidates))
        log_box.code("\n".join(logs[-18:]), language="text")

    if not trial_results:
        raise RuntimeError("所有候选模型训练失败，请检查数据和依赖。")

    best = max(trial_results, key=lambda item: item["metrics"]["roc_auc"])
    best_model = best["model"]
    metrics = best["metrics"]
    feature_names = list(X_train.columns)
    importances = _feature_importance(best_model, feature_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"automl_{best['algorithm']}_{timestamp}"
    model_path = st.session_state.model_dir / f"model_{model_name}.joblib"
    summary_path = st.session_state.model_dir / f"train_summary_{model_name}.joblib"

    training_results = {
        "model": best_model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "field_config": config_df,
        "algorithm": best["algorithm"],
        "model_params": best["best_params"],
        "test_auc": metrics["roc_auc"],
        "best_threshold": metrics["best_threshold"],
        "best_f1": metrics["best_f1"],
        "feature_names": feature_names,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test.values,
        "y_pred_proba": metrics["y_pred_proba"],
        "y_pred": metrics["y_pred"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": datetime.now().isoformat(),
        "automl": True,
        "eda_summary": eda_summary,
        "model_path": str(model_path),
        "summary_path": str(summary_path),
    }

    joblib.dump(best_model, model_path)
    joblib.dump(training_results, summary_path)

    leaderboard = []
    for item in sorted(trial_results, key=lambda x: x["metrics"]["roc_auc"], reverse=True):
        leaderboard.append({
            "模型": ModelFactory.get_display_name(item["algorithm"]),
            "测试 ROC-AUC": round(item["metrics"]["roc_auc"], 5),
            "测试 F1": round(item["metrics"]["best_f1"], 5),
            "CV 分数": None if pd.isna(item["cv_score"]) else round(item["cv_score"], 5),
            "耗时(秒)": item["fit_seconds"],
            "最佳参数": json.dumps(item["best_params"], ensure_ascii=False),
        })

    st.session_state.training_results = training_results
    st.session_state.automl_leaderboard = leaderboard
    st.session_state.automl_param_notes = param_notes
    st.session_state.automl_logs = logs

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = []
    st.session_state.trained_models.append({
        "name": model_path.stem,
        "path": str(model_path),
        "auc": metrics["roc_auc"],
        "f1": metrics["best_f1"],
        "timestamp": datetime.now().isoformat(),
    })

    log(f"6/6 完成，最佳模型：{ModelFactory.get_display_name(best['algorithm'])}")
    progress.progress(1.0)
    status.success("AutoML 全流程完成")
    return training_results


st.title("📊 AutoML 训练")
st.caption("自动执行：数据读取 → 字段配置 → EDA → LLM 分析 → 模型选择 → 参数搜索 → 评估保存")

with st.sidebar:
    st.header("自动训练预算")
    max_candidates = st.slider("候选模型数量", 1, 5, 3)
    n_iter = st.slider("每个模型搜索次数", 1, 30, 8)
    cv_folds = st.slider("交叉验证折数", 2, 10, 3)
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.number_input("随机种子", min_value=0, max_value=10000, value=42, step=1)
    use_llm = st.checkbox("调用 LLM 分析与调参", value=True)
    st.info("未配置 LLM 时会自动回退到启发式推荐和默认搜索空间。")

st.header("1. 数据")
current_df = st.session_state.get("uploaded_data")
uploaded_file = st.file_uploader("上传 CSV/Excel 文件", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        current_df = _read_uploaded_file(uploaded_file)
        st.session_state.uploaded_data = current_df
        st.success(f"数据加载成功：{current_df.shape[0]:,} 行 × {current_df.shape[1]} 列")
    except Exception as e:
        st.error(f"数据读取失败：{e}")
        current_df = None

if current_df is None:
    st.info("请先上传数据，或在「手动训练流程 / 数据准备」中加载数据。")
    st.stop()

with st.expander("数据预览", expanded=True):
    st.dataframe(current_df.head(30), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", f"{len(current_df):,}")
    c2.metric("列数", current_df.shape[1])
    c3.metric("缺失率", f"{current_df.isna().sum().sum() / max(current_df.size, 1):.2%}")
    c4.metric("内存", f"{current_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

st.header("2. 自动目标识别")
detected_target = _detect_target_col(current_df)
target_col = st.selectbox(
    "目标变量列",
    options=current_df.columns.tolist(),
    index=current_df.columns.tolist().index(detected_target),
    help="AutoML 会自动完成其余配置；这里仅用于确认监督学习目标。",
)

business_context = st.text_area(
    "业务背景（可选，提供给 LLM 做模型/参数推荐）",
    placeholder="例如：希望兼顾召回率与稳定性，模型用于金融风险二分类。",
    height=90,
)

auto_config = _build_auto_config(current_df, target_col)
eda_summary = _build_eda_summary(current_df, target_col)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("自动字段配置")
    st.dataframe(auto_config, use_container_width=True, hide_index=True)
with col_b:
    st.subheader("EDA 摘要")
    st.json(eda_summary)

st.header("3. 一键全自动训练")
run_clicked = st.button(
    "🚀 开始全自动 AutoML",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.get("automl_running", False),
)

if run_clicked:
    st.session_state.automl_running = True
    st.session_state.automl_logs = []
    if st.session_state.get("training_results") is not None and st.session_state.training_results.get("automl"):
        st.session_state.training_results = None
    try:
        _run_automl(
            df=current_df.copy(),
            target_col=target_col,
            max_candidates=max_candidates,
            n_iter=n_iter,
            cv_folds=cv_folds,
            test_size=test_size,
            random_state=int(random_state),
            use_llm=use_llm,
            business_context=business_context,
        )
        st.session_state.automl_running = False
    except Exception as e:
        st.session_state.automl_running = False
        st.error(f"AutoML 训练失败：{type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="text")

if st.session_state.get("automl_logs"):
    with st.expander("训练日志", expanded=False):
        st.code("\n".join(st.session_state.automl_logs), language="text")

if st.session_state.get("training_results") is not None and st.session_state.training_results.get("automl"):
    _display_results(st.session_state.training_results)

st.markdown("---")
st.caption("AutoML 结果会写入手动训练同款 training_results，并保存到 model_code/saved，可继续在模型管理和预测页面使用。")
