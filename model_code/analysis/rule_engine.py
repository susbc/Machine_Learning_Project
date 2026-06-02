"""Deterministic diagnostics for AI analysis.

The rules here are intentionally conservative. They provide evidence for the
LLM layer and keep the UI useful even when no LLM provider is active.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schemas import (
    AnalysisResult,
    EvidenceItem,
    RecommendationItem,
    RiskItem,
    max_risk_level,
    safe_float,
)


LEAKAGE_TOKENS = (
    "target",
    "label",
    "result",
    "outcome",
    "future",
    "after",
    "post",
    "标签",
    "目标",
    "结果",
    "未来",
    "贷后",
)


def _ratio(value: float, total: float) -> float:
    return float(value) / max(float(total), 1.0)


def _risk(
    risks: List[RiskItem],
    title: str,
    severity: str,
    reason: str,
    recommendation: str,
    evidence: Optional[List[str]] = None,
) -> None:
    risks.append(
        RiskItem(
            title=title,
            severity=severity,
            reason=reason,
            recommendation=recommendation,
            evidence=evidence or [],
        )
    )


def diagnose_dataset(df: pd.DataFrame, target_col: str) -> AnalysisResult:
    """Diagnose data quality and modeling risks."""
    evidence: List[EvidenceItem] = []
    risks: List[RiskItem] = []
    recommendations: List[RecommendationItem] = []
    next_actions: List[str] = []

    if df is None or df.empty:
        _risk(
            risks,
            "empty_dataset",
            "critical",
            "数据集为空，无法进行建模分析。",
            "重新加载数据后再执行 AI 智能分析。",
        )
        return AnalysisResult(
            analysis_type="dataset_rules",
            summary="数据集为空，无法诊断。",
            risk_level="critical",
            confidence=1.0,
            risks=risks,
            next_actions=["加载有效数据集"],
        )

    rows, columns = df.shape
    feature_df = df.drop(columns=[target_col], errors="ignore")
    missing_rate = float(df.isna().sum().sum()) / max(df.size, 1)
    duplicate_rate = float(df.duplicated().mean()) if rows else 0.0
    evidence.extend(
        [
            EvidenceItem("rows", rows, "样本行数"),
            EvidenceItem("columns", columns, "总列数"),
            EvidenceItem("feature_count", feature_df.shape[1], "不含目标列的特征数"),
            EvidenceItem("missing_rate", round(missing_rate, 6), "全表缺失率"),
            EvidenceItem("duplicate_rate", round(duplicate_rate, 6), "重复行比例"),
        ]
    )

    if target_col not in df.columns:
        _risk(
            risks,
            "missing_target",
            "critical",
            f"目标列 {target_col} 不存在。",
            "重新选择目标列或修正字段配置。",
        )
        return AnalysisResult(
            analysis_type="dataset_rules",
            summary=f"目标列 {target_col} 不存在，无法继续诊断。",
            risk_level="critical",
            confidence=1.0,
            evidence=evidence,
            risks=risks,
            next_actions=["修正目标列配置"],
        )

    if missing_rate >= 0.5:
        _risk(
            risks,
            "high_missing_rate",
            "high",
            f"整体缺失率达到 {missing_rate:.1%}。",
            "先做缺失模式分析，删除严重缺失字段或采用分组填充策略。",
            ["missing_rate"],
        )
    elif missing_rate >= 0.2:
        _risk(
            risks,
            "moderate_missing_rate",
            "medium",
            f"整体缺失率达到 {missing_rate:.1%}。",
            "检查缺失是否和目标相关，避免简单均值填充引入偏差。",
            ["missing_rate"],
        )

    if duplicate_rate >= 0.2:
        _risk(
            risks,
            "high_duplicate_rate",
            "high",
            f"重复行比例达到 {duplicate_rate:.1%}。",
            "训练前去重，或确认重复样本是否代表真实权重。",
            ["duplicate_rate"],
        )
    elif duplicate_rate >= 0.05:
        _risk(
            risks,
            "moderate_duplicate_rate",
            "medium",
            f"重复行比例达到 {duplicate_rate:.1%}。",
            "抽样核查重复记录来源。",
            ["duplicate_rate"],
        )

    constant_cols = [
        col for col in feature_df.columns
        if feature_df[col].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        evidence.append(EvidenceItem("constant_columns", constant_cols[:20], "常量或近似无信息字段"))
        _risk(
            risks,
            "constant_features",
            "medium",
            f"发现 {len(constant_cols)} 个常量字段。",
            "训练前删除常量字段，减少噪声和搜索空间。",
            ["constant_columns"],
        )
        next_actions.append("删除常量字段")

    high_cardinality = []
    for col in feature_df.select_dtypes(include=["object", "category"]).columns:
        unique_count = int(feature_df[col].nunique(dropna=True))
        unique_ratio = _ratio(unique_count, rows)
        if unique_count >= 50 and unique_ratio >= 0.2:
            high_cardinality.append({"field": col, "unique": unique_count, "ratio": round(unique_ratio, 4)})
    if high_cardinality:
        evidence.append(EvidenceItem("high_cardinality_columns", high_cardinality[:20], "高基数类别字段"))
        _risk(
            risks,
            "high_cardinality_features",
            "medium",
            f"发现 {len(high_cardinality)} 个高基数类别字段。",
            "优先使用目标编码/频率编码/聚合特征，避免直接 one-hot 造成维度爆炸。",
            ["high_cardinality_columns"],
        )

    suspicious_cols = []
    target_name = str(target_col).lower()
    for col in feature_df.columns:
        name = str(col).lower()
        if name == target_name:
            continue
        if any(token in name for token in LEAKAGE_TOKENS):
            suspicious_cols.append(col)
    if suspicious_cols:
        evidence.append(EvidenceItem("possible_leakage_columns", suspicious_cols[:20], "字段名疑似包含目标/未来信息"))
        _risk(
            risks,
            "possible_target_leakage",
            "high",
            f"发现 {len(suspicious_cols)} 个字段名疑似包含目标或未来信息。",
            "人工确认这些字段是否在预测时可获得，必要时从训练特征中剔除。",
            ["possible_leakage_columns"],
        )
        next_actions.append("人工核查目标泄漏字段")

    target_unique = int(df[target_col].nunique(dropna=True))
    evidence.append(EvidenceItem("target_unique", target_unique, "目标列唯一值数量"))
    if target_unique < 2:
        _risk(
            risks,
            "single_class_target",
            "critical",
            "目标列少于 2 个类别。",
            "重新选择目标列或补充包含多类别的训练数据。",
            ["target_unique"],
        )
    elif target_unique == 2:
        counts = df[target_col].value_counts(dropna=True)
        minority_ratio = float(counts.min() / counts.sum())
        evidence.append(EvidenceItem("minority_class_ratio", round(minority_ratio, 6), "二分类少数类占比"))
        if minority_ratio < 0.01:
            _risk(
                risks,
                "severe_class_imbalance",
                "high",
                f"少数类占比仅 {minority_ratio:.2%}。",
                "使用分层抽样、类别权重、PR-AUC 和召回率作为重点评估指标。",
                ["minority_class_ratio"],
            )
        elif minority_ratio < 0.05:
            _risk(
                risks,
                "class_imbalance",
                "medium",
                f"少数类占比为 {minority_ratio:.2%}。",
                "训练和验证必须分层，并重点关注 PR 曲线和阈值选择。",
                ["minority_class_ratio"],
            )
    elif pd.api.types.is_numeric_dtype(df[target_col]) and target_unique > 20:
        _risk(
            risks,
            "continuous_like_target",
            "medium",
            f"目标列有 {target_unique} 个唯一值，可能是连续目标。",
            "确认当前训练链路是否应支持回归，或先将目标离散化。",
            ["target_unique"],
        )

    if not risks:
        recommendations.append(
            RecommendationItem(
                title="data_ready_for_baseline",
                action="数据质量未触发高风险规则，可以先训练基线模型并观察验证指标。",
                priority="medium",
                reason="缺失、重复、目标分布未出现明显异常。",
            )
        )
        next_actions.append("训练基线模型")

    risk_level = max_risk_level(risks)
    summary = f"数据集包含 {rows:,} 行、{feature_df.shape[1]} 个特征，规则诊断风险等级为 {risk_level}。"
    return AnalysisResult(
        analysis_type="dataset_rules",
        summary=summary,
        risk_level=risk_level,
        confidence=0.95,
        evidence=evidence,
        risks=risks,
        recommendations=recommendations,
        next_actions=next_actions,
        metadata={"target_col": target_col},
    )


def diagnose_training(
    metrics: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[Sequence[Tuple[str, float]]] = None,
) -> AnalysisResult:
    """Diagnose training metrics and model configuration."""
    metrics = metrics or {}
    params = params or {}
    evidence: List[EvidenceItem] = []
    risks: List[RiskItem] = []
    recommendations: List[RecommendationItem] = []
    next_actions: List[str] = []

    train_auc = _as_optional_float(metrics.get("train_auc"))
    val_auc = _as_optional_float(metrics.get("val_auc", metrics.get("test_auc")))
    train_f1 = _as_optional_float(metrics.get("train_f1"))
    val_f1 = _as_optional_float(metrics.get("val_f1", metrics.get("best_f1")))
    train_size = int(safe_float(metrics.get("train_size"), 0.0))
    val_size = int(safe_float(metrics.get("val_size", metrics.get("test_size")), 0.0))
    n_features = int(safe_float(metrics.get("n_features"), 0.0))
    best_threshold = _as_optional_float(metrics.get("best_threshold"))

    evidence.extend(
        [
            EvidenceItem("train_auc", _none_or_round(train_auc), "训练集 AUC"),
            EvidenceItem("val_auc", _none_or_round(val_auc), "验证/测试集 AUC"),
            EvidenceItem("train_f1", _none_or_round(train_f1), "训练集 F1"),
            EvidenceItem("val_f1", _none_or_round(val_f1), "验证/测试集 F1"),
            EvidenceItem("train_size", train_size, "训练样本数"),
            EvidenceItem("val_size", val_size, "验证/测试样本数"),
            EvidenceItem("n_features", n_features, "特征数"),
        ]
    )

    if val_auc is not None:
        if val_auc < 0.6:
            _risk(
                risks,
                "weak_validation_auc",
                "high",
                f"验证/测试 AUC 仅 {val_auc:.4f}。",
                "优先检查目标定义、特征有效性和数据泄漏，再扩大模型搜索。",
                ["val_auc"],
            )
        elif val_auc < 0.7:
            _risk(
                risks,
                "moderate_validation_auc",
                "medium",
                f"验证/测试 AUC 为 {val_auc:.4f}。",
                "补充特征工程、类别不平衡处理，并尝试树模型参数搜索。",
                ["val_auc"],
            )

    if val_f1 is not None and val_f1 < 0.45:
        _risk(
            risks,
            "weak_validation_f1",
            "medium",
            f"验证/测试 F1 为 {val_f1:.4f}。",
            "重新搜索阈值，并检查正负样本分布和业务损失函数。",
            ["val_f1"],
        )

    if train_auc is not None and val_auc is not None:
        auc_gap = train_auc - val_auc
        evidence.append(EvidenceItem("auc_gap", round(auc_gap, 6), "训练 AUC - 验证 AUC"))
        if auc_gap >= 0.15:
            _risk(
                risks,
                "strong_overfit_signal",
                "high",
                f"训练 AUC 比验证/测试 AUC 高 {auc_gap:.4f}。",
                "降低模型复杂度、加强正则、增加验证集稳定性，并检查泄漏字段。",
                ["auc_gap"],
            )
        elif auc_gap >= 0.08:
            _risk(
                risks,
                "overfit_signal",
                "medium",
                f"训练 AUC 比验证/测试 AUC 高 {auc_gap:.4f}。",
                "优先调小树深/叶子数，增大 min_child_samples 或正则项。",
                ["auc_gap"],
            )

    if n_features and train_size:
        feature_sample_ratio = n_features / max(train_size, 1)
        evidence.append(EvidenceItem("feature_sample_ratio", round(feature_sample_ratio, 6), "特征数/训练样本数"))
        if feature_sample_ratio >= 0.2:
            _risk(
                risks,
                "high_dimensional_small_sample",
                "medium",
                f"特征数/训练样本数比例为 {feature_sample_ratio:.2%}。",
                "增加样本、做特征选择，或优先使用强正则模型。",
                ["feature_sample_ratio"],
            )

    if best_threshold is not None:
        evidence.append(EvidenceItem("best_threshold", round(best_threshold, 6), "最佳阈值"))
        if best_threshold <= 0.15 or best_threshold >= 0.85:
            _risk(
                risks,
                "extreme_threshold",
                "medium",
                f"最佳阈值为 {best_threshold:.4f}，明显偏离 0.5。",
                "检查概率校准、类别比例和业务阈值约束。",
                ["best_threshold"],
            )

    if feature_importance:
        top_feature, top_value = feature_importance[0]
        total_importance = sum(abs(float(v)) for _, v in feature_importance) or 1.0
        top_share = abs(float(top_value)) / total_importance
        evidence.append(EvidenceItem("top_feature_share", round(top_share, 6), f"Top 特征 {top_feature} 重要性占比"))
        if top_share >= 0.7:
            _risk(
                risks,
                "dominant_feature",
                "medium",
                f"Top 特征 {top_feature} 重要性占比达到 {top_share:.1%}。",
                "核查该字段是否稳定、是否存在泄漏，必要时做去除该特征的对照实验。",
                ["top_feature_share"],
            )

    if not risks:
        recommendations.append(
            RecommendationItem(
                title="promote_candidate_model",
                action="当前训练指标没有触发明显风险，可进入模型对比或 MLOps 基线保存。",
                priority="medium",
                reason="验证指标、过拟合信号和阈值均未触发规则告警。",
            )
        )
        next_actions.append("保存模型并建立监控基线")

    risk_level = max_risk_level(risks)
    summary = f"训练结果规则诊断风险等级为 {risk_level}。"
    return AnalysisResult(
        analysis_type="training_rules",
        summary=summary,
        risk_level=risk_level,
        confidence=0.9,
        evidence=evidence,
        risks=risks,
        recommendations=recommendations,
        next_actions=next_actions,
        metadata={"model_type": params.get("model_type") or params.get("algorithm")},
    )


def diagnose_prediction(
    predictions: Iterable[Any],
    threshold: float = 0.5,
    reference_predictions: Optional[Iterable[Any]] = None,
) -> AnalysisResult:
    """Diagnose prediction probability distribution."""
    series = pd.to_numeric(pd.Series(list(predictions)), errors="coerce").dropna()
    evidence: List[EvidenceItem] = []
    risks: List[RiskItem] = []
    recommendations: List[RecommendationItem] = []
    next_actions: List[str] = []

    if series.empty:
        _risk(
            risks,
            "empty_predictions",
            "critical",
            "预测结果为空，无法分析分布。",
            "重新执行预测或检查模型输出。",
        )
        return AnalysisResult(
            analysis_type="prediction_rules",
            summary="预测结果为空。",
            risk_level="critical",
            confidence=1.0,
            risks=risks,
            next_actions=["重新执行预测"],
        )

    threshold = float(threshold)
    positive_rate = float((series >= threshold).mean())
    low_conf_rate = float(((series >= 0.4) & (series <= 0.6)).mean())
    std_value = float(series.std(ddof=0))
    evidence.extend(
        [
            EvidenceItem("prediction_count", len(series), "预测样本数"),
            EvidenceItem("threshold", threshold, "当前分类阈值"),
            EvidenceItem("positive_rate", round(positive_rate, 6), "预测正例比例"),
            EvidenceItem("mean_probability", round(float(series.mean()), 6), "平均预测概率"),
            EvidenceItem("std_probability", round(std_value, 6), "预测概率标准差"),
            EvidenceItem("low_confidence_rate", round(low_conf_rate, 6), "0.4-0.6 低置信区间比例"),
        ]
    )

    if positive_rate <= 0.01 or positive_rate >= 0.99:
        _risk(
            risks,
            "extreme_positive_rate",
            "high",
            f"预测正例比例为 {positive_rate:.2%}，几乎全为单一类别。",
            "检查预测数据分布、阈值和模型校准，必要时回看训练/线上数据漂移。",
            ["positive_rate"],
        )
    elif positive_rate <= 0.05 or positive_rate >= 0.95:
        _risk(
            risks,
            "skewed_positive_rate",
            "medium",
            f"预测正例比例为 {positive_rate:.2%}。",
            "与训练集正例比例对比，确认是否符合业务预期。",
            ["positive_rate"],
        )

    if low_conf_rate >= 0.3:
        _risk(
            risks,
            "many_low_confidence_predictions",
            "medium",
            f"低置信区间样本占比 {low_conf_rate:.1%}。",
            "优先抽样复核低置信样本，并考虑补充特征或校准概率。",
            ["low_confidence_rate"],
        )

    if std_value <= 0.03:
        _risk(
            risks,
            "collapsed_probability_distribution",
            "medium",
            f"预测概率标准差仅 {std_value:.4f}。",
            "检查模型是否输出近似常数、特征是否缺失或预处理是否失配。",
            ["std_probability"],
        )

    if reference_predictions is not None:
        ref = pd.to_numeric(pd.Series(list(reference_predictions)), errors="coerce").dropna()
        if not ref.empty:
            distance = _histogram_distance(ref, series)
            mean_delta = float(series.mean() - ref.mean())
            evidence.append(EvidenceItem("prediction_distribution_distance", round(distance, 6), "与参考预测分布的直方图距离"))
            evidence.append(EvidenceItem("mean_probability_delta", round(mean_delta, 6), "当前平均概率 - 参考平均概率"))
            if distance >= 0.35:
                _risk(
                    risks,
                    "prediction_distribution_shift",
                    "high",
                    f"当前预测分布与参考分布距离为 {distance:.4f}。",
                    "联动 MLOps PSI 检查输入特征漂移，并复核预测数据来源。",
                    ["prediction_distribution_distance"],
                )
            elif distance >= 0.2:
                _risk(
                    risks,
                    "moderate_prediction_shift",
                    "medium",
                    f"当前预测分布与参考分布距离为 {distance:.4f}。",
                    "观察正例率和主要特征分布，必要时重新校准阈值。",
                    ["prediction_distribution_distance"],
                )

    if not risks:
        recommendations.append(
            RecommendationItem(
                title="prediction_distribution_looks_stable",
                action="预测概率分布未触发明显风险，可按业务阈值抽样复核后上线使用。",
                priority="medium",
                reason="正例率、低置信样本比例和概率分布均在规则阈值内。",
            )
        )
        next_actions.append("抽样复核预测结果")

    risk_level = max_risk_level(risks)
    summary = f"预测结果包含 {len(series):,} 条记录，规则诊断风险等级为 {risk_level}。"
    return AnalysisResult(
        analysis_type="prediction_rules",
        summary=summary,
        risk_level=risk_level,
        confidence=0.9,
        evidence=evidence,
        risks=risks,
        recommendations=recommendations,
        next_actions=next_actions,
        metadata={"threshold": threshold},
    )


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _none_or_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _histogram_distance(left: pd.Series, right: pd.Series, bins: int = 20) -> float:
    left_values = left.clip(0, 1).to_numpy(dtype=float)
    right_values = right.clip(0, 1).to_numpy(dtype=float)
    left_hist, edges = np.histogram(left_values, bins=bins, range=(0, 1), density=False)
    right_hist, _ = np.histogram(right_values, bins=edges, density=False)
    left_prob = left_hist / max(left_hist.sum(), 1)
    right_prob = right_hist / max(right_hist.sum(), 1)
    return float(np.abs(left_prob - right_prob).sum() / 2.0)
