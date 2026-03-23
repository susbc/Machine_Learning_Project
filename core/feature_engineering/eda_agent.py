"""
EDA Agent - 探索性数据分析智能体

Phase 2 核心组件：扮演"数据科学家"角色，阅读统计特征画像，输出 EDA 洞察报告。

核心设计理念:
- Python 负责"出苦力"算统计量
- LLM 只负责"做决策"并输出结构化分析

输入：eda_summary.json (几 KB 的统计压缩数据)
输出：EDA 诊断报告 (人类可读 + 下游 Feature_Agent 上下文)
"""

from typing import Dict, Any, Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class EDAAgent:
    """
    探索性数据分析智能体
    
    职责:
    1. 阅读压缩后的统计特征画像
    2. 结合业务场景输出 EDA 洞察
    3. 圈出重点风险 (数据质量/业务冲突/数据穿越)
    4. 生成下游 Feature_Agent 的上下文
    """
    
    # 系统提示词模板
    SYSTEM_PROMPT = """
# Role
你是一位资深数据科学家，专注于机器学习数据质量诊断和探索性分析 (EDA)。你的任务是阅读数据集的统计特征画像，并针对特定业务场景输出专业的 EDA 洞察报告。

# Input
统计特征画像 (JSON 格式):
<eda_summary>
{eda_summary}
</eda_summary>

业务场景:
<business_scenario>
{business_scenario}
</business_scenario>

# Task
请输出一份结构化的 EDA 分析报告，必须包含以下部分:

## 1. 数据概览
- 样本量、特征数、目标变量分布
- 数据量级评估 (小/中/大规模)

## 2. 数据质量告警
指出以下问题字段:
- **高缺失率**: null_rate > 10% 的字段
- **严重偏态**: skewness > 3 或 < -3 的数值字段
- **单一值过多**: unique_values = 1 或 top_1_ratio > 0.95 的字段
- **异常值风险**: 存在明显超出合理范围的字段

## 3. 业务逻辑冲突
判断是否有违背业务常识的数据:
- 年龄/驾龄存在负数
- 金额存在负数 (除非允许退款)
- 时间字段存在未来日期
- 比例/概率字段超出 [0, 1] 范围

## 4. 数据穿越风险 (Leakage Detection)
指出与目标变量相关性高得不正常的特征:
- **高风险**: corr_with_target > 0.8 (可能包含未来信息)
- **中风险**: 0.5 < corr_with_target <= 0.8 (需谨慎使用)
- 特别说明：某些高相关性可能是合理的 (如余额与透支状态)

## 5. 特征处理建议
针对上述问题，给出初步处理建议:
- 哪些字段建议直接删除 (drop)
- 哪些字段需要特殊变换 (log/sqrt)
- 哪些高基数类别特征需要编码 (target_encode)

# Output Format
请使用清晰的 Markdown 格式输出报告，便于前端展示和下游 Feature_Agent 理解。
"""
    
    def __init__(self, llm_provider=None):
        """
        初始化 EDA Agent
        
        Args:
            llm_provider: LLM 提供者 (可选)
        """
        self.llm_provider = llm_provider
    
    def analyze(self, eda_summary: Dict[str, Any],
                business_scenario: str = "通用机器学习建模") -> Dict[str, Any]:
        """
        执行 EDA 分析
        
        Args:
            eda_summary: 统计特征画像
                {
                    "dataset_meta": {"rows": 100000, "columns": 15, "target_col": "is_fraud"},
                    "features": {
                        "driver_age": {"type": "numeric", "null_rate": 0.05, "mean": 35.2, ...},
                        "city_code": {"type": "categorical", "unique_values": 342, ...},
                        ...
                    }
                }
            business_scenario: 业务场景描述
            
        Returns:
            Dict: {
                "report_text": str,  # 人类可读报告
                "risk_flags": List[Dict],  # 风险标记列表
                "recommendations": List[Dict],  # 处理建议
                "context_for_feature_agent": str  # 传递给 Feature_Agent 的上下文
            }
        """
        logger.info(f"开始 EDA 分析，业务场景：{business_scenario}")
        
        # 1. 构建 Prompt
        prompt = self.SYSTEM_PROMPT.format(
            eda_summary=json.dumps(eda_summary, indent=2, ensure_ascii=False),
            business_scenario=business_scenario
        )
        
        # 2. 调用 LLM
        llm_response = self._call_llm(prompt)
        
        # 3. 解析响应，提取关键信息
        analysis_result = self._parse_analysis(llm_response, eda_summary)
        
        logger.info(f"EDA 分析完成，发现 {len(analysis_result['risk_flags'])} 个风险点")
        
        return analysis_result
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成分析"""
        if not self.llm_provider:
            logger.warning("未配置 LLM，使用规则引擎生成 EDA 报告")
            return self._rule_based_eda()
        
        try:
            response = self.llm_provider.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return self._rule_based_eda()
    
    def _rule_based_eda(self) -> str:
        """基于规则的 EDA 分析 (LLM 不可用时的降级方案)"""
        report = []
        report.append("## EDA 分析报告 (规则引擎生成)\n")
        report.append("**注意**: 未配置 LLM，使用基础统计规则生成报告。\n")
        report.append("---\n")
        return "\n".join(report)
    
    def _parse_analysis(self, llm_response: str,
                        eda_summary: Dict) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取结构化信息
        
        Args:
            llm_response: LLM 生成的报告文本
            eda_summary: 原始统计画像
            
        Returns:
            Dict: 结构化分析结果
        """
        result = {
            "report_text": llm_response,
            "risk_flags": [],
            "recommendations": [],
            "context_for_feature_agent": ""
        }
        
        # 从统计画像中自动提取风险标记
        features = eda_summary.get("features", {})
        
        for col_name, stats in features.items():
            # 高缺失率风险
            if stats.get("null_rate", 0) > 0.1:
                result["risk_flags"].append({
                    "type": "high_missing_rate",
                    "column": col_name,
                    "value": stats.get("null_rate"),
                    "severity": "high" if stats.get("null_rate") > 0.5 else "medium"
                })
            
            # 严重偏态风险
            skewness = stats.get("skewness", 0)
            if abs(skewness) > 3:
                result["risk_flags"].append({
                    "type": "severe_skewness",
                    "column": col_name,
                    "value": skewness,
                    "severity": "medium"
                })
            
            # 高相关性风险 (数据穿越)
            corr = stats.get("corr_with_target", 0)
            if abs(corr) > 0.8:
                result["risk_flags"].append({
                    "type": "potential_leakage",
                    "column": col_name,
                    "value": corr,
                    "severity": "high"
                })
            elif abs(corr) > 0.5:
                result["risk_flags"].append({
                    "type": "high_correlation",
                    "column": col_name,
                    "value": corr,
                    "severity": "low"
                })
            
            # 高基数类别特征
            if stats.get("type") == "categorical":
                unique_vals = stats.get("unique_values", 0)
                if unique_vals > 100:
                    result["risk_flags"].append({
                        "type": "high_cardinality",
                        "column": col_name,
                        "value": unique_vals,
                        "severity": "medium"
                    })
        
        # 生成传递给 Feature_Agent 的上下文
        result["context_for_feature_agent"] = self._build_feature_context(
            eda_summary, result["risk_flags"]
        )
        
        # 生成处理建议
        result["recommendations"] = self._generate_recommendations(
            eda_summary, result["risk_flags"]
        )
        
        return result
    
    def _build_feature_context(self, eda_summary: Dict,
                                risk_flags: List[Dict]) -> str:
        """
        构建传递给 Feature_Agent 的上下文
        
        Args:
            eda_summary: 统计画像
            risk_flags: 风险标记列表
            
        Returns:
            str: 上下文文本
        """
        lines = []
        lines.append("# EDA 诊断摘要 (供 Feature_Agent 参考)\n")
        
        # 数据集概览
        meta = eda_summary.get("dataset_meta", {})
        lines.append(f"**数据集规模**: {meta.get('rows', 0):,} 行 × {meta.get('columns', 0)} 列")
        lines.append(f"**目标变量**: {meta.get('target_col', 'N/A')}\n")
        
        # 风险汇总
        if risk_flags:
            lines.append("**发现的风险**:\n")
            by_severity = {"high": [], "medium": [], "low": []}
            for flag in risk_flags:
                by_severity[flag["severity"]].append(flag)
            
            if by_severity["high"]:
                lines.append(f"- 🔴 高风险 ({len(by_severity['high'])} 个): " +
                            ", ".join([f"{f['column']}({f['type']})" for f in by_severity['high']]))
            if by_severity["medium"]:
                lines.append(f"- 🟡 中风险 ({len(by_severity['medium'])} 个): " +
                            ", ".join([f"{f['column']}({f['type']})" for f in by_severity['medium']]))
            if by_severity["low"]:
                lines.append(f"- 🟢 低风险 ({len(by_severity['low'])} 个): " +
                            ", ".join([f"{f['column']}({f['type']})" for f in by_severity['low']]))
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, eda_summary: Dict,
                                   risk_flags: List[Dict]) -> List[Dict]:
        """
        生成处理建议
        
        Args:
            eda_summary: 统计画像
            risk_flags: 风险标记列表
            
        Returns:
            List[Dict]: 建议列表
        """
        recommendations = []
        
        for flag in risk_flags:
            if flag["type"] == "high_missing_rate":
                if flag["value"] > 0.5:
                    recommendations.append({
                        "column": flag["column"],
                        "action": "drop",
                        "reason": f"缺失率过高 ({flag['value']:.1%})，建议删除"
                    })
                else:
                    recommendations.append({
                        "column": flag["column"],
                        "action": "fillna",
                        "reason": f"缺失率 {flag['value']:.1%}，需要填充"
                    })
            
            elif flag["type"] == "severe_skewness":
                recommendations.append({
                    "column": flag["column"],
                    "action": "log_transform",
                    "reason": f"严重偏态 (skewness={flag['value']:.2f})，建议对数变换"
                })
            
            elif flag["type"] == "potential_leakage":
                recommendations.append({
                    "column": flag["column"],
                    "action": "review_leakage",
                    "reason": f"与目标变量相关性过高 (corr={flag['value']:.2f})，可能存在数据穿越，需人工审核"
                })
            
            elif flag["type"] == "high_cardinality":
                recommendations.append({
                    "column": flag["column"],
                    "action": "target_encode",
                    "reason": f"高基数类别特征 ({flag['value']} 个唯一值)，建议使用 Target Encoding"
                })
        
        return recommendations


def compute_eda_summary(df, target_col: str = None) -> Dict[str, Any]:
    """
    计算 EDA 统计摘要 (Python 前置压缩)
    
    Args:
        df: pandas DataFrame
        target_col: 目标变量列名
        
    Returns:
        Dict: 统计特征画像
    """
    import pandas as pd
    import numpy as np
    
    summary = {
        "dataset_meta": {
            "rows": len(df),
            "columns": len(df.columns),
            "target_col": target_col
        },
        "features": {}
    }
    
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        null_rate = col_data.isnull().mean()
        
        feature_stats = {
            "type": _infer_type(dtype),
            "null_rate": float(null_rate)
        }
        
        # 数值型特征统计
        if feature_stats["type"] == "numeric":
            non_null = col_data.dropna()
            if len(non_null) > 0:
                feature_stats["mean"] = float(non_null.mean())
                feature_stats["std"] = float(non_null.std())
                feature_stats["min"] = float(non_null.min())
                feature_stats["max"] = float(non_null.max())
                
                # 偏度
                if len(non_null) > 2:
                    feature_stats["skewness"] = float((non_null - non_null.mean()).pow(3).mean() / (non_null.std() ** 3))
                
                # 与目标变量的相关性
                if target_col and target_col in df.columns:
                    target_data = df[target_col]
                    if _infer_type(str(target_data.dtype)) == "numeric":
                        corr = col_data.corr(target_data)
                        if not np.isnan(corr):
                            feature_stats["corr_with_target"] = float(corr)
        
        # 类别型特征统计
        elif feature_stats["type"] == "categorical":
            non_null = col_data.dropna().astype(str)
            if len(non_null) > 0:
                feature_stats["unique_values"] = int(non_null.nunique())
                top_value = non_null.value_counts().iloc[0] if len(non_null) > 0 else 0
                feature_stats["top_1_ratio"] = float(top_value / len(non_null))
                feature_stats["top_1_value"] = str(non_null.mode().iloc[0]) if len(non_null) > 0 else None
        
        # 时间型特征统计
        elif feature_stats["type"] == "datetime":
            non_null = col_data.dropna()
            if len(non_null) > 0:
                feature_stats["min"] = str(non_null.min())
                feature_stats["max"] = str(non_null.max())
        
        summary["features"][col] = feature_stats
    
    return summary


def _infer_type(dtype: str) -> str:
    """推断字段类型"""
    dtype_lower = dtype.lower()
    
    if any(t in dtype_lower for t in ["int", "float", "double", "decimal"]):
        return "numeric"
    elif any(t in dtype_lower for t in ["datetime", "timestamp", "date", "time"]):
        return "datetime"
    elif any(t in dtype_lower for t in ["bool"]):
        return "boolean"
    else:
        return "categorical"
