"""Structured analysis result schemas.

These dataclasses intentionally avoid extra runtime dependencies. They are
small enough for Streamlit session state, report export, and JSONL tracking.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RISK_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
PRIORITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def json_default(value: Any) -> Any:
    """JSON fallback that handles numpy/pandas-like scalar values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass
class EvidenceItem:
    name: str
    value: Any
    detail: str = ""
    source: str = "rule"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskItem:
    title: str
    severity: str = "medium"
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationItem:
    title: str
    action: str
    priority: str = "medium"
    reason: str = ""
    expected_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    analysis_type: str
    summary: str
    risk_level: str = "low"
    confidence: float = 0.0
    evidence: List[EvidenceItem] = field(default_factory=list)
    risks: List[RiskItem] = field(default_factory=list)
    recommendations: List[RecommendationItem] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    raw_llm_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "summary": self.summary,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "risks": [item.to_dict() for item in self.risks],
            "recommendations": [item.to_dict() for item in self.recommendations],
            "next_actions": list(self.next_actions),
            "raw_llm_response": self.raw_llm_response,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnalysisResult":
        return cls(
            analysis_type=str(payload.get("analysis_type") or "generic"),
            summary=str(payload.get("summary") or ""),
            risk_level=normalize_risk_level(payload.get("risk_level") or "low"),
            confidence=safe_float(payload.get("confidence"), 0.0),
            evidence=[EvidenceItem(**item) for item in payload.get("evidence", []) if isinstance(item, dict)],
            risks=[RiskItem(**item) for item in payload.get("risks", []) if isinstance(item, dict)],
            recommendations=[
                RecommendationItem(**item)
                for item in payload.get("recommendations", [])
                if isinstance(item, dict)
            ],
            next_actions=[str(item) for item in payload.get("next_actions", [])],
            raw_llm_response=str(payload.get("raw_llm_response") or ""),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or datetime.now().isoformat(timespec="seconds")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=json_default)

    def to_markdown(self, include_raw: bool = False) -> str:
        lines = [
            f"## {self.analysis_type}",
            "",
            f"**风险等级**：{self.risk_level}",
            f"**置信度**：{self.confidence:.2f}",
            "",
            "### 摘要",
            self.summary or "无摘要",
        ]

        if self.evidence:
            lines.extend(["", "### 关键证据"])
            for item in self.evidence:
                detail = f" - {item.detail}" if item.detail else ""
                lines.append(f"- **{item.name}**: {item.value}{detail}")

        if self.risks:
            lines.extend(["", "### 风险"])
            for item in self.risks:
                lines.append(f"- **[{item.severity}] {item.title}**：{item.reason}")
                if item.recommendation:
                    lines.append(f"  建议：{item.recommendation}")

        if self.recommendations:
            lines.extend(["", "### 建议"])
            for item in self.recommendations:
                lines.append(f"- **[{item.priority}] {item.title}**：{item.action}")
                if item.reason:
                    lines.append(f"  理由：{item.reason}")

        if self.next_actions:
            lines.extend(["", "### 下一步"])
            lines.extend([f"- {item}" for item in self.next_actions])

        if include_raw and self.raw_llm_response:
            lines.extend(["", "### LLM 原文", self.raw_llm_response])

        return "\n".join(lines)


def normalize_risk_level(value: str) -> str:
    normalized = str(value or "low").strip().lower()
    aliases = {
        "normal": "low",
        "warning": "medium",
        "danger": "high",
        "severe": "critical",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RISK_ORDER:
        return "low"
    return normalized


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def max_risk_level(items: Iterable[RiskItem], default: str = "low") -> str:
    level = normalize_risk_level(default)
    for item in items:
        item_level = normalize_risk_level(item.severity)
        if RISK_ORDER[item_level] > RISK_ORDER[level]:
            level = item_level
    return level


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a messy LLM response."""
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    return None


def _coerce_evidence(items: Any) -> List[EvidenceItem]:
    if not isinstance(items, list):
        return []
    evidence = []
    for item in items:
        if isinstance(item, dict):
            evidence.append(
                EvidenceItem(
                    name=str(item.get("name") or item.get("title") or "evidence"),
                    value=item.get("value", ""),
                    detail=str(item.get("detail") or item.get("reason") or ""),
                    source=str(item.get("source") or "llm"),
                )
            )
        else:
            evidence.append(EvidenceItem(name="evidence", value=str(item), source="llm"))
    return evidence


def _coerce_risks(items: Any) -> List[RiskItem]:
    if not isinstance(items, list):
        return []
    risks = []
    for item in items:
        if isinstance(item, dict):
            risks.append(
                RiskItem(
                    title=str(item.get("title") or item.get("risk") or "risk"),
                    severity=normalize_risk_level(item.get("severity") or item.get("risk_level") or "medium"),
                    reason=str(item.get("reason") or item.get("detail") or ""),
                    evidence=[str(v) for v in item.get("evidence", [])] if isinstance(item.get("evidence"), list) else [],
                    recommendation=str(item.get("recommendation") or item.get("action") or ""),
                )
            )
        else:
            risks.append(RiskItem(title=str(item), severity="medium"))
    return risks


def _coerce_recommendations(items: Any) -> List[RecommendationItem]:
    if not isinstance(items, list):
        return []
    recommendations = []
    for item in items:
        if isinstance(item, dict):
            recommendations.append(
                RecommendationItem(
                    title=str(item.get("title") or item.get("name") or "recommendation"),
                    action=str(item.get("action") or item.get("recommendation") or item.get("detail") or ""),
                    priority=str(item.get("priority") or "medium"),
                    reason=str(item.get("reason") or ""),
                    expected_impact=str(item.get("expected_impact") or item.get("impact") or ""),
                )
            )
        else:
            recommendations.append(RecommendationItem(title="recommendation", action=str(item)))
    return recommendations


def coerce_analysis_result(
    text: str,
    analysis_type: str,
    fallback_summary: str = "",
    base_result: Optional[AnalysisResult] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Coerce LLM text into AnalysisResult with robust fallback."""
    payload = extract_json_payload(text)
    if payload:
        result = AnalysisResult(
            analysis_type=str(payload.get("analysis_type") or analysis_type),
            summary=str(payload.get("summary") or fallback_summary or text[:500]),
            risk_level=normalize_risk_level(payload.get("risk_level") or "low"),
            confidence=safe_float(payload.get("confidence"), 0.5),
            evidence=_coerce_evidence(payload.get("evidence", [])),
            risks=_coerce_risks(payload.get("risks", [])),
            recommendations=_coerce_recommendations(payload.get("recommendations", [])),
            next_actions=[str(item) for item in payload.get("next_actions", [])]
            if isinstance(payload.get("next_actions"), list)
            else [],
            raw_llm_response=text,
            metadata=dict(metadata or {}),
        )
    else:
        result = AnalysisResult(
            analysis_type=analysis_type,
            summary=fallback_summary or "LLM 返回了非结构化内容，已保留原文。",
            risk_level="low",
            confidence=0.35,
            recommendations=[
                RecommendationItem(
                    title="review_llm_response",
                    action="查看 LLM 原文，并结合规则诊断判断是否采纳。",
                    priority="medium",
                    reason="模型输出未能解析为结构化 JSON。",
                )
            ],
            raw_llm_response=text,
            metadata=dict(metadata or {}),
        )

    if base_result is not None:
        result.evidence = [*base_result.evidence, *result.evidence]
        result.risks = [*base_result.risks, *result.risks]
        result.recommendations = [*base_result.recommendations, *result.recommendations]
        result.next_actions = [*base_result.next_actions, *result.next_actions]
        result.risk_level = max_risk_level(result.risks, default=base_result.risk_level)
        result.confidence = max(result.confidence, base_result.confidence)
        result.metadata = {**base_result.metadata, **result.metadata}

    return result
