"""Markdown report builder for AI analysis diagnostics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .registry import ensure_within_project, project_root, record_analysis
from .schemas import AnalysisResult, RiskItem, max_risk_level


def default_report_dir() -> Path:
    return project_root() / "model_code" / "analysis_reports"


def build_composite_report(
    title: str,
    rule_results: Iterable[AnalysisResult],
    llm_sections: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Build a composite AnalysisResult from rule results and optional LLM text."""
    rule_results = list(rule_results)
    llm_sections = llm_sections or {}
    all_risks: list[RiskItem] = []
    evidence = []
    recommendations = []
    next_actions = []
    for result in rule_results:
        all_risks.extend(result.risks)
        evidence.extend(result.evidence)
        recommendations.extend(result.recommendations)
        next_actions.extend(result.next_actions)

    risk_level = max_risk_level(all_risks)
    summary = f"{title}：共汇总 {len(rule_results)} 个规则诊断结果，整体风险等级为 {risk_level}。"
    raw_llm_response = "\n\n".join(
        [f"## {name}\n{text}" for name, text in llm_sections.items() if text]
    )

    return AnalysisResult(
        analysis_type="composite_diagnosis",
        summary=summary,
        risk_level=risk_level,
        confidence=0.9 if rule_results else 0.5,
        evidence=evidence[:80],
        risks=all_risks,
        recommendations=recommendations,
        next_actions=_dedupe(next_actions),
        raw_llm_response=raw_llm_response,
        metadata=metadata or {},
    )


def render_composite_markdown(result: AnalysisResult, llm_sections: Optional[Dict[str, str]] = None) -> str:
    lines = [
        f"# {result.summary}",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 整体风险等级：{result.risk_level}",
        f"- 置信度：{result.confidence:.2f}",
        "",
    ]

    if result.risks:
        lines.extend(["## 风险清单", ""])
        for item in result.risks:
            lines.append(f"- **[{item.severity}] {item.title}**：{item.reason}")
            if item.recommendation:
                lines.append(f"  - 建议：{item.recommendation}")
        lines.append("")

    if result.evidence:
        lines.extend(["## 关键证据", ""])
        for item in result.evidence[:40]:
            detail = f" - {item.detail}" if item.detail else ""
            lines.append(f"- **{item.name}**: {item.value}{detail}")
        lines.append("")

    if result.recommendations:
        lines.extend(["## 建议", ""])
        for item in result.recommendations:
            lines.append(f"- **[{item.priority}] {item.title}**：{item.action}")
            if item.reason:
                lines.append(f"  - 理由：{item.reason}")
        lines.append("")

    if result.next_actions:
        lines.extend(["## 下一步操作", ""])
        lines.extend([f"- {item}" for item in result.next_actions])
        lines.append("")

    if llm_sections:
        lines.extend(["## AI 分析原文", ""])
        for name, text in llm_sections.items():
            if not text:
                continue
            lines.extend([f"### {name}", "", text, ""])

    return "\n".join(lines)


def save_composite_report(
    result: AnalysisResult,
    llm_sections: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
    filename_prefix: str = "composite_diagnosis",
) -> str:
    report_dir = ensure_within_project(output_dir or default_report_dir())
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ensure_within_project(report_dir / f"{filename_prefix}_{timestamp}.md")
    markdown = render_composite_markdown(result, llm_sections=llm_sections)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)

    record_analysis(result, report_path=str(path))
    return str(path)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        item = str(item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
