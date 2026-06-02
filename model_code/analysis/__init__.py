"""Analysis utilities for AI-assisted diagnostics."""

from .schemas import (
    AnalysisResult,
    EvidenceItem,
    RecommendationItem,
    RiskItem,
    coerce_analysis_result,
    extract_json_payload,
)
from .rule_engine import (
    diagnose_dataset,
    diagnose_prediction,
    diagnose_training,
)
from .report_builder import (
    build_composite_report,
    render_composite_markdown,
    save_composite_report,
)
from .registry import (
    list_recent_records,
    record_analysis,
)

__all__ = [
    "AnalysisResult",
    "EvidenceItem",
    "RecommendationItem",
    "RiskItem",
    "coerce_analysis_result",
    "extract_json_payload",
    "diagnose_dataset",
    "diagnose_prediction",
    "diagnose_training",
    "build_composite_report",
    "render_composite_markdown",
    "save_composite_report",
    "list_recent_records",
    "record_analysis",
]
