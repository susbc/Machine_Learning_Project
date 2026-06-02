import pandas as pd

from model_code.analysis.report_builder import build_composite_report, render_composite_markdown
from model_code.analysis.rule_engine import diagnose_dataset, diagnose_prediction, diagnose_training
from model_code.analysis.schemas import (
    AnalysisResult,
    EvidenceItem,
    RiskItem,
    coerce_analysis_result,
    extract_json_payload,
)
from model_code.analysis.registry import scrub_sensitive


def test_extract_json_payload_from_fenced_response():
    text = """
LLM text

```json
{"summary": "ok", "risk_level": "medium", "confidence": 0.8}
```
"""
    payload = extract_json_payload(text)
    assert payload["summary"] == "ok"
    assert payload["risk_level"] == "medium"


def test_coerce_analysis_result_merges_rule_result_when_llm_is_plain_text():
    base = AnalysisResult(
        analysis_type="dataset_rules",
        summary="rules",
        risk_level="high",
        confidence=0.9,
        evidence=[EvidenceItem("missing_rate", 0.3)],
        risks=[RiskItem("high_missing_rate", "high")],
    )
    result = coerce_analysis_result(
        "plain markdown without json",
        analysis_type="data_analysis",
        base_result=base,
    )
    assert result.risk_level == "high"
    assert result.raw_llm_response == "plain markdown without json"
    assert any(item.name == "missing_rate" for item in result.evidence)


def test_dataset_rules_detect_leakage_and_imbalance():
    df = pd.DataFrame(
        {
            "user_id": range(100),
            "future_result_score": range(100),
            "constant": 1,
            "feature": range(100),
            "label": [1] + [0] * 99,
        }
    )
    result = diagnose_dataset(df, "label")
    risk_titles = {item.title for item in result.risks}
    assert "possible_target_leakage" in risk_titles
    assert "constant_features" in risk_titles
    assert "class_imbalance" in risk_titles or "severe_class_imbalance" in risk_titles
    assert result.risk_level == "high"


def test_training_rules_handle_missing_train_metrics():
    result = diagnose_training(
        metrics={"val_auc": 0.58, "val_f1": 0.4, "train_size": 80, "n_features": 30},
        params={"model_type": "lightgbm"},
    )
    risk_titles = {item.title for item in result.risks}
    assert "weak_validation_auc" in risk_titles
    assert "weak_validation_f1" in risk_titles
    assert result.risk_level == "high"


def test_prediction_rules_detect_extreme_positive_rate_and_shift():
    current = [0.99] * 100
    reference = [0.05] * 50 + [0.95] * 50
    result = diagnose_prediction(current, threshold=0.5, reference_predictions=reference)
    risk_titles = {item.title for item in result.risks}
    assert "extreme_positive_rate" in risk_titles
    assert "prediction_distribution_shift" in risk_titles
    assert result.risk_level == "high"


def test_composite_report_uses_max_risk_level():
    low = AnalysisResult("low_rule", "low", risk_level="low")
    high = AnalysisResult("high_rule", "high", risk_level="high", risks=[RiskItem("r", "high")])
    composite = build_composite_report("综合诊断", [low, high])
    markdown = render_composite_markdown(composite)
    assert composite.risk_level == "high"
    assert "整体风险等级：high" in markdown


def test_registry_scrubs_sensitive_metadata():
    payload = {
        "api_key": "abc",
        "nested": {"password": "secret", "safe": "value"},
    }
    cleaned = scrub_sensitive(payload)
    assert cleaned["api_key"] == "***REDACTED***"
    assert cleaned["nested"]["password"] == "***REDACTED***"
    assert cleaned["nested"]["safe"] == "value"
