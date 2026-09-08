import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


def _dataset_file() -> tuple[str, io.BytesIO, str]:
    df = pd.DataFrame(
        {
            "user_id": range(1, 41),
            "future_result_score": range(40),
            "feature": range(40),
            "label": [1] + [0] * 39,
        }
    )
    return "analysis.csv", io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv"


def test_dataset_analysis_returns_structured_rule_result():
    client = TestClient(app)
    file_name, file_body, content_type = _dataset_file()

    response = client.post(
        "/api/analysis/dataset",
        files={"file": (file_name, file_body, content_type)},
        data={"target_col": "label"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["analysis_type"] == "dataset_rules"
    assert result["risk_level"] == "high"
    assert any(item["title"] == "possible_target_leakage" for item in result["risks"])
    assert "api_key" not in str(result).lower()


def test_training_and_prediction_analysis_accept_json_payloads():
    client = TestClient(app)

    training_response = client.post(
        "/api/analysis/training",
        json={
            "metrics": {"val_auc": 0.58, "val_f1": 0.4, "train_size": 80, "n_features": 30},
            "params": {"model_type": "lightgbm", "api_key": "secret"},
        },
    )
    prediction_response = client.post(
        "/api/analysis/prediction",
        json={"predictions": [0.99] * 30, "threshold": 0.5, "reference_predictions": [0.1, 0.2, 0.8, 0.9]},
    )

    assert training_response.status_code == 200
    assert training_response.json()["risk_level"] == "high"
    assert "secret" not in str(training_response.json())
    assert prediction_response.status_code == 200
    assert prediction_response.json()["risk_level"] == "high"


def test_composite_analysis_can_save_report_and_list_registry():
    client = TestClient(app)
    payload = {
        "title": "AI 智能分析综合诊断",
        "results": [
            {
                "analysis_type": "training_rules",
                "summary": "训练风险",
                "risk_level": "high",
                "confidence": 0.9,
                "risks": [{"title": "weak_validation_auc", "severity": "high"}],
            }
        ],
        "save_report": True,
        "metadata": {"api_key": "must_not_leak", "safe": "ok"},
    }

    response = client.post("/api/analysis/composite", json=payload)
    reports_response = client.get("/api/analysis/reports")

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["analysis_type"] == "composite_diagnosis"
    assert body["result"]["risk_level"] == "high"
    assert body["report_path"].endswith(".md")
    assert "must_not_leak" not in str(body)

    assert reports_response.status_code == 200
    reports = reports_response.json()["reports"]
    assert any(record["analysis_type"] == "composite_diagnosis" for record in reports)
    assert "must_not_leak" not in str(reports)


def test_get_analysis_report_by_id_returns_markdown():
    client = TestClient(app)
    payload = {
        "title": "按 ID 读取报告测试",
        "results": [
            {
                "analysis_type": "training_rules",
                "summary": "训练风险",
                "risk_level": "high",
                "confidence": 0.9,
                "risks": [{"title": "weak_validation_auc", "severity": "high"}],
            }
        ],
        "save_report": True,
    }
    client.post("/api/analysis/composite", json=payload)
    reports = client.get("/api/analysis/reports").json()["reports"]
    record = next(item for item in reports if item.get("report_path"))

    response = client.get(f"/api/analysis/reports/{record['analysis_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["record"]["analysis_id"] == record["analysis_id"]
    assert "按 ID 读取报告测试" in body["markdown"]


def test_get_analysis_report_unknown_id_returns_404():
    client = TestClient(app)

    response = client.get("/api/analysis/reports/nonexistent-analysis-id")

    assert response.status_code == 404
