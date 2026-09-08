"""Regression milestone acceptance test across every public training entry point."""

from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.api.automl as automl_api
import backend.api.models as model_registry
import backend.api.training as manual_training
import backend.api.workflow as workflow
from backend.main import app
from backend.runtime_state import RuntimeStateStore, background_tasks
from core.llm.agent_schemas import FeatureAction, FeatureStrategy


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _linear_frame(rows: int = 120) -> pd.DataFrame:
    x = np.linspace(-20.0, 20.0, rows)
    return pd.DataFrame(
        {
            "x": x,
            "segment": np.where(x > 0, "positive", "negative"),
            "price": 3.25 * x + 17.0,
        }
    )


def _csv_file(frame: pd.DataFrame) -> tuple[str, io.BytesIO, str]:
    return (
        "linear-regression.csv",
        io.BytesIO(frame.to_csv(index=False).encode("utf-8")),
        "text/csv",
    )


def _wait_for_job(
    client: TestClient,
    status_path: str,
    *,
    task_key: str,
    timeout: float = 15.0,
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(status_path)
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get("status") in TERMINAL_STATUSES:
            if last["status"] != "completed":
                raise AssertionError(
                    f"Regression training ended as {last['status']}: {last.get('error')}"
                )
            while background_tasks.is_active(task_key) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not background_tasks.is_active(task_key), f"Background task did not exit: {task_key}"
            return last
        time.sleep(0.05)
    raise AssertionError(f"Regression training timed out: {last}")


def _assert_exact_linear_metrics(metrics: dict) -> None:
    # A noiseless affine target is represented exactly by ordinary least squares.
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["r2"] >= 0.95


def _prepare_workflow_input(root: Path, frame: pd.DataFrame) -> str:
    task_id = "task_e2e00001"
    task_dir = root / "output" / task_id
    task_dir.mkdir(parents=True)
    frame.to_parquet(task_dir / "raw_data.parquet", index=False)
    (task_dir / "eda_summary.json").write_text(
        json.dumps(
            {"target": "price", "user_query": "预测连续价格"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    strategy = FeatureStrategy(
        actions=[FeatureAction(column="x", action="fillna_mean", reason="保留数值特征")]
    )
    (task_dir / "feature_strategy.json").write_text(
        strategy.model_dump_json(), encoding="utf-8"
    )
    return task_id


def _workflow_events(client: TestClient, task_id: str) -> list[dict]:
    with client.stream(
        "GET",
        "/api/workflow/step3_train_stream",
        params={
            "task_id": task_id,
            "problem_type": "auto",
            "n_trials": 10,
            "cv_folds": 3,
        },
    ) as response:
        assert response.status_code == 200, response.text
        return [
            json.loads(line.removeprefix("data:").strip())
            for line in response.iter_lines()
            if line.startswith("data:")
        ]


def test_regression_end_to_end_across_manual_automl_and_agent_workflow(
    tmp_path, monkeypatch
):
    """Catches any break in regression train, recovery, registry, or inference."""
    runtime_store = RuntimeStateStore(tmp_path / "runtime_state")
    monkeypatch.setattr(manual_training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        manual_training, "ARTIFACT_DIR", tmp_path / "saved" / "manual_training"
    )
    monkeypatch.setattr(manual_training, "runtime_state", runtime_store)
    monkeypatch.setattr(manual_training, "training_datasets", {})
    monkeypatch.setattr(manual_training, "training_jobs", {})
    monkeypatch.setattr(automl_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(automl_api, "ARTIFACT_DIR", tmp_path / "saved" / "automl")
    monkeypatch.setattr(automl_api, "runtime_state", runtime_store)
    monkeypatch.setattr(automl_api, "automl_jobs", {})
    monkeypatch.setattr(workflow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        workflow,
        "recommend_automl_config",
        lambda _features, _scenario: {
            "task_type": "binary_classification",
            "optimization_metric": "roc_auc",
            "candidate_models": ["logistic_regression"],
            "n_trials": 10,
            "search_spaces": {},
            "rationale": "deterministic offline mismatch",
        },
    )

    frame = _linear_frame()
    client = TestClient(app)

    upload = client.post(
        "/api/training/datasets",
        files={"file": _csv_file(frame)},
        data={"target_col": "price", "problem_type": "auto"},
    )
    assert upload.status_code == 201, upload.text
    uploaded = upload.json()
    assert uploaded["problem_spec"]["requested_type"] == "auto"
    assert uploaded["problem_spec"]["resolved_type"] == "regression"
    assert uploaded["problem_spec"]["source"] == "auto_inferred"
    assert uploaded["problem_spec"]["capabilities"]["metrics"] == ["rmse", "mae", "r2"]

    created = client.post(
        "/api/training/jobs",
        json={
            "dataset_id": uploaded["dataset_id"],
            "algorithm": "linear_regression",
            "problem_type": "auto",
            "params": {},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert created.status_code == 201, created.text
    manual_job_id = created.json()["job_id"]
    _wait_for_job(
        client,
        f"/api/training/jobs/{manual_job_id}",
        task_key=f"training-{manual_job_id}",
    )

    manual_training.training_jobs.pop(manual_job_id)
    manual_training.training_datasets.pop(uploaded["dataset_id"])
    restored_manual = client.get(f"/api/training/jobs/{manual_job_id}")
    assert restored_manual.status_code == 200, restored_manual.text
    assert restored_manual.json()["problem_spec"]["resolved_type"] == "regression"
    manual_result = client.get(f"/api/training/jobs/{manual_job_id}/result")
    assert manual_result.status_code == 200, manual_result.text
    _assert_exact_linear_metrics(manual_result.json()["metrics"])
    assert manual_result.json()["confusion_matrix"] is None
    assert manual_result.json()["threshold_calibration"] is None

    saved_manual = client.post(f"/api/training/jobs/{manual_job_id}/save")
    assert saved_manual.status_code == 200, saved_manual.text
    manual_artifact_path = tmp_path / saved_manual.json()["artifact_path"]
    manual_artifact = joblib.load(manual_artifact_path)
    assert manual_artifact["artifact_schema_version"] == 3
    assert manual_artifact["problem_type"] == "regression"
    assert manual_artifact["problem_spec"]["resolved_type"] == "regression"
    assert manual_artifact["label_encoder"] is None
    assert manual_artifact["decision_threshold"] is None

    automl_created = client.post(
        "/api/automl/jobs",
        files={"file": _csv_file(frame)},
        data={
            "target_col": "price",
            "problem_type": "regression",
            "optimization_metric": "auto",
            "candidate_models": '["linear_regression"]',
            "n_trials": 1,
            "cv_folds": 3,
            "use_llm": "false",
        },
    )
    assert automl_created.status_code == 201, automl_created.text
    automl_payload = automl_created.json()
    assert automl_payload["problem_spec"]["requested_type"] == "regression"
    assert automl_payload["problem_spec"]["resolved_type"] == "regression"
    assert automl_payload["problem_spec"]["source"] == "user_override"
    assert automl_payload["optimization_metric"] == "rmse"
    automl_job_id = automl_payload["job_id"]
    _wait_for_job(
        client,
        f"/api/automl/jobs/{automl_job_id}",
        task_key=f"automl-{automl_job_id}",
    )

    automl_api.automl_jobs.pop(automl_job_id)
    restored_automl = client.get(f"/api/automl/jobs/{automl_job_id}")
    assert restored_automl.status_code == 200, restored_automl.text
    assert restored_automl.json()["problem_spec"]["resolved_type"] == "regression"
    automl_result = client.get(f"/api/automl/jobs/{automl_job_id}/result")
    assert automl_result.status_code == 200, automl_result.text
    _assert_exact_linear_metrics(automl_result.json()["metrics"])
    assert automl_result.json()["threshold_calibration"] is None
    saved_automl = client.post(f"/api/automl/jobs/{automl_job_id}/save")
    assert saved_automl.status_code == 200, saved_automl.text
    automl_artifact = joblib.load(tmp_path / saved_automl.json()["artifact_path"])
    assert automl_artifact["artifact_schema_version"] == 3
    assert automl_artifact["problem_type"] == "regression"

    workflow_task_id = _prepare_workflow_input(tmp_path, frame)
    events = _workflow_events(client, workflow_task_id)
    assert events
    assert all(event["status"] != "ERROR" for event in events)
    workflow_done = events[-1]
    assert workflow_done["status"] == "DONE"
    assert workflow_done["problem_spec"]["requested_type"] == "auto"
    assert workflow_done["problem_spec"]["resolved_type"] == "regression"
    assert workflow_done["suggested_type"] == "binary_classification"
    assert workflow_done["best_metric"] == "rmse"
    assert workflow_done["selected_algorithm"] == "linear_regression"
    assert math.isfinite(workflow_done["best_value"])
    assert workflow_done["best_value"] == pytest.approx(0.0, abs=1e-10)

    workflow_status = client.get(f"/api/workflow/task/{workflow_task_id}/status")
    assert workflow_status.status_code == 200, workflow_status.text
    assert workflow_status.json()["status"] == "completed"
    assert workflow_status.json()["problem_spec"]["resolved_type"] == "regression"
    workflow_artifact_path = tmp_path / "saved" / f"model_{workflow_task_id}.pkl"
    workflow_artifact = joblib.load(workflow_artifact_path)
    assert workflow_artifact["artifact_schema_version"] == 3
    assert workflow_artifact["problem_type"] == "regression"
    assert workflow_artifact["label_encoder"] is None
    (tmp_path / "output" / workflow_task_id / "workflow_state.json").unlink()
    artifact_restored_status = client.get(f"/api/workflow/task/{workflow_task_id}/status")
    assert artifact_restored_status.status_code == 200, artifact_restored_status.text
    assert artifact_restored_status.json()["status"] == "completed"
    assert artifact_restored_status.json()["problem_spec"]["resolved_type"] == "regression"

    model_ids = {
        item["model_id"]: item for item in client.get("/api/models").json()["models"]
    }
    manual_model_id = f"manual:manual_training_{manual_job_id}"
    automl_model_id = f"automl:automl_{automl_job_id}"
    workflow_model_id = f"workflow:model_{workflow_task_id}"
    assert set(model_ids) == {manual_model_id, automl_model_id, workflow_model_id}
    for model_id in (manual_model_id, automl_model_id, workflow_model_id):
        encoded_model_id = quote(model_id, safe="")
        registered = client.get(f"/api/models/{encoded_model_id}")
        assert registered.status_code == 200, registered.text
        assert registered.json()["problem_type"] == "regression"
        assert registered.json()["problem_spec"]["resolved_type"] == "regression"
        assert registered.json()["legacy"] is False

        single = client.post(
            f"/api/models/{encoded_model_id}/predict",
            json={"features": {"x": 8.0, "segment": "positive"}},
        )
        assert single.status_code == 200, single.text
        assert single.json()["prediction"] == pytest.approx(43.0, abs=0.5)
        assert single.json()["encoded_prediction"] is None
        assert single.json()["probability"] is None

        batch = client.post(
            f"/api/models/{encoded_model_id}/batch-predict",
            json={
                "rows": [
                    {"x": -4.0, "segment": "negative"},
                    {"x": 8.0, "segment": "positive"},
                ]
            },
        )
        assert batch.status_code == 200, batch.text
        assert batch.json()["summary"] == {
            "count": 2,
            "positive_rate": None,
            "average_probability": None,
        }
        assert [row["prediction"] for row in batch.json()["rows"]] == pytest.approx(
            [4.0, 43.0], abs=0.5
        )
        assert all(row["encoded_prediction"] is None for row in batch.json()["rows"])
        assert all(row["probability"] is None for row in batch.json()["rows"])

    workflow_prediction = client.post(
        "/api/workflow/predict",
        json={
            "model_version": workflow_task_id,
            "features": {"x": 8.0, "segment": "positive"},
        },
    )
    assert workflow_prediction.status_code == 200, workflow_prediction.text
    assert workflow_prediction.json()["prediction"] == pytest.approx(43.0, abs=0.5)
    assert workflow_prediction.json()["encoded_prediction"] is None
    assert workflow_prediction.json()["probability"] is None
