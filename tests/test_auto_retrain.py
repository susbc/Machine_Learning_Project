"""自动重训链路测试：缺数据/不支持来源如实返回；完整流程重训、比指标、按结果保存。"""

import numpy as np
import pandas as pd
import pytest

from backend.api import training as training_api
from backend.runtime_state import RuntimeStateStore
from backend.services import auto_retrain
from backend.services import mlops_monitoring as monitoring
from core.automl.problem_spec import resolve_problem_spec

ARTIFACT_REL = "saved/manual_training/old.joblib"


def _binary_frame(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(0, 1, rows)
    label = ((x1 + rng.normal(0, 0.5, rows)) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


@pytest.fixture()
def retrain_env(tmp_path, monkeypatch):
    store = RuntimeStateStore(root=tmp_path / "state")
    (tmp_path / "saved" / "manual_training").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(training_api, "runtime_state", store)
    monkeypatch.setattr(training_api, "ARTIFACT_DIR", tmp_path / "saved" / "manual_training")
    monkeypatch.setattr(training_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(monitoring, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(monitoring, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(monitoring, "PREDICTION_LOG_PATH", tmp_path / "logs" / "prediction_log.jsonl")
    monkeypatch.setattr(monitoring, "ONLINE_SOURCES_PATH", tmp_path / "online_sources.json")
    training_api.training_datasets.clear()
    training_api.training_jobs.clear()
    yield store, tmp_path
    training_api.training_datasets.clear()
    training_api.training_jobs.clear()


def _seed_source_job(store, frame, spec):
    store.save_blob("training_dataset_frames", "ds1", frame)
    store.put(
        "training_datasets",
        "ds1",
        {"dataset_id": "ds1", "filename": "old.csv", "target_col": "label", "fields": [], "created_at": "2026-01-01T00:00:00"},
    )
    store.put(
        "training_jobs",
        "oldjob",
        {
            "job_id": "oldjob",
            "dataset_id": "ds1",
            "algorithm": "logistic_regression",
            "params": {},
            "status": "completed",
            "progress": 100,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "artifact_path": ARTIFACT_REL,
            "test_size": 0.3,
            "random_state": 42,
            "problem_spec": spec,
            "has_result": False,
        },
    )


def _record(spec, old_auc):
    return {
        "model_id": "manual:old",
        "artifact_path": ARTIFACT_REL,
        "metrics": {"roc_auc": old_auc},
        "best_metric": "roc_auc",
        "optimization_metric": "roc_auc",
        "problem_spec": spec,
    }


def test_auto_retrain_saves_better_model(retrain_env, monkeypatch):
    store, tmp_path = retrain_env
    frame = _binary_frame()
    spec = resolve_problem_spec(frame["label"], "label", "auto").to_dict()
    _seed_source_job(store, frame, spec)
    monkeypatch.setattr(auto_retrain, "list_model_records", lambda: [_record(spec, 0.5)])

    result = auto_retrain.run_auto_retrain("manual:old")

    assert result["status"] == "retrained"
    assert result["new_model_id"].startswith("manual:")
    saved = list((tmp_path / "saved" / "manual_training").glob("*.joblib"))
    assert len(saved) == 1
    baselines = list((tmp_path / "saved").glob("baseline_manual_*.json"))
    assert len(baselines) == 1


def test_auto_retrain_keeps_old_model_when_not_improved(retrain_env, monkeypatch):
    store, tmp_path = retrain_env
    frame = _binary_frame()
    spec = resolve_problem_spec(frame["label"], "label", "auto").to_dict()
    _seed_source_job(store, frame, spec)
    monkeypatch.setattr(auto_retrain, "list_model_records", lambda: [_record(spec, 0.999)])

    result = auto_retrain.run_auto_retrain("manual:old")

    assert result["status"] == "not_improved"
    assert list((tmp_path / "saved" / "manual_training").glob("*.joblib")) == []


def test_auto_retrain_reports_missing_labelled_data(retrain_env, monkeypatch):
    store, tmp_path = retrain_env
    frame = _binary_frame()
    spec = resolve_problem_spec(frame["label"], "label", "auto").to_dict()
    _seed_source_job(store, frame, spec)
    store.delete("training_dataset_frames", "ds1")
    store.delete("training_datasets", "ds1")
    monkeypatch.setattr(auto_retrain, "list_model_records", lambda: [_record(spec, 0.5)])

    result = auto_retrain.run_auto_retrain("manual:old")

    assert result["status"] == "no_data"


def test_auto_retrain_rejects_unsupported_source(retrain_env, monkeypatch):
    _, tmp_path = retrain_env
    record = {
        "model_id": "workflow:task_x",
        "artifact_path": "saved/model_task_x.pkl",
        "metrics": {},
        "best_metric": None,
        "optimization_metric": None,
        "problem_spec": {},
    }
    monkeypatch.setattr(auto_retrain, "list_model_records", lambda: [record])

    result = auto_retrain.run_auto_retrain("workflow:task_x")

    assert result["status"] == "unsupported"
    assert list((tmp_path / "saved" / "manual_training").glob("*.joblib")) == []
