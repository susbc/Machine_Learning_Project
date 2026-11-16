import json
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from backend.api import overview as overview_api
from backend.main import app
from backend.runtime_state import RuntimeStateStore


class _StubConnections:
    def __init__(self, **kwargs):
        self.connections = {"c1": {"db_type": "sqlite"}, "c2": {"db_type": "mysql"}}


class _StubLlm:
    def __init__(self, **kwargs):
        self.configs = {"p1": {"name": "DeepSeek"}, "p2": {"name": "Qwen"}}


def _records():
    now = datetime.now().timestamp()
    return [
        {"model_id": "m1", "problem_type": "binary_classification", "updated_at": now},
        {"model_id": "m2", "problem_type": "regression", "updated_at": now - 86400 * 9},
        {"model_id": "m3", "problem_type": None, "updated_at": now},
    ]


def _install_stubs(monkeypatch, tmp_path, store):
    monkeypatch.setattr(overview_api, "runtime_state", store)
    monkeypatch.setattr(overview_api, "list_model_records", _records)
    monkeypatch.setattr(overview_api, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(overview_api, "ConnectionManager", _StubConnections)
    monkeypatch.setattr(overview_api, "LLMManager", _StubLlm)


def test_overview_summary_aggregates_real_runtime(monkeypatch, tmp_path):
    store = RuntimeStateStore(root=tmp_path / "state")
    now = datetime.now()
    store.put(
        "training_datasets",
        "ds1",
        {"dataset_id": "ds1", "filename": "train.csv"},
    )
    store.put(
        "training_jobs",
        "job1",
        {
            "job_id": "job1",
            "dataset_id": "ds1",
            "algorithm": "lightgbm",
            "status": "running",
            "progress": 40,
            "created_at": now.isoformat(timespec="seconds"),
            "updated_at": now.isoformat(timespec="seconds"),
            "problem_spec": {"resolved_type": "binary_classification"},
        },
    )
    store.put(
        "automl_jobs",
        "job2",
        {
            "job_id": "job2",
            "status": "completed",
            "target_col": "label",
            "best_model": "xgboost",
            "progress": 100,
            "created_at": (now - timedelta(days=1)).isoformat(timespec="seconds"),
            "updated_at": (now - timedelta(days=1)).isoformat(timespec="seconds"),
            "problem_spec": {"resolved_type": "regression"},
        },
    )
    task_dir = tmp_path / "output" / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "workflow_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "selected_algorithm": "catboost",
                "problem_spec": {"resolved_type": "binary_classification"},
            }
        ),
        encoding="utf-8",
    )
    _install_stubs(monkeypatch, tmp_path, store)

    response = TestClient(app).get("/api/overview/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]["total"] == 3
    assert payload["models"]["recent_7d"] == 2
    assert payload["models"]["distribution"] == [
        {"name": "分类模型", "count": 1},
        {"name": "回归模型", "count": 1},
        {"name": "其他模型", "count": 1},
    ]
    assert payload["training"] == {"running": 1, "queued": 0, "completed": 2}
    assert payload["connections"] == {"total": 2, "db_types": ["mysql", "sqlite"]}
    assert payload["llm"] == {"total": 2, "names": ["DeepSeek", "Qwen"]}
    assert payload["workflow"] == {"total": 1, "running": 0}

    assert len(payload["trend"]) == 7
    assert payload["trend"][-1]["submitted"] == 2
    assert payload["trend"][-2]["submitted"] == 1
    assert payload["trend"][-2]["completed"] == 1

    tasks = payload["recent_tasks"]
    assert {task["name"] for task in tasks} == {"train.csv", "AutoML · label", "task_a"}
    running = next(task for task in tasks if task["status"] == "running")
    assert running["name"] == "train.csv"
    assert running["model"] == "lightgbm"
    assert running["progress"] == 40
    workflow_task = next(task for task in tasks if task["source"] == "Agent 工作流")
    assert workflow_task["model"] == "catboost"
    assert workflow_task["progress"] == 100


def test_overview_summary_handles_empty_runtime(monkeypatch, tmp_path):
    store = RuntimeStateStore(root=tmp_path / "state")

    class _EmptyRecords:
        def __call__(self):
            return []

    monkeypatch.setattr(overview_api, "runtime_state", store)
    monkeypatch.setattr(overview_api, "list_model_records", _EmptyRecords())
    monkeypatch.setattr(overview_api, "OUTPUT_DIR", tmp_path / "missing-output")
    monkeypatch.setattr(overview_api, "ConnectionManager", _StubConnections)
    monkeypatch.setattr(overview_api, "LLMManager", _StubLlm)

    response = TestClient(app).get("/api/overview/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == {"total": 0, "recent_7d": 0, "distribution": []}
    assert payload["training"] == {"running": 0, "queued": 0, "completed": 0}
    assert payload["trend"][-1]["submitted"] == 0
    assert payload["recent_tasks"] == []
