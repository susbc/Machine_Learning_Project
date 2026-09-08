import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backend.runtime_state import BoundedTaskExecutor, RuntimeStateStore, TaskQueueFullError


def test_runtime_state_round_trip(tmp_path: Path):
    store = RuntimeStateStore(tmp_path / "runtime")

    store.put("jobs", "job123", {"status": "queued", "progress": 0})
    store.save_blob("inputs", "job123", {"rows": [1, 2, 3]})

    assert store.get("jobs", "job123") == {"status": "queued", "progress": 0}
    assert store.list("jobs") == [{"status": "queued", "progress": 0}]
    assert store.load_blob("inputs", "job123") == {"rows": [1, 2, 3]}
    assert store.delete("jobs", "job123", ("inputs",)) is True
    assert store.get("jobs", "job123") is None
    with pytest.raises(FileNotFoundError):
        store.load_blob("inputs", "job123")


def test_background_executor_respects_worker_limit(monkeypatch):
    monkeypatch.setenv("ML_BACKGROUND_WORKERS", "1")
    monkeypatch.setenv("ML_MAX_PENDING_JOBS", "2")
    executor = BoundedTaskExecutor()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first_task():
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first"

    def second_task():
        second_started.set()
        return "second"

    first = executor.submit("first", first_task)
    assert first_started.wait(timeout=1)
    second = executor.submit("second", second_task)
    assert not second_started.wait(timeout=0.1)

    release_first.set()
    assert first.result(timeout=2) == "first"
    assert second.result(timeout=2) == "second"


def test_background_executor_rejects_an_overfull_queue(monkeypatch):
    monkeypatch.setenv("ML_BACKGROUND_WORKERS", "1")
    monkeypatch.setenv("ML_MAX_PENDING_JOBS", "1")
    executor = BoundedTaskExecutor()
    started = threading.Event()
    release = threading.Event()

    def blocking_task():
        started.set()
        assert release.wait(timeout=2)

    future = executor.submit("active", blocking_task)
    assert started.wait(timeout=1)
    try:
        with pytest.raises(TaskQueueFullError, match="队列已满"):
            executor.submit("rejected", lambda: None)
    finally:
        release.set()
        future.result(timeout=2)


def test_runtime_retention_prunes_only_expired_terminal_records(tmp_path: Path, monkeypatch):
    import backend.api.automl as automl_api
    import backend.api.training as training_api

    store = RuntimeStateStore(tmp_path / "retention")
    old = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
    recent = datetime.now().isoformat(timespec="seconds")
    store.put(
        training_api.DATASET_NAMESPACE,
        "datasetold",
        {"dataset_id": "datasetold", "created_at": old},
    )
    store.save_blob(training_api.DATASET_BLOB_NAMESPACE, "datasetold", {"data": True})
    store.put(
        training_api.JOB_NAMESPACE,
        "jobold",
        {"job_id": "jobold", "status": "completed", "updated_at": old},
    )
    store.save_blob(training_api.RESULT_BLOB_NAMESPACE, "jobold", {"result": True})
    store.put(
        automl_api.JOB_NAMESPACE,
        "jobrunning",
        {"job_id": "jobrunning", "status": "running", "updated_at": old},
    )
    store.put(
        automl_api.JOB_NAMESPACE,
        "jobrecent",
        {"job_id": "jobrecent", "status": "completed", "updated_at": recent},
    )
    monkeypatch.setattr(training_api, "runtime_state", store)
    monkeypatch.setattr(automl_api, "runtime_state", store)

    assert training_api.prune_training_runtime(30) == 2
    assert automl_api.prune_automl_runtime(30) == 0
    assert store.get(training_api.DATASET_NAMESPACE, "datasetold") is None
    assert store.get(training_api.JOB_NAMESPACE, "jobold") is None
    assert store.get(automl_api.JOB_NAMESPACE, "jobrunning") is not None
    assert store.get(automl_api.JOB_NAMESPACE, "jobrecent") is not None
