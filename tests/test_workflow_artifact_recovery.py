import json
import os
import threading
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

import backend.api.models as model_registry
import backend.api.workflow as workflow
import backend.api._workflow_artifacts as workflow_artifacts
from backend.main import app


class SimulatedProcessStop(BaseException):
    """Skip Exception rollback paths just like an abruptly terminated process."""


def _artifact(run_id: str, offset: float) -> dict:
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    model = LinearRegression().fit(frame, [offset, offset + 1, offset + 2, offset + 3])
    return {
        "run_id": run_id,
        "model": model,
        "feature_columns": ["feature"],
        "metrics": {"r2": 1.0},
    }


def _state(run_id: str, status: str = "completed") -> dict:
    return {
        "run_id": run_id,
        "status": status,
        "best_metric": "r2",
        "best_value": 1.0,
        "metrics": {"r2": 1.0},
    }


@pytest.fixture
def recovery_paths(tmp_path, monkeypatch):
    task_id = "task_a1b2c3d4"
    output_dir = tmp_path / "output"
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir(parents=True)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(workflow, "SAVED_DIR", saved_dir)
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    workflow.active_workflow_runs.clear()
    yield task_id, saved_dir / f"model_{task_id}.pkl"
    workflow.active_workflow_runs.clear()


def _start_commit(task_id: str, run_id: str) -> None:
    workflow.active_workflow_runs[task_id] = workflow.WorkflowRunControl(
        run_id=run_id,
        cancel_event=threading.Event(),
    )


def _registry_workflow_records(task_id: str) -> list[dict]:
    response = TestClient(app).get("/api/models")
    assert response.status_code == 200
    return [
        record
        for record in response.json()["models"]
        if record["model_id"] == f"workflow:model_{task_id}"
    ]


def _assert_prediction_uses_run(task_id: str, expected: float) -> None:
    response = TestClient(app).post(
        f"/api/models/workflow%3Amodel_{task_id}/predict",
        json={"features": {"feature": 10.0}},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(expected)


def _assert_batch_prediction_uses_run(task_id: str, expected: float) -> None:
    response = TestClient(app).post(
        f"/api/models/workflow%3Amodel_{task_id}/batch-predict",
        json={"rows": [{"feature": 10.0}]},
    )
    assert response.status_code == 200
    assert response.json()["rows"][0]["prediction"] == pytest.approx(expected)


def _assert_workflow_prediction_uses_run(task_id: str, expected: float) -> None:
    response = TestClient(app).post(
        "/api/workflow/predict",
        json={"model_version": task_id, "features": {"feature": 10.0}},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(expected)


def _transaction_residue(artifact_path: Path, run_id: str) -> list[Path]:
    residue = list(artifact_path.parent.glob(f".{artifact_path.name}.{run_id}.*"))
    journal = artifact_path.with_name(f".{artifact_path.name}.transaction.json")
    if journal.exists():
        residue.append(journal)
    return sorted(residue)


def test_recovery_keeps_old_canonical_if_process_stops_after_backup_before_replace(
    recovery_paths, monkeypatch
):
    task_id, artifact_path = recovery_paths
    old_run_id = "old-run"
    new_run_id = "new-run"
    joblib.dump(_artifact(old_run_id, 10.0), artifact_path)
    workflow._write_workflow_state(task_id, _state(old_run_id))
    _start_commit(task_id, new_run_id)
    original_copy = workflow_artifacts.shutil.copy2

    def stop_after_backup(source, destination, *args, **kwargs):
        result = original_copy(source, destination, *args, **kwargs)
        raise SimulatedProcessStop("after backup, before replace")

    monkeypatch.setattr(workflow_artifacts.shutil, "copy2", stop_after_backup)
    with pytest.raises(SimulatedProcessStop):
        workflow._commit_completed_run(
            task_id,
            new_run_id,
            _artifact(new_run_id, 20.0),
            artifact_path,
            _state(new_run_id),
        )

    workflow.active_workflow_runs.clear()
    monkeypatch.setattr(workflow_artifacts.shutil, "copy2", original_copy)
    status = TestClient(app).get(f"/api/workflow/task/{task_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["run_id"] == old_run_id
    assert len(_registry_workflow_records(task_id)) == 1
    assert joblib.load(artifact_path)["run_id"] == old_run_id
    _assert_workflow_prediction_uses_run(task_id, 20.0)
    _assert_prediction_uses_run(task_id, 20.0)
    assert _transaction_residue(artifact_path, new_run_id) == []


def test_recovery_restores_old_if_process_stops_after_replace_before_state_commit(
    recovery_paths, monkeypatch
):
    task_id, artifact_path = recovery_paths
    old_run_id = "old-run"
    new_run_id = "new-run"
    joblib.dump(_artifact(old_run_id, 10.0), artifact_path)
    workflow._write_workflow_state(task_id, _state(old_run_id))
    _start_commit(task_id, new_run_id)
    original_replace = os.replace

    def stop_after_publish(source, destination):
        original_replace(source, destination)
        if Path(destination) == artifact_path:
            raise SimulatedProcessStop("after replace, before state commit")

    monkeypatch.setattr(workflow.os, "replace", stop_after_publish)
    with pytest.raises(SimulatedProcessStop):
        workflow._commit_completed_run(
            task_id,
            new_run_id,
            _artifact(new_run_id, 20.0),
            artifact_path,
            _state(new_run_id),
        )

    workflow.active_workflow_runs.clear()
    monkeypatch.setattr(workflow.os, "replace", original_replace)
    assert len(_registry_workflow_records(task_id)) == 1
    assert joblib.load(artifact_path)["run_id"] == old_run_id
    assert json.loads(
        (artifact_path.parents[1] / "output" / task_id / "workflow_state.json").read_text()
    )["run_id"] == old_run_id
    _assert_prediction_uses_run(task_id, 20.0)
    assert _transaction_residue(artifact_path, new_run_id) == []


def test_recovery_keeps_new_if_process_stops_after_state_commit_before_cleanup(
    recovery_paths, monkeypatch
):
    task_id, artifact_path = recovery_paths
    old_run_id = "old-run"
    new_run_id = "new-run"
    joblib.dump(_artifact(old_run_id, 10.0), artifact_path)
    workflow._write_workflow_state(task_id, _state(old_run_id))
    _start_commit(task_id, new_run_id)
    original_unlink = Path.unlink

    def stop_before_cleanup(path, *args, **kwargs):
        if path.name == f".{artifact_path.name}.{new_run_id}.backup":
            raise SimulatedProcessStop("after state commit, before cleanup")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as crash:
        crash.setattr(Path, "unlink", stop_before_cleanup)
        with pytest.raises(SimulatedProcessStop):
            workflow._commit_completed_run(
                task_id,
                new_run_id,
                _artifact(new_run_id, 20.0),
                artifact_path,
                _state(new_run_id),
            )

    workflow.active_workflow_runs.clear()
    assert len(_registry_workflow_records(task_id)) == 1
    assert joblib.load(artifact_path)["run_id"] == new_run_id
    _assert_batch_prediction_uses_run(task_id, 30.0)
    _assert_prediction_uses_run(task_id, 30.0)
    assert _transaction_residue(artifact_path, new_run_id) == []


def test_recovery_hides_first_run_if_process_stops_after_replace_before_state_commit(
    recovery_paths, monkeypatch
):
    task_id, artifact_path = recovery_paths
    new_run_id = "first-run"
    _start_commit(task_id, new_run_id)
    original_replace = os.replace

    def stop_after_publish(source, destination):
        original_replace(source, destination)
        if Path(destination) == artifact_path:
            raise SimulatedProcessStop("first run after replace, before state commit")

    monkeypatch.setattr(workflow.os, "replace", stop_after_publish)
    with pytest.raises(SimulatedProcessStop):
        workflow._commit_completed_run(
            task_id,
            new_run_id,
            _artifact(new_run_id, 20.0),
            artifact_path,
            _state(new_run_id),
        )

    workflow.active_workflow_runs.clear()
    monkeypatch.setattr(workflow.os, "replace", original_replace)
    assert _registry_workflow_records(task_id) == []
    prediction = TestClient(app).post(
        f"/api/models/workflow%3Amodel_{task_id}/predict",
        json={"features": {"feature": 10.0}},
    )
    assert prediction.status_code == 404
    assert not artifact_path.exists()
    assert _transaction_residue(artifact_path, new_run_id) == []


def test_registry_hides_old_artifact_without_persisted_completed_evidence(recovery_paths):
    task_id, artifact_path = recovery_paths
    joblib.dump(_artifact("old-run", 10.0), artifact_path)
    workflow._write_workflow_state(task_id, _state("new-run", "running"))

    assert _registry_workflow_records(task_id) == []


def test_running_state_persists_last_known_good_completed_evidence(recovery_paths):
    task_id, artifact_path = recovery_paths
    old_state = _state("old-run")
    joblib.dump(_artifact("old-run", 10.0), artifact_path)
    workflow._write_workflow_state(task_id, old_state)

    control = workflow._claim_workflow_run(task_id)
    workflow._write_run_state(task_id, control.run_id, {"status": "running"})
    persisted = json.loads(
        (artifact_path.parents[1] / "output" / task_id / "workflow_state.json").read_text()
    )
    workflow.active_workflow_runs.clear()

    assert persisted["committed_artifact_state"]["status"] == "completed"
    assert persisted["committed_artifact_state"]["run_id"] == "old-run"
    assert len(_registry_workflow_records(task_id)) == 1
    assert joblib.load(artifact_path)["run_id"] == "old-run"
