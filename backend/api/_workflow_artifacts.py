"""Crash-safe publication and reconciliation for Agent workflow artifacts.

This module deliberately has no dependency on either API router.  Both the
workflow endpoints and the unified registry can therefore enforce the same
publication visibility rules without introducing a router import cycle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib

logger = logging.getLogger(__name__)

TRANSACTION_SCHEMA_VERSION = 1
WORKFLOW_STATE_FILENAME = "workflow_state.json"
COMMITTED_ARTIFACT_STATE_KEY = "committed_artifact_state"
artifact_transaction_lock = threading.RLock()


@dataclass(frozen=True)
class WorkflowArtifactPaths:
    task_id: str
    artifact: Path
    state: Path
    journal: Path
    staged: Path
    backup: Path


def workflow_artifact_path(project_root: Path, task_id: str) -> Path:
    return project_root / "saved" / f"model_{task_id}.pkl"


def workflow_state_path(project_root: Path, task_id: str) -> Path:
    return project_root / "output" / task_id / WORKFLOW_STATE_FILENAME


def workflow_journal_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f".{artifact_path.name}.transaction.json")


def workflow_transaction_paths(artifact_path: Path, state_path: Path, task_id: str, run_id: str) -> WorkflowArtifactPaths:
    return WorkflowArtifactPaths(
        task_id=task_id,
        artifact=artifact_path,
        state=state_path,
        journal=workflow_journal_path(artifact_path),
        staged=artifact_path.with_name(f".{artifact_path.name}.{run_id}.staged"),
        backup=artifact_path.with_name(f".{artifact_path.name}.{run_id}.backup"),
    )


def _fsync_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def artifact_run_id(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    if not isinstance(artifact, dict):
        return None
    run_id = artifact.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def committed_completed_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not state:
        return None
    if state.get("status") == "completed" and isinstance(state.get("run_id"), str):
        return state
    previous = state.get(COMMITTED_ARTIFACT_STATE_KEY)
    if (
        isinstance(previous, dict)
        and previous.get("status") == "completed"
        and isinstance(previous.get("run_id"), str)
    ):
        return previous
    return None


def _journal_payload(
    paths: WorkflowArtifactPaths,
    run_id: str,
    previous_completed_state: Optional[Dict[str, Any]],
    old_run_id: Optional[str],
    stage: str,
) -> Dict[str, Any]:
    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "task_id": paths.task_id,
        "new_run_id": run_id,
        "old_run_id": old_run_id,
        "artifact_name": paths.artifact.name,
        "staged_name": paths.staged.name,
        "backup_name": paths.backup.name,
        "stage": stage,
        "previous_completed_state": previous_completed_state,
    }


def update_transaction_stage(paths: WorkflowArtifactPaths, payload: Dict[str, Any], stage: str) -> Dict[str, Any]:
    updated = {**payload, "stage": stage}
    write_json_atomic(paths.journal, updated)
    return updated


def begin_artifact_transaction(
    paths: WorkflowArtifactPaths,
    run_id: str,
    previous_completed_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Journal the transaction and take a durable copy of the old canonical."""
    old_run_id = artifact_run_id(paths.artifact)
    payload = _journal_payload(
        paths,
        run_id,
        previous_completed_state,
        old_run_id,
        "prepared",
    )
    write_json_atomic(paths.journal, payload)
    if paths.artifact.is_file():
        shutil.copy2(paths.artifact, paths.backup)
        _fsync_file(paths.backup)
        _fsync_directory(paths.backup.parent)
    return update_transaction_stage(paths, payload, "backed_up")


def stage_workflow_artifact(paths: WorkflowArtifactPaths, artifact: Dict[str, Any]) -> None:
    paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, paths.staged)
    _fsync_file(paths.staged)
    _fsync_directory(paths.staged.parent)


def publish_staged_artifact(paths: WorkflowArtifactPaths, payload: Dict[str, Any]) -> Dict[str, Any]:
    os.replace(paths.staged, paths.artifact)
    _fsync_directory(paths.artifact.parent)
    return update_transaction_stage(paths, payload, "published")


def cleanup_artifact_transaction(paths: WorkflowArtifactPaths) -> None:
    """Delete recoverable material last; leave the journal if any cleanup fails."""
    for path in (paths.staged, paths.backup):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("工作流 artifact 事务清理失败 task_id=%s file=%s", paths.task_id, path.name)
            return
    paths.journal.unlink(missing_ok=True)
    _fsync_directory(paths.artifact.parent)


def _paths_from_journal(
    artifact: Path,
    state: Path,
    task_id: str,
    journal: Dict[str, Any],
) -> Optional[WorkflowArtifactPaths]:
    if (
        journal.get("schema_version") != TRANSACTION_SCHEMA_VERSION
        or journal.get("task_id") != task_id
        or not isinstance(journal.get("new_run_id"), str)
    ):
        return None
    expected = workflow_transaction_paths(
        artifact,
        state,
        task_id,
        journal["new_run_id"],
    )
    if (
        journal.get("artifact_name") != expected.artifact.name
        or journal.get("staged_name") != expected.staged.name
        or journal.get("backup_name") != expected.backup.name
    ):
        return None
    return expected


def _restore_previous_state(paths: WorkflowArtifactPaths, journal: Dict[str, Any]) -> None:
    previous = journal.get("previous_completed_state")
    if isinstance(previous, dict) and previous.get("status") == "completed":
        write_json_atomic(paths.state, previous)


def _mark_interrupted_first_run(paths: WorkflowArtifactPaths, journal: Dict[str, Any]) -> None:
    current = _read_json(paths.state)
    new_run_id = journal.get("new_run_id")
    if current and current.get("run_id") == new_run_id and current.get("status") != "completed":
        write_json_atomic(
            paths.state,
            {
                **current,
                "status": "failed",
                "error": "训练发布在服务重启时中断，请重新运行",
            },
        )


def reconcile_workflow_artifact_paths(artifact: Path, state_path: Path, task_id: str) -> bool:
    """Recover one interrupted publication and report canonical visibility."""
    journal_path = workflow_journal_path(artifact)
    with artifact_transaction_lock:
        journal = _read_json(journal_path)
        if journal is not None:
            paths = _paths_from_journal(artifact, state_path, task_id, journal)
            if paths is None:
                logger.error("拒绝损坏的工作流 artifact 事务 task_id=%s", task_id)
                return False
            canonical_run_id = artifact_run_id(paths.artifact)
            state = _read_json(paths.state)
            state_run_id = state.get("run_id") if state else None
            committed = (
                canonical_run_id is not None
                and state is not None
                and state.get("status") == "completed"
                and state_run_id == canonical_run_id
            )
            if committed:
                cleanup_artifact_transaction(paths)
                return True

            old_run_id = journal.get("old_run_id")
            if paths.backup.is_file() and artifact_run_id(paths.backup) == old_run_id:
                os.replace(paths.backup, paths.artifact)
                _fsync_directory(paths.artifact.parent)
                _restore_previous_state(paths, journal)
            elif old_run_id is not None and canonical_run_id == old_run_id:
                _restore_previous_state(paths, journal)
            else:
                if canonical_run_id == journal.get("new_run_id"):
                    paths.artifact.unlink(missing_ok=True)
                    _fsync_directory(paths.artifact.parent)
                _mark_interrupted_first_run(paths, journal)
            cleanup_artifact_transaction(paths)

        canonical_run_id = artifact_run_id(artifact)
        if not artifact.is_file():
            return False
        # Artifacts without a run id predate transactional workflow publication.
        if canonical_run_id is None:
            return True
        state = _read_json(state_path)
        completed_state = committed_completed_state(state)
        if completed_state is None:
            return False
        return completed_state.get("run_id") == canonical_run_id


def reconcile_workflow_artifact(project_root: Path, task_id: str) -> bool:
    return reconcile_workflow_artifact_paths(
        workflow_artifact_path(project_root, task_id),
        workflow_state_path(project_root, task_id),
        task_id,
    )


def _task_id_from_artifact_name(name: str) -> Optional[str]:
    if not name.startswith("model_") or not name.endswith(".pkl"):
        return None
    return name[len("model_") : -len(".pkl")]


def iter_visible_workflow_artifact_paths(saved: Path, output: Path) -> Iterable[Path]:
    if not saved.is_dir():
        return []
    task_ids = {
        task_id
        for path in saved.glob("model_task_*.pkl")
        if (task_id := _task_id_from_artifact_name(path.name)) is not None
    }
    for journal in saved.glob(".model_task_*.pkl.transaction.json"):
        artifact_name = journal.name[1 : -len(".transaction.json")]
        task_id = _task_id_from_artifact_name(artifact_name)
        if task_id is not None:
            task_ids.add(task_id)
    visible = []
    for task_id in sorted(task_ids):
        artifact = saved / f"model_{task_id}.pkl"
        state = output / task_id / WORKFLOW_STATE_FILENAME
        if reconcile_workflow_artifact_paths(artifact, state, task_id) and artifact.is_file():
            visible.append(artifact)
    return visible


def iter_visible_workflow_artifacts(project_root: Path) -> Iterable[Path]:
    return iter_visible_workflow_artifact_paths(
        project_root / "saved",
        project_root / "output",
    )
