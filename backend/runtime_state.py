"""Durable runtime state and bounded background execution.

SQLite stores small JSON metadata while joblib stores server-generated dataframes
and fitted model objects.  Keeping these concerns behind one module makes it
straightforward to replace the local executor with Redis/Celery later without
changing the HTTP API.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path(os.getenv("ML_RUNTIME_STATE_DIR", PROJECT_ROOT / "saved" / "runtime_state")).resolve()
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class RuntimeStateStore:
    """Thread-safe SQLite metadata store with atomic local blob persistence."""

    def __init__(self, root: Path = STATE_ROOT):
        self.root = Path(root).resolve()
        self.db_path = self.root / "state.sqlite3"
        self.blob_dir = self.root / "blobs"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_state (
                    namespace TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, entity_id)
                )
                """
            )

    @staticmethod
    def _validate_name(value: str, label: str) -> None:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError(f"Invalid {label}: {value!r}")

    def put(self, namespace: str, entity_id: str, payload: Dict[str, Any]) -> None:
        self._validate_name(namespace, "namespace")
        self._validate_name(entity_id, "entity_id")
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_state(namespace, entity_id, payload, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(namespace, entity_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (namespace, entity_id, serialized),
            )

    def get(self, namespace: str, entity_id: str) -> Optional[Dict[str, Any]]:
        self._validate_name(namespace, "namespace")
        self._validate_name(entity_id, "entity_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM runtime_state WHERE namespace = ? AND entity_id = ?",
                (namespace, entity_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, namespace: str) -> List[Dict[str, Any]]:
        self._validate_name(namespace, "namespace")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM runtime_state WHERE namespace = ? ORDER BY updated_at",
                (namespace,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_blob(self, namespace: str, entity_id: str, value: Any) -> str:
        """Atomically save a trusted, server-generated Python object."""
        self._validate_name(namespace, "namespace")
        self._validate_name(entity_id, "entity_id")
        target_dir = self.blob_dir / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{entity_id}.joblib"
        temporary = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            try:
                joblib.dump(value, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return str(target.relative_to(self.root))

    def load_blob(self, namespace: str, entity_id: str) -> Any:
        self._validate_name(namespace, "namespace")
        self._validate_name(entity_id, "entity_id")
        path = self.blob_dir / namespace / f"{entity_id}.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        # The path is fixed by validated identifiers and only written by this server.
        return joblib.load(path)

    def delete(self, namespace: str, entity_id: str, blob_namespaces: tuple[str, ...] = ()) -> bool:
        """Delete one metadata record and its explicitly associated blobs."""
        self._validate_name(namespace, "namespace")
        self._validate_name(entity_id, "entity_id")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM runtime_state WHERE namespace = ? AND entity_id = ?",
                (namespace, entity_id),
            )
            deleted = cursor.rowcount > 0
        for blob_namespace in blob_namespaces:
            self._validate_name(blob_namespace, "blob_namespace")
            (self.blob_dir / blob_namespace / f"{entity_id}.joblib").unlink(missing_ok=True)
        return deleted


class BoundedTaskExecutor:
    """Shared bounded executor that prevents one thread per incoming request."""

    def __init__(self) -> None:
        try:
            workers = int(os.getenv("ML_BACKGROUND_WORKERS", "2"))
            pending = int(os.getenv("ML_MAX_PENDING_JOBS", "32"))
        except ValueError as exc:
            raise ValueError("ML_BACKGROUND_WORKERS 和 ML_MAX_PENDING_JOBS 必须是整数") from exc
        self.max_workers = max(1, min(workers, 32))
        self.max_pending = max(self.max_workers, min(pending, 1000))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ml-job")
        self._futures: Dict[str, Future[Any]] = {}
        self._lock = threading.RLock()
        self._capacity = threading.BoundedSemaphore(self.max_pending)

    def submit(self, task_key: str, function: Callable[..., Any], *args: Any) -> Future[Any]:
        with self._lock:
            existing = self._futures.get(task_key)
            if existing and not existing.done():
                return existing
            if not self._capacity.acquire(blocking=False):
                raise TaskQueueFullError(f"后台任务队列已满（上限 {self.max_pending}）")
            try:
                future = self._executor.submit(function, *args)
            except Exception:
                self._capacity.release()
                raise
            self._futures[task_key] = future
            future.add_done_callback(lambda completed, key=task_key: self._discard(key, completed))
            return future

    def _discard(self, task_key: str, future: Future[Any]) -> None:
        with self._lock:
            if self._futures.get(task_key) is future:
                self._futures.pop(task_key, None)
                self._capacity.release()

    def is_active(self, task_key: str) -> bool:
        with self._lock:
            future = self._futures.get(task_key)
            return bool(future and not future.done())


class TaskQueueFullError(RuntimeError):
    """Raised when the bounded local task queue has no remaining capacity."""


runtime_state = RuntimeStateStore()
background_tasks = BoundedTaskExecutor()
