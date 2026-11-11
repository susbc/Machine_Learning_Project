import sqlite3

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.exc import OperationalError

from backend.api.database import get_connector
from backend.main import app
import backend.api.prediction_exports as prediction_exports
from model_code.database.connection_manager import ConnectionManager


def _export_rows():
    return [
        {
            "index": 0,
            "features": {"age": 30, "city": "北京"},
            "prediction": "正常",
            "encoded_prediction": 0,
            "probability": 0.4,
            "diagnostics": {"unknown_input_keys": [], "missing_columns_imputed": ["city"]},
        },
        {
            "index": 1,
            "features": {"age": 70},
            "prediction": "违约",
            "encoded_prediction": 1,
            "probability": 0.84,
            "diagnostics": None,
        },
    ]


def _sqlite_connection_id(tmp_path, monkeypatch, database: str) -> str:
    storage = tmp_path / "connections.json"
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    manager = ConnectionManager(storage_path=str(storage))
    return manager.add_connection(name="export-sqlite", config={"db_type": "sqlite", "database": database})


def test_file_export_writes_csv_to_default_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_exports, "ARTIFACT_ROOT", tmp_path)

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={"model_id": "manual:m1", "target": "file", "rows": _export_rows()},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["exported"] is True
    assert payload["row_count"] == 2
    assert payload["path"].startswith("output/")
    written = tmp_path / payload["path"]
    assert written.parent == tmp_path / "output"
    lines = written.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "row_index,age,city,prediction,encoded_prediction,probability"
    assert lines[1].startswith("0,30,北京,正常,0.0,0.4")
    assert "违约" in lines[2]


def test_file_export_honors_custom_relative_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_exports, "ARTIFACT_ROOT", tmp_path)

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={
            "model_id": "manual:m1",
            "target": "file",
            "rows": _export_rows(),
            "output_dir": "exports/nov",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"].startswith("exports/nov/")
    assert (tmp_path / "exports" / "nov").is_dir()


@pytest.mark.parametrize("output_dir", ["../../outside", "/etc"])
def test_file_export_rejects_dirs_outside_project(tmp_path, monkeypatch, output_dir):
    monkeypatch.setattr(prediction_exports, "ARTIFACT_ROOT", tmp_path)

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={
            "model_id": "manual:m1",
            "target": "file",
            "rows": _export_rows(),
            "output_dir": output_dir,
        },
    )

    assert response.status_code == 400
    assert not (tmp_path / "outside").exists()


def test_database_export_creates_default_table_and_appends(tmp_path, monkeypatch):
    db_file = tmp_path / "export.db"
    connection_id = _sqlite_connection_id(tmp_path, monkeypatch, str(db_file))
    client = TestClient(app)
    payload = {"model_id": "manual:m1", "target": "database", "connection_id": connection_id, "rows": _export_rows()}

    response = client.post("/api/prediction-exports", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["exported"] is True
    assert result["table_name"] == "prediction_results"
    assert result["table_created"] is True
    assert result["row_count"] == 2

    with sqlite3.connect(db_file) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(prediction_results)")]
        rows = conn.execute(
            "SELECT row_index, features_json, prediction, probability, diagnostics_json"
            " FROM prediction_results ORDER BY row_index"
        ).fetchall()
    assert columns == [
        "id",
        "exported_at",
        "model_id",
        "row_index",
        "features_json",
        "prediction",
        "encoded_prediction",
        "probability",
        "diagnostics_json",
    ]
    assert [row[0] for row in rows] == [0, 1]
    assert "北京" in rows[0][1]
    assert rows[0][2] == "正常"
    assert rows[0][4] is not None
    assert rows[1][4] is None

    second = client.post("/api/prediction-exports", json=payload)
    assert second.status_code == 200
    assert second.json()["table_created"] is False
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0]
    assert count == 4


def test_database_export_honors_custom_table_name(tmp_path, monkeypatch):
    db_file = tmp_path / "export.db"
    connection_id = _sqlite_connection_id(tmp_path, monkeypatch, str(db_file))

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={
            "model_id": "manual:m1",
            "target": "database",
            "connection_id": connection_id,
            "table_name": "my_exports",
            "rows": _export_rows(),
        },
    )

    assert response.status_code == 200
    assert response.json()["table_name"] == "my_exports"
    with sqlite3.connect(db_file) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        count = conn.execute("SELECT COUNT(*) FROM my_exports").fetchone()[0]
    assert "my_exports" in tables
    assert count == 2


def test_database_export_rejects_invalid_table_name(tmp_path, monkeypatch):
    db_file = tmp_path / "export.db"
    connection_id = _sqlite_connection_id(tmp_path, monkeypatch, str(db_file))

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={
            "model_id": "manual:m1",
            "target": "database",
            "connection_id": connection_id,
            "table_name": "bad name;",
            "rows": _export_rows(),
        },
    )

    assert response.status_code == 400


def test_database_export_requires_connection_id(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_exports, "ARTIFACT_ROOT", tmp_path)

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={"model_id": "manual:m1", "target": "database", "rows": _export_rows()},
    )

    assert response.status_code == 400


def test_database_export_reports_privilege_warning(tmp_path, monkeypatch):
    db_file = tmp_path / "export.db"
    connection_id = _sqlite_connection_id(tmp_path, monkeypatch, str(db_file))
    connector = get_connector(connection_id)
    assert connector.connect() is True

    def _deny(*args, **kwargs):
        raise OperationalError(
            "INSERT INTO prediction_results",
            {},
            Exception(1142, "INSERT command denied to user 'app'@'host' for table 'prediction_results'"),
        )

    monkeypatch.setattr(connector.engine, "begin", _deny)
    monkeypatch.setattr(prediction_exports, "get_connector", lambda _connection_id: connector)

    response = TestClient(app).post(
        "/api/prediction-exports",
        json={
            "model_id": "manual:m1",
            "target": "database",
            "connection_id": connection_id,
            "rows": _export_rows(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["exported"] is False
    assert payload["status"] == "permission_denied"
    assert payload["row_count"] == 0
    assert payload["warnings"]
    assert "数据库权限不足" in payload["warnings"][0]
    with sqlite3.connect(db_file) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables == []


def test_privilege_warning_classifies_driver_errors():
    class PgOrig(Exception):
        pgcode = "42501"

    postgres = OperationalError("stmt", {}, PgOrig("permission denied for table prediction_results"))
    mysql = OperationalError("stmt", {}, Exception(1142, "INSERT command denied to user"))
    readonly = OperationalError("stmt", {}, Exception("attempt to write a readonly database"))
    plain = OperationalError("stmt", {}, Exception("no such table: missing"))

    assert prediction_exports._privilege_warning(postgres)
    assert prediction_exports._privilege_warning(mysql)
    assert prediction_exports._privilege_warning(readonly)
    assert prediction_exports._privilege_warning(plain) is None
