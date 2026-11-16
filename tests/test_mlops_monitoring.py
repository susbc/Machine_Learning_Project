"""MLOps 真实数据监控链路测试：预测日志、基线回填、漂移巡检与 API。"""

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import mlops as mlops_api
from backend.main import app
from backend.runtime_state import RuntimeStateStore
from backend.services import mlops_monitoring as monitoring

import backend.scheduler as scheduler


@pytest.fixture()
def monitoring_dirs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    saved_dir = tmp_path / "saved"
    logs_dir.mkdir()
    saved_dir.mkdir()
    monkeypatch.setattr(monitoring, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(monitoring, "SAVED_DIR", saved_dir)
    monkeypatch.setattr(monitoring, "PREDICTION_LOG_PATH", logs_dir / "prediction_log.jsonl")
    return tmp_path


def _train_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({"age": rng.normal(35, 5, 600), "label": rng.integers(0, 2, 600)})


def test_prediction_log_roundtrip(monitoring_dirs):
    monitoring.append_predictions("manual:m1", [{"age": 40}, {"age": 41}])
    monitoring.append_predictions("manual:m2", [{"age": 50}])

    assert monitoring.prediction_counts() == {"manual:m1": 2, "manual:m2": 1}
    frame = monitoring.online_frame("manual:m1", days=7)
    assert len(frame) == 2
    assert list(frame["age"]) == [40, 41]
    # 只取指定模型：m2 的日志不会串入 m1
    assert len(monitoring.online_frame("manual:m2", days=7)) == 1


def test_prediction_log_prunes_old_and_oversized_entries(monitoring_dirs, monkeypatch):
    old_stamp = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
    monitoring.append_predictions("manual:new", [{"a": 2}, {"a": 3}])
    with monitoring.PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        for _ in range(3):
            handle.write(json.dumps({"ts": old_stamp, "model_id": "manual:old", "features": {"a": 1}}) + "\n")
        handle.write("corrupted-line\n")

    removed = monitoring.prune_prediction_log()

    assert removed == 4
    remaining = monitoring.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(remaining) == 2
    assert all(json.loads(line)["model_id"] == "manual:new" for line in remaining)

    monkeypatch.setattr(monitoring, "PREDICTION_LOG_MAX_BYTES", 1)
    for i in range(8):
        monitoring.append_predictions("manual:new", [{"a": i}])
    remaining = monitoring.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert 0 < len(remaining) < 10


def test_inspect_model_reports_real_drift(monitoring_dirs):
    monitoring.write_baseline_for_saved_model("manual:m1", _train_frame(), "label")

    result = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")
    assert result["status"] == "insufficient_online"

    drifted = [{"age": float(60 + i * 0.1)} for i in range(monitoring.MIN_ONLINE_ROWS + 10)]
    monitoring.append_predictions("manual:m1", drifted)

    result = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")
    assert result["status"] == "ok"
    summary = result["report"]["summary"]
    assert summary["total_features"] == 1
    assert summary["drifted_features"] == 1
    assert summary["needs_retrain"] is True
    assert monitoring.report_paths("manual:m1")


def test_inspect_model_without_baseline_is_honest(monitoring_dirs):
    result = monitoring.inspect_model("manual:ghost", days=7, source="手动巡检")
    assert result["status"] == "no_baseline"
    assert not monitoring.report_paths("manual:ghost")


def test_backfill_baseline_from_training_runtime(monitoring_dirs, tmp_path):
    store = RuntimeStateStore(root=tmp_path / "state")
    frame = _train_frame()
    frame.insert(0, "user_id", range(1, len(frame) + 1))
    store.save_blob("training_dataset_frames", "ds1", frame)
    store.put(
        "training_datasets",
        "ds1",
        {
            "dataset_id": "ds1",
            "target_col": "label",
            "fields": [
                {"field_name": "user_id", "field_type": "key"},
                {"field_name": "age", "field_type": "num"},
                {"field_name": "label", "field_type": "label"},
            ],
        },
    )
    store.put(
        "training_jobs",
        "job1",
        {"job_id": "job1", "dataset_id": "ds1", "artifact_path": "saved/manual_training/m1.joblib"},
    )
    record = {"model_id": "manual:m1", "artifact_path": "saved/manual_training/m1.joblib"}

    path = monitoring.backfill_baseline(record, store=store)

    assert path is not None and path.is_file()
    baseline = json.loads(path.read_text(encoding="utf-8"))
    assert "age" in baseline
    assert "user_id" not in baseline, "key 列不应进入基线（ID 漂移是必然噪声）"
    assert baseline["age_stats"]["type"] == "numeric"
    # 第二次调用直接命中已有基线，不重复写入
    assert monitoring.backfill_baseline(record, store=store) == path


def test_inspect_reports_feature_mismatch_instead_of_fake_healthy(monitoring_dirs):
    monitoring.write_baseline_for_saved_model("manual:m1", _train_frame(), "label")
    unrelated = [{"noise_a": float(i), "noise_b": float(i % 5)} for i in range(monitoring.MIN_ONLINE_ROWS + 5)]
    monitoring.append_predictions("manual:m1", unrelated)

    result = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")

    assert result["status"] == "feature_mismatch"
    assert monitoring.report_paths("manual:m1") == []
    assert monitoring.health_status("manual:m1")[0] == "待巡检"


def _seed_ready_model(monitoring_dirs):
    monitoring.write_baseline_for_saved_model("manual:m1", _train_frame(), "label")
    drifted = [{"age": float(60 + i * 0.1)} for i in range(monitoring.MIN_ONLINE_ROWS + 5)]
    monitoring.append_predictions("manual:m1", drifted)


def test_mlops_api_serves_real_inspection_data(monitoring_dirs, monkeypatch):
    _seed_ready_model(monitoring_dirs)
    record = {
        "model_id": "manual:m1",
        "artifact_path": "saved/manual_training/m1.joblib",
        "metrics": {"roc_auc": 0.9123},
        "optimization_metric": "roc_auc",
        "best_metric": None,
    }
    monkeypatch.setattr(mlops_api, "list_model_records", lambda: [record])
    monkeypatch.setattr(monitoring, "backfill_baseline", lambda rec, store=None: None)
    client = TestClient(app)

    inspect_response = client.post("/api/mlops/inspect", json={"model_id": "manual:m1", "days": 7})
    assert inspect_response.status_code == 200
    payload = inspect_response.json()
    assert payload["inspected"] == 1
    assert payload["reports"] == 1
    assert payload["results"][0]["status"] == "ok"

    drift = client.get("/api/mlops/drift-report", params={"model_version": "manual:m1"}).json()
    assert drift["summary"]["total_features"] == 1
    assert drift["summary"]["needs_retrain"] is True
    assert "age" in drift["features"]

    health = client.get("/api/mlops/health").json()
    assert len(health["models"]) == 1
    row = health["models"][0]
    assert row["模型"] == "manual:m1"
    assert row["最佳指标"] == 0.9123
    assert row["指标名"] == "roc_auc"
    assert row["健康度"] == "需重训"
    assert row["调用次数"] == monitoring.MIN_ONLINE_ROWS + 5

    logs = client.get("/api/mlops/logs", params={"days": 7}).json()["logs"]
    assert logs
    assert logs[0]["事件"] == "手动巡检"
    assert logs[0]["结果"] == "重训"
    assert "manual:m1" in logs[0]["详情"]


def test_mlops_api_drift_report_without_any_data(monitoring_dirs, monkeypatch):
    monkeypatch.setattr(mlops_api, "list_model_records", lambda: [])

    payload = TestClient(app).get("/api/mlops/drift-report", params={"model_version": "manual:none"}).json()

    assert payload["summary"]["total_features"] == 0
    assert "未找到基线数据或巡检报告" in payload["message"]


def test_scheduler_drift_job_alerts_and_schedules_retrain(monkeypatch):
    alerts = []
    retrain_calls = []

    async def fake_send_alert(title, message, level="info", drift_report=None):
        alerts.append({"title": title, "level": level})

    async def fake_schedule_retrain(task_id):
        retrain_calls.append(task_id)

    monkeypatch.setattr(scheduler, "list_model_records", lambda: [{"model_id": "manual:m1"}])
    monkeypatch.setattr(scheduler.monitoring, "backfill_baseline", lambda record, store=None: None)
    monkeypatch.setattr(
        scheduler.monitoring,
        "inspect_model",
        lambda model_id, days, source: {
            "model_id": model_id,
            "status": "ok",
            "message": "巡检完成",
            "report": {
                "summary": {
                    "total_features": 2,
                    "drifted_features": 2,
                    "critical_features": 1,
                    "needs_retrain": True,
                }
            },
        },
    )
    monkeypatch.setattr(scheduler, "send_alert", fake_send_alert)
    monkeypatch.setattr(scheduler, "schedule_auto_retrain", fake_schedule_retrain)

    asyncio.run(scheduler.check_data_drift_job())

    assert alerts and alerts[0]["level"] == "critical"
    assert retrain_calls == ["manual:m1"]


def test_scheduler_drift_job_skips_without_models(monkeypatch):
    alerts = []

    async def fake_send_alert(title, message, level="info", drift_report=None):
        alerts.append(level)

    monkeypatch.setattr(scheduler, "list_model_records", lambda: [])
    monkeypatch.setattr(scheduler, "send_alert", fake_send_alert)

    asyncio.run(scheduler.check_data_drift_job())

    assert alerts == []


def _sqlite_online_db(tmp_path, rows: int) -> str:
    import sqlite3

    db_file = tmp_path / "online.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE prod (age REAL, label INTEGER, created_at TEXT)")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        "INSERT INTO prod VALUES (?, ?, ?)",
        [(60.0 + i * 0.5, i % 2, stamp) for i in range(rows)],
    )
    conn.commit()
    conn.close()
    return str(db_file)


def test_inspect_uses_bound_database_table_as_online_source(monitoring_dirs, tmp_path, monkeypatch):
    from model_code.database.connection_manager import ConnectionManager

    db_file = _sqlite_online_db(tmp_path, monitoring.MIN_ONLINE_ROWS + 20)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(tmp_path / "connections.json"))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    monkeypatch.setattr(monitoring, "ONLINE_SOURCES_PATH", tmp_path / "online_sources.json")
    manager = ConnectionManager(storage_path=str(tmp_path / "connections.json"))
    connection_id = manager.add_connection(name="online", config={"db_type": "sqlite", "database": db_file})

    monitoring.write_baseline_for_saved_model("manual:m1", _train_frame(), "label")
    monitoring.set_online_source(
        "manual:m1",
        {"connection_id": connection_id, "table_name": "prod", "time_column": "created_at"},
    )
    assert monitoring.get_online_source("manual:m1")["table_name"] == "prod"

    result = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")

    assert result["status"] == "ok"
    assert result["report"]["summary"]["total_features"] == 1
    assert result["report"]["summary"]["drifted_features"] == 1

    monitoring.clear_online_source("manual:m1")
    assert monitoring.get_online_source("manual:m1") is None


def test_online_source_binding_api_validates_table(monitoring_dirs, tmp_path, monkeypatch):
    from model_code.database.connection_manager import ConnectionManager

    db_file = _sqlite_online_db(tmp_path, 5)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(tmp_path / "connections.json"))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    monkeypatch.setattr(monitoring, "ONLINE_SOURCES_PATH", tmp_path / "online_sources.json")
    manager = ConnectionManager(storage_path=str(tmp_path / "connections.json"))
    connection_id = manager.add_connection(name="online", config={"db_type": "sqlite", "database": db_file})
    client = TestClient(app)

    bound = client.put(
        "/api/mlops/online-sources/manual%3Am1",
        json={"connection_id": connection_id, "table_name": "prod"},
    )
    assert bound.status_code == 200
    assert bound.json()["source"]["table_name"] == "prod"
    assert client.get("/api/mlops/online-sources").json()["sources"]["manual:m1"]["connection_id"] == connection_id

    missing = client.put(
        "/api/mlops/online-sources/manual%3Am1",
        json={"connection_id": connection_id, "table_name": "no_such_table"},
    )
    assert missing.status_code == 404

    invalid = client.put(
        "/api/mlops/online-sources/manual%3Am1",
        json={"connection_id": connection_id, "table_name": "bad name"},
    )
    assert invalid.status_code == 400

    assert client.delete("/api/mlops/online-sources/manual%3Am1").status_code == 204
    assert client.get("/api/mlops/online-sources").json()["sources"] == {}


def test_psi_thresholds_configurable(monitoring_dirs, monkeypatch):
    monkeypatch.setattr(monitoring, "MLOPS_SETTINGS_PATH", monitoring_dirs / "settings.json")

    assert monitoring.get_psi_thresholds() == (0.1, 0.2)
    monitoring.set_psi_thresholds(10.0, 20.0)
    assert monitoring.get_psi_thresholds() == (10.0, 20.0)
    with pytest.raises(ValueError):
        monitoring.set_psi_thresholds(0.5, 0.5)
    # 损坏的配置文件回退默认阈值
    (monitoring_dirs / "settings.json").write_text("{broken", encoding="utf-8")
    assert monitoring.get_psi_thresholds() == (0.1, 0.2)


def test_inspect_respects_custom_thresholds(monitoring_dirs, monkeypatch):
    monkeypatch.setattr(monitoring, "MLOPS_SETTINGS_PATH", monitoring_dirs / "settings.json")
    monitoring.write_baseline_for_saved_model("manual:m1", _train_frame(), "label")
    drifted = [{"age": float(60 + i * 0.1)} for i in range(monitoring.MIN_ONLINE_ROWS + 5)]
    monitoring.append_predictions("manual:m1", drifted)

    default_report = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")["report"]
    assert default_report["summary"]["drifted_features"] == 1
    assert default_report["thresholds"] == {"low": 0.1, "high": 0.2}

    monitoring.set_psi_thresholds(10.0, 20.0)
    relaxed = monitoring.inspect_model("manual:m1", days=7, source="手动巡检")["report"]
    assert relaxed["summary"]["drifted_features"] == 0
    assert relaxed["thresholds"] == {"low": 10.0, "high": 20.0}


def test_settings_api_validates_and_roundtrips(monitoring_dirs, monkeypatch):
    monkeypatch.setattr(monitoring, "MLOPS_SETTINGS_PATH", monitoring_dirs / "settings.json")
    client = TestClient(app)

    assert client.get("/api/mlops/settings").json() == {
        "psi_threshold_low": 0.1,
        "psi_threshold_high": 0.2,
    }
    ok = client.put("/api/mlops/settings", json={"psi_threshold_low": 0.3, "psi_threshold_high": 0.6})
    assert ok.status_code == 200
    assert client.get("/api/mlops/settings").json()["psi_threshold_low"] == 0.3

    reversed_order = client.put("/api/mlops/settings", json={"psi_threshold_low": 0.6, "psi_threshold_high": 0.3})
    assert reversed_order.status_code == 422


def test_catchup_inspection_only_covers_models_without_today_report(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "list_model_records", lambda: [{"model_id": "m1"}, {"model_id": "m2"}])
    monkeypatch.setattr(scheduler.monitoring, "today_report_exists", lambda model_id: model_id == "m1")
    monkeypatch.setattr(scheduler.monitoring, "backfill_baseline", lambda record, store=None: None)

    def fake_inspect(model_id, days, source):
        calls.append((model_id, source))
        return {"model_id": model_id, "status": "insufficient_online", "message": "样本不足"}

    monkeypatch.setattr(scheduler.monitoring, "inspect_model", fake_inspect)

    asyncio.run(scheduler.catchup_inspection_job())

    assert calls == [("m2", "补跑巡检")]
