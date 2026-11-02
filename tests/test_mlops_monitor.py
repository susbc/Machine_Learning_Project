"""DataDriftMonitor 回归测试。

覆盖三类历史缺陷：
1. calculate_psi 对 pandas Series 输入抛 AttributeError（np.array 后无 dropna）；
2. 线上取值超出训练分箱边界时被 np.histogram 静默丢弃，漂移被低估；
3. 类别特征基线只有 {col}_dist 键，generate_drift_report 原循环将其跳过，
   类别漂移永远漏检。

另覆盖 scheduler.auto_retrain_job 的"假成功"修复：数据源未接入时必须
以跳过告警收尾，绝不能报告"重训完成"。
"""

import asyncio

import numpy as np
import pandas as pd

from model_code.mlops_monitor import DataDriftMonitor


def test_calculate_psi_accepts_pandas_series():
    monitor = DataDriftMonitor()
    expected = pd.Series(np.random.default_rng(0).normal(0, 1, 500))
    actual = pd.Series(np.random.default_rng(1).normal(0, 1, 500))

    psi = monitor.calculate_psi(expected, actual)

    assert psi < 0.1


def test_calculate_psi_counts_out_of_range_samples():
    monitor = DataDriftMonitor()
    expected = np.linspace(0, 1, 1000)
    actual = np.full(1000, 100.0)  # 全部超出训练范围

    psi = monitor.calculate_psi(expected, actual)

    assert psi >= 0.2


def test_calculate_psi_zero_for_identical_distribution():
    monitor = DataDriftMonitor()
    expected = np.linspace(0, 1, 1000)

    assert monitor.calculate_psi(expected, expected.copy()) == 0.0


def test_categorical_drift_is_evaluated():
    monitor = DataDriftMonitor()
    df_train = pd.DataFrame({"city": ["北京"] * 500 + ["上海"] * 500})
    baseline = monitor.extract_baseline(df_train, ["city"])
    df_online = pd.DataFrame({"city": ["深圳"] * 900 + ["北京"] * 100})

    report = monitor.generate_drift_report(baseline, df_online)

    assert report["summary"]["total_features"] == 1
    assert report["summary"]["drifted_features"] == 1
    assert report["features"]["city"]["severity"] == "critical"
    assert report["summary"]["needs_retrain"] is True


def test_stable_categorical_feature_stays_normal():
    monitor = DataDriftMonitor()
    df_train = pd.DataFrame({"city": ["北京"] * 500 + ["上海"] * 500})
    baseline = monitor.extract_baseline(df_train, ["city"])

    report = monitor.generate_drift_report(baseline, df_train.copy())

    assert report["summary"]["total_features"] == 1
    assert report["summary"]["drifted_features"] == 0
    assert report["summary"]["needs_retrain"] is False


def test_mixed_numeric_and_categorical_features_all_counted():
    monitor = DataDriftMonitor()
    df_train = pd.DataFrame(
        {
            "age": np.random.default_rng(2).integers(20, 60, 1000),
            "city": ["北京"] * 500 + ["上海"] * 500,
        }
    )
    baseline = monitor.extract_baseline(df_train, ["age", "city"])

    report = monitor.generate_drift_report(baseline, df_train.copy())

    assert report["summary"]["total_features"] == 2


def test_auto_retrain_skips_honestly_without_data_source(monkeypatch):
    import backend.scheduler as scheduler

    alerts = []

    async def fake_send_alert(title, message, level="info", drift_report=None):
        alerts.append({"title": title, "message": message, "level": level})

    monkeypatch.setattr(scheduler, "send_alert", fake_send_alert)

    asyncio.run(scheduler.auto_retrain_job("task_deadbeef"))

    assert alerts, "数据源未接入时必须发出告警"
    assert alerts[0]["level"] == "warning"
    assert all("完成" not in alert["title"] for alert in alerts)
