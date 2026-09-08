"""ModelFactory 构建正确性测试（多分类参数注入）。"""
import numpy as np
import pytest

from model_code.model_factory import ModelFactory


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 3, 60)
    return X, y


_MINIMAL_PARAMS = {
    "lightgbm": {"n_estimators": 10, "n_jobs": 1},
    "xgboost": {"n_estimators": 10, "n_jobs": 1},
    "catboost": {"iterations": 10, "thread_count": 1},
    "random_forest": {"n_estimators": 10, "n_jobs": 1},
    "logistic_regression": {"max_iter": 200},
}


@pytest.mark.parametrize(
    "algorithm",
    ["lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"],
)
def test_factory_builds_and_fits_all_algorithms_multiclass(algorithm, multiclass_data):
    X, y = multiclass_data
    model = ModelFactory.build(algorithm, task_type="multiclass", params=_MINIMAL_PARAMS[algorithm])

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(model.classes_) == 3
    assert len(predictions) == len(y)


def test_lightgbm_multiclass_does_not_inject_none_num_classes():
    model = ModelFactory.build("lightgbm", task_type="multiclass", params={"n_estimators": 10})

    params = model.get_params()
    assert "num_classes" not in params
    assert params.get("num_class") is None  # 未显式提供时不注入，交由 fit 推断


def test_lightgbm_multiclass_accepts_explicit_num_class():
    model = ModelFactory.build("lightgbm", task_type="multiclass", params={"num_class": 3, "n_estimators": 10})

    assert model.get_params()["num_class"] == 3


@pytest.mark.parametrize(
    "bad_params",
    [
        {"iterations": 5, "train_dir": "/tmp/evil-dir"},
        {"iterations": 5, "allow_writing_files": True},
        {"iterations": 5, "snapshot_file": "/tmp/evil-dir/snap"},
    ],
)
def test_catboost_rejects_filesystem_related_params(bad_params):
    """训练参数来自 API 调用方，必须封禁 CatBoost 的文件系统写入参数，
    防止 fit() 被诱导在任意目录创建文件（受限文件写入原语）。"""
    with pytest.raises(ValueError, match="文件系统"):
        ModelFactory.build("catboost", task_type="binary", params=bad_params)


@pytest.mark.parametrize(
    ("algorithm", "params"),
    [
        ("lightgbm", {"n_estimators": 5, "n_jobs": 1}),
        ("xgboost", {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}),
        ("catboost", {"iterations": 5, "depth": 2, "thread_count": 1}),
        ("random_forest", {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}),
        ("linear_regression", {}),
    ],
)
def test_factory_builds_regressors(algorithm, params):
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = X[:, 0] * 2.5 - X[:, 1]

    model = ModelFactory.build(algorithm, task_type="regression", params=params)
    model.fit(X, y)
    prediction = model.predict(X[:2])

    assert prediction.shape == (2,)
    assert np.issubdtype(prediction.dtype, np.floating)


def test_factory_rejects_task_incompatible_linear_models():
    with pytest.raises(ValueError, match="logistic_regression"):
        ModelFactory.build("logistic_regression", task_type="regression")
    with pytest.raises(ValueError, match="linear_regression"):
        ModelFactory.build("linear_regression", task_type="binary")
