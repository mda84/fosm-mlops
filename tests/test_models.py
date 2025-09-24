from __future__ import annotations

import pytest
from sklearn.datasets import make_classification

from fosm_mlops.models.anomaly import AnomalyModelConfig, BaseAnomalyModel
from fosm_mlops.models.classical import ClassicalModel, ClassicalModelConfig
from fosm_mlops.models.utils import (
    classification_metrics,
    load_model,
    persist_model,
    tune_threshold,
)


@pytest.fixture(scope="module")
def binary_dataset():
    x, y = make_classification(n_samples=200, n_features=5, random_state=42)
    return x, y


def test_classical_logistic_regression(binary_dataset):
    x, y = binary_dataset
    model = ClassicalModel(
        ClassicalModelConfig(name="logistic_regression", oversample=None)
    )
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape[0] == x.shape[0]
    prob = model.predict_proba(x)
    assert prob.shape == (x.shape[0], 2)


def test_anomaly_zscore(binary_dataset):
    x, y = binary_dataset
    model = BaseAnomalyModel(AnomalyModelConfig(name="zscore"))
    model.fit(x)
    scores = model.predict_scores(x)
    assert scores.shape[0] == x.shape[0]


def test_threshold_tuning(binary_dataset):
    x, y = binary_dataset
    model = ClassicalModel(
        ClassicalModelConfig(
            name="random_forest", oversample=None, params={"n_estimators": 10}
        )
    )
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    threshold, score = tune_threshold(y, prob)
    metrics = classification_metrics(y, prob, threshold=threshold)
    assert 0 <= threshold <= 1
    assert "roc_auc" in metrics


def test_model_serialization(tmp_path, binary_dataset):
    x, y = binary_dataset
    model = ClassicalModel(
        ClassicalModelConfig(
            name="random_forest", oversample=None, params={"n_estimators": 5}
        )
    )
    model.fit(x, y)
    path = tmp_path / "model.joblib"
    persist_model(model.model, path)
    loaded = load_model(path)
    preds = loaded.predict(x)
    assert preds.shape[0] == x.shape[0]
