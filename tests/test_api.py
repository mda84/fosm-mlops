from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression


@pytest.fixture()
def model_dir(tmp_path) -> Path:
    x = np.random.randn(50, 4)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = LogisticRegression(max_iter=100)
    model.fit(x, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    feature_path = tmp_path / "feature_columns.json"
    feature_path.write_text(json.dumps([f"f{i}" for i in range(4)]), encoding="utf-8")
    os.environ["FOSM_MODEL_DIR"] = str(tmp_path)
    return tmp_path


def _load_app():
    sys.modules.pop("fosm_mlops.serve.app", None)
    module = importlib.import_module("fosm_mlops.serve.app")
    importlib.reload(module)
    return module


def test_predict_endpoint(model_dir):
    module = _load_app()
    client = TestClient(module.app)
    payload = {"rows": [{"f0": 0.1, "f1": -0.2, "f2": 0.05, "f3": 0.3}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert len(data["probabilities"]) == 1


def test_stream_endpoint(model_dir):
    module = _load_app()
    client = TestClient(module.app)
    chunk = {"row": {"f0": 0.1, "f1": -0.2, "f2": 0.05, "f3": 0.3}, "flush": True}
    response = client.post("/stream", json=chunk)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
