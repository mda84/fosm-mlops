"""Tests for evaluation utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_evaluate_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("_evaluate", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to load evaluate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyModel:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns


def test_prepare_feature_matrix_uses_model_metadata(tmp_path: Path) -> None:
    cfg = {"feature_columns_path": tmp_path / "missing.json"}
    data = pd.DataFrame(
        {
            "sensor_id": [1, 1],
            "time_start": [0.0, 1.0],
            "time_end": [0.5, 1.5],
            "feature_a": [0.1, 0.2],
            "feature_b": [0.3, 0.4],
        }
    )
    data["target"] = 0

    model = _DummyModel(["feature_b", "feature_c", "feature_a"])
    evaluate_module = _load_evaluate_module()
    matrix = evaluate_module._prepare_feature_matrix(cfg, data, model)

    assert list(matrix.columns) == ["feature_b", "feature_c", "feature_a"]
    assert (matrix["feature_c"] == 0.0).all()
