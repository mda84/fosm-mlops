"""Lightweight anomaly detection primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import joblib
import numpy as np

from .utils import persist_model


@dataclass(slots=True)
class AnomalyModelConfig:
    """Configuration for anomaly detectors."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class _ZScoreModel:
    """Simple z-score based anomaly scoring."""

    def __init__(self, **_: Any) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-8

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            msg = "Model must be fitted before calling predict_scores."
            raise RuntimeError(msg)
        z = np.abs((x - self.mean_) / self.std_)
        return np.max(z, axis=1)

    def save(self, path: str | PathLike[str]) -> None:
        persist_model(self, path)


class BaseAnomalyModel:
    """Unified interface for anomaly detection models."""

    def __init__(self, config: AnomalyModelConfig) -> None:
        self.config = config
        self.model = self._build_model(config)

    @staticmethod
    def _build_model(config: AnomalyModelConfig) -> Any:
        name = config.name.split("/", maxsplit=1)[-1]
        if name == "zscore":
            return _ZScoreModel(**config.params)
        if name == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            return IsolationForest(**config.params)
        if name == "lof":
            from sklearn.neighbors import LocalOutlierFactor

            return LocalOutlierFactor(novelty=True, **config.params)
        msg = f"Unsupported anomaly model: {config.name}"
        raise ValueError(msg)

    def fit(self, x: np.ndarray) -> None:
        self.model.fit(x)

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_scores"):
            return self.model.predict_scores(x)  # type: ignore[return-value]
        scores = self.model.decision_function(x)
        if scores.ndim == 1:
            return scores
        return scores.squeeze()

    def predict(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        scores = self.predict_scores(x)
        return (scores > threshold).astype(int)

    def save(self, path: str | PathLike[str]) -> None:
        persist_model(self.model, path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> BaseAnomalyModel:
        model = joblib.load(path)
        instance = object.__new__(cls)
        instance.config = AnomalyModelConfig(name="loaded")
        instance.model = model
        return instance
