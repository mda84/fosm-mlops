"""Classical supervised learning models."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import persist_model

try:  # pragma: no cover - optional dependency imports
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for classical models") from exc

try:  # pragma: no cover
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

try:  # pragma: no cover
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]


@dataclass(slots=True)
class ClassicalModelConfig:
    """Configuration for classical classifiers."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    oversample: str | None = None


class ClassicalModel:
    """Wrapper around scikit-learn style classifiers with optional oversampling."""

    def __init__(self, config: ClassicalModelConfig) -> None:
        self.config = config
        self.model = self._build_model(config)
        self._oversampler = self._build_sampler(config.oversample)

    @staticmethod
    def _build_sampler(name: str | None) -> Any | None:
        if name is None:
            return None
        name = name.lower()
        if name == "smote":
            return SMOTE()
        if name == "random":
            return RandomOverSampler()
        msg = f"Unsupported oversampler: {name}"
        raise ValueError(msg)

    @staticmethod
    def _build_model(config: ClassicalModelConfig) -> Any:
        name = config.name.split("/", maxsplit=1)[-1]
        params = {**config.params}
        if name == "logistic_regression":
            estimator = LogisticRegression(**params)
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
        if name == "random_forest":
            return RandomForestClassifier(**params)
        if name == "xgboost":
            if XGBClassifier is None:
                msg = "xgboost is not installed."
                raise ImportError(msg)
            default_params = {"use_label_encoder": False, "eval_metric": "logloss"}
            default_params.update(params)
            return XGBClassifier(**default_params)
        if name == "lightgbm":
            if LGBMClassifier is None:
                msg = "lightgbm is not installed."
                raise ImportError(msg)
            return LGBMClassifier(**params)
        msg = f"Unsupported classical model: {config.name}"
        raise ValueError(msg)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self._oversampler is not None:
            x, y = self._oversampler.fit_resample(x, y)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            msg = "Underlying model does not support predict_proba"
            raise AttributeError(msg)
        return self.model.predict_proba(x)

    def save(self, path: PathLike[str] | str) -> None:
        persist_model(self.model, path)

    @classmethod
    def load(cls, path: PathLike[str] | str) -> ClassicalModel:
        from .utils import load_model

        model = load_model(path)
        instance = cls(ClassicalModelConfig(name="loaded"))
        instance.model = model
        instance._oversampler = None
        return instance
