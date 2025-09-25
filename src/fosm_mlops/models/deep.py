"""Lightweight deep learning inspired models.

This module provides minimal NumPy/Scikit-learn based fallbacks for
"deep" models so the training pipeline can operate without optional heavy
dependencies such as TensorFlow or PyTorch.  The implementation is not a
true neural network but instead performs feature engineering that mimics
simple convolutional filters before fitting an ``MLPClassifier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Iterable

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import persist_model


@dataclass(slots=True)
class DeepModelConfig:
    """Configuration container for lightweight deep models."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Conv1DModel:
    """Approximation of a 1D convolutional classifier.

    The class emulates a convolutional neural network by computing a small
    collection of moving statistics (mean, standard deviation and local
    gradients) across the input time dimension.  The enriched feature
    representation is then passed to a multilayer perceptron implemented by
    ``scikit-learn``'s :class:`~sklearn.neural_network.MLPClassifier`.
    """

    def __init__(self, config: DeepModelConfig) -> None:
        self.config = config
        params = {**config.params}

        # Hyper-parameters for the synthetic convolutional features
        kernel_sizes = params.pop("kernel_sizes", (3, 5))
        if isinstance(kernel_sizes, Iterable) and not isinstance(kernel_sizes, (str, bytes)):
            self.kernel_sizes = tuple(int(max(1, k)) for k in kernel_sizes)
        else:  # pragma: no cover - configuration edge case
            self.kernel_sizes = (int(max(1, kernel_sizes)),)

        # Map training hyper-parameters to the scikit-learn estimator.  Only
        # a subset are meaningful for the fallback implementation.
        mlp_params: dict[str, Any] = {
            "hidden_layer_sizes": params.pop("hidden_layers", (64,)),
            "learning_rate_init": params.pop("learning_rate", 1e-3),
            "alpha": params.pop("l2", 1e-4),
            "max_iter": params.pop("epochs", 20),
            "batch_size": params.pop("batch_size", 32),
            "random_state": params.pop("random_state", 42),
            "activation": params.pop("activation", "relu"),
        }

        # Drop unsupported high-level knobs that are not available in the
        # scikit-learn implementation but may be present in the config.
        params.pop("optimizer", None)
        params.pop("dropout", None)

        # Any remaining parameters are forwarded directly to the classifier.
        mlp_params.update(params)

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**mlp_params)),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        features = self._prepare_features(x)
        self.model.fit(features, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self._prepare_features(x)
        return self.model.predict(features)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        features = self._prepare_features(x)
        if not hasattr(self.model, "predict_proba"):
            msg = "Underlying model does not support predict_proba"
            raise AttributeError(msg)
        return self.model.predict_proba(features)

    def save(self, path: PathLike[str] | str) -> None:
        persist_model(self.model, path)

    @classmethod
    def load(cls, path: PathLike[str] | str) -> Conv1DModel:  # pragma: no cover - convenience
        from .utils import load_model

        model = load_model(path)
        instance = cls(DeepModelConfig(name="loaded"))
        instance.model = model
        return instance

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 2:
            msg = "Input data must be of shape (n_samples, n_timesteps[, 1])"
            raise ValueError(msg)

        # Original features plus engineered statistics.
        feature_list = [x]
        for kernel in self.kernel_sizes:
            if kernel <= 1 or kernel > x.shape[1]:
                continue
            feature_list.append(self._moving_average(x, kernel))
            feature_list.append(self._moving_std(x, kernel))
            feature_list.append(self._local_gradient(x, kernel))

        if len(feature_list) == 1:  # pragma: no cover - degenerate case
            return x
        return np.concatenate(feature_list, axis=1)

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        padded = np.pad(x, ((0, 0), (window // 2, window - window // 2 - 1)), mode="edge")
        result = np.empty_like(x)
        for idx in range(x.shape[1]):
            window_slice = padded[:, idx : idx + window]
            result[:, idx] = window_slice.mean(axis=1)
        return result

    @staticmethod
    def _moving_std(x: np.ndarray, window: int) -> np.ndarray:
        padded = np.pad(x, ((0, 0), (window // 2, window - window // 2 - 1)), mode="edge")
        result = np.empty_like(x)
        for idx in range(x.shape[1]):
            window_slice = padded[:, idx : idx + window]
            result[:, idx] = window_slice.std(axis=1)
        return result

    @staticmethod
    def _local_gradient(x: np.ndarray, window: int) -> np.ndarray:
        padded = np.pad(x, ((0, 0), (1, 1)), mode="edge")
        left = padded[:, :-2]
        right = padded[:, 2:]
        gradient = (right - left) / max(window - 1, 1)
        return gradient

