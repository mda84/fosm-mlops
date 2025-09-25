"""Factory helpers to instantiate models from config names."""

from __future__ import annotations

from typing import Any

from .anomaly import AnomalyModelConfig, BaseAnomalyModel
from .classical import ClassicalModel, ClassicalModelConfig


def get_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a model by its registry ``name``.

    Parameters
    ----------
    name:
        Registry path such as ``"classical/random_forest"`` or ``"anomaly/zscore"``.
    **kwargs:
        Additional keyword arguments forwarded to the underlying config.
    """

    if name.startswith("classical/"):
        config = ClassicalModelConfig(name=name, **kwargs)
        return ClassicalModel(config)
    if name.startswith("anomaly/"):
        config = AnomalyModelConfig(name=name, **kwargs)
        return BaseAnomalyModel(config)
    if name.startswith("deep/"):
        msg = (
            "Deep learning models are not implemented in this lightweight runtime. "
            "Please integrate a TensorFlow/PyTorch implementation and update the registry."
        )
        raise NotImplementedError(msg)
    msg = f"Unknown model type for name: {name}"
    raise ValueError(msg)
