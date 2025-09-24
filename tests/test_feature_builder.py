from __future__ import annotations

import numpy as np
import pandas as pd

from fosm_mlops.features.build_features import (
    FeatureBuilder,
    FeatureBuilderConfig,
    SlidingWindowConfig,
)


def create_dummy_df() -> pd.DataFrame:
    time = np.arange(0, 10, 0.1)
    data = {
        "time": time,
        "sensor_id": np.zeros_like(time),
        "value": np.sin(time),
    }
    return pd.DataFrame(data)


def test_feature_builder_creates_windows(tmp_path):
    df = create_dummy_df()
    cfg = FeatureBuilderConfig(
        window=SlidingWindowConfig(window_size=20, step_size=10, sample_rate=10),
        output_dir=tmp_path,
    )
    builder = FeatureBuilder(cfg)
    artifacts = builder.build(df)
    feature_df = pd.read_parquet(artifacts.features_path)
    assert not feature_df.empty
    assert {"rms", "spectral_entropy"}.issubset(feature_df.columns)
