"""Feature engineering pipeline for fiber-optic signals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import signal_processing as sp


@dataclass
class SlidingWindowConfig:
    window_size: int
    step_size: int
    sample_rate: float


@dataclass
class FeatureBuilderConfig:
    window: SlidingWindowConfig
    scaler: str = "standard"
    bands: tuple[tuple[float, float], ...] = ((0.1, 2.0), (2.0, 10.0), (10.0, 40.0))
    output_dir: Path = Path("data/processed")


@dataclass
class FeatureArtifacts:
    features_path: Path
    scaler_path: Path
    metadata: dict[str, Any]


def segment_windows(
    df: pd.DataFrame, config: SlidingWindowConfig
) -> Iterable[pd.DataFrame]:
    """Yield sliding windows per sensor."""
    for sensor_id, sensor_df in df.groupby("sensor_id"):
        sensor_df = sensor_df.sort_values("time")
        values = sensor_df["value"].to_numpy()
        times = sensor_df["time"].to_numpy()
        for start in range(0, len(values) - config.window_size + 1, config.step_size):
            window_values = values[start : start + config.window_size]
            window_times = times[start : start + config.window_size]
            yield pd.DataFrame(
                {
                    "sensor_id": sensor_id,
                    "time": window_times,
                    "value": window_values,
                }
            )


def _select_scaler(name: str):
    if name.lower() == "standard":
        return StandardScaler()
    if name.lower() == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


def extract_features(
    window: pd.DataFrame, config: FeatureBuilderConfig
) -> dict[str, float]:
    values = window["value"].to_numpy()
    fs = config.window.sample_rate
    freq, spectrum = sp.fft_spectrum(values, fs)
    rms = sp.root_mean_square(values)
    p2p = sp.peak_to_peak(values)
    centroid = sp.spectral_centroid(freq, spectrum)
    entropy = sp.spectral_entropy(spectrum)
    band_feats = {
        f"band_energy_{low}_{high}": sp.band_energy(freq, spectrum, (low, high))
        for low, high in config.bands
    }
    return {
        "sensor_id": float(window["sensor_id"].iloc[0]),
        "time_start": float(window["time"].iloc[0]),
        "time_end": float(window["time"].iloc[-1]),
        "rms": rms,
        "peak_to_peak": p2p,
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
        **band_feats,
    }


class FeatureBuilder:
    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config
        self.scaler = _select_scaler(config.scaler)

    def build(self, df: pd.DataFrame, train: bool = True) -> FeatureArtifacts:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        features: list[dict[str, float]] = []
        for window in segment_windows(df, self.config.window):
            features.append(extract_features(window, self.config))
        feature_df = pd.DataFrame(features)
        feature_df.sort_values(["sensor_id", "time_start"], inplace=True)
        numeric_cols = feature_df.columns.difference(
            ["sensor_id", "time_start", "time_end"]
        )
        if train:
            feature_df[numeric_cols] = self.scaler.fit_transform(
                feature_df[numeric_cols]
            )
        else:
            feature_df[numeric_cols] = self.scaler.transform(feature_df[numeric_cols])
        features_path = self.config.output_dir / "features.parquet"
        scaler_path = self.config.output_dir / "scaler.joblib"
        feature_df.to_parquet(features_path, index=False)
        joblib.dump(self.scaler, scaler_path)
        metadata = {
            "num_windows": len(feature_df),
            "window_size": self.config.window.window_size,
            "step_size": self.config.window.step_size,
            "sample_rate": self.config.window.sample_rate,
        }
        return FeatureArtifacts(
            features_path=features_path, scaler_path=scaler_path, metadata=metadata
        )

    def transform(self, df: pd.DataFrame) -> FeatureArtifacts:
        return self.build(df, train=False)
