"""Data ingestion utilities for batch processing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BatchLoaderConfig:
    """Configuration for batch loading."""

    path: Path
    format: str = "parquet"
    columns: list[str] | None = None


class BatchLoader:
    """Load time-series sensor data from disk."""

    def __init__(self, config: BatchLoaderConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        if self.config.format == "parquet":
            return pd.read_parquet(self.config.path, columns=self.config.columns)
        if self.config.format == "csv":
            return pd.read_csv(self.config.path, usecols=self.config.columns)
        raise ValueError(f"Unsupported format: {self.config.format}")

    def load_labels(self) -> pd.DataFrame:
        return self.load()

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path)
        elif path.suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported suffix: {path.suffix}")


def load_multiple(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths
    ]
    return pd.concat(frames, ignore_index=True)
