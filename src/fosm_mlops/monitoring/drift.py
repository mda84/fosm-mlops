"""Simple statistical drift detection for monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DriftReport:
    feature: str
    training_mean: float
    live_mean: float
    mean_diff: float
    drift_score: float


def compute_population_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in df.columns
        if df[col].dtype != "object"
    }


def compare_stats(
    training_stats: dict[str, dict[str, float]],
    live_df: pd.DataFrame,
    threshold: float = 3.0,
) -> list[DriftReport]:
    reports: list[DriftReport] = []
    for feature, stats in training_stats.items():
        if feature not in live_df.columns:
            continue
        live_mean = float(live_df[feature].mean())
        diff = live_mean - stats["mean"]
        drift_score = abs(diff) / (stats["std"] + 1e-6)
        if drift_score >= threshold:
            reports.append(
                DriftReport(
                    feature=feature,
                    training_mean=stats["mean"],
                    live_mean=live_mean,
                    mean_diff=diff,
                    drift_score=drift_score,
                )
            )
    return reports


def save_drift_report(reports: list[DriftReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([report.__dict__ for report in reports], f, indent=2)
