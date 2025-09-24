"""Evaluate trained models and produce reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fosm_mlops.models import utils


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(Path("configs/eval/default.yaml"))
    features = pd.read_parquet(cfg["features_path"])
    labels = pd.read_csv(cfg["labels_path"])
    model = utils.load_model(Path(cfg["model_path"]))

    def assign_event(row):
        sensor_labels = labels[labels["sensor_id"] == row["sensor_id"]]
        mask = (sensor_labels["start_time"] <= row["time_end"]) & (
            sensor_labels["end_time"] >= row["time_start"]
        )
        if mask.any():
            return sensor_labels.loc[mask, "event"].iloc[0]
        return "normal"

    merged = features.copy()
    merged["event"] = merged.apply(assign_event, axis=1)
    merged["target"] = (merged["event"] != "normal").astype(int)
    x = merged.drop(columns=["target", "event", "start_time", "end_time"], errors="ignore").to_numpy()
    y_true = merged["target"].to_numpy()

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x)[:, 1]
    elif hasattr(model, "predict_scores"):
        scores = model.predict_scores(x)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    else:
        scores = model.predict(x)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    threshold, _ = utils.tune_threshold(y_true, y_prob, metric="f1")
    metrics = utils.classification_metrics(y_true, y_prob, threshold=threshold)

    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    utils.save_metrics_plots(y_true, y_prob, reports_dir / "figures", threshold)

    print(f"Evaluation complete. Metrics stored at {metrics_path}")


if __name__ == "__main__":
    main()
