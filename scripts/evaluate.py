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


def _load_feature_columns(cfg: dict) -> list[str]:
    """Load the feature column ordering used during training if available."""

    feature_path = cfg.get("feature_columns_path")
    if feature_path is None:
        feature_path = Path(cfg["model_path"]).with_name("feature_columns.json")
    else:
        feature_path = Path(feature_path)

    if not feature_path.exists():
        return []

    with feature_path.open("r", encoding="utf-8") as f:
        columns = json.load(f)

    if not isinstance(columns, list):  # Defensive check against corrupt files
        raise ValueError(
            "feature_columns.json must contain a JSON array of column names"
        )

    return [str(col) for col in columns]


def _infer_model_features(model: object) -> list[str]:
    """Attempt to recover feature names from a fitted model pipeline."""

    # Standard scikit-learn estimators expose ``feature_names_in_`` after fitting.
    if hasattr(model, "feature_names_in_"):
        return [str(col) for col in model.feature_names_in_]

    # Pipelines may delegate this attribute to one of their steps (typically the
    # first transformer or the estimator itself). Iterate in order to find the
    # first match.
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(col) for col in step.feature_names_in_]

    # Some custom wrappers might expose a ``feature_columns`` attribute.
    if hasattr(model, "feature_columns"):
        feature_cols = getattr(model, "feature_columns")
        if isinstance(feature_cols, (list, tuple)):
            return [str(col) for col in feature_cols]

    return []


def _prepare_feature_matrix(
    cfg: dict, features: pd.DataFrame, model: object
) -> pd.DataFrame:
    feature_columns = _load_feature_columns(cfg)
    if not feature_columns:
        feature_columns = _infer_model_features(model)

    df = features.drop(
        columns=["target", "event", "sensor_id", "start_time", "end_time"],
        errors="ignore",
    ).copy()

    if feature_columns:
        missing_cols = [col for col in feature_columns if col not in df.columns]
        for column in missing_cols:
            df[column] = 0.0
        df = df.reindex(columns=feature_columns)
    else:
        # Fall back to deterministic ordering if no metadata is available
        df = df[sorted(df.columns)]

    return df


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

    feature_matrix = _prepare_feature_matrix(cfg, merged, model)
    x = feature_matrix.to_numpy()
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
