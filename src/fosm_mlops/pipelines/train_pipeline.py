"""End-to-end training pipeline orchestrated via Hydra."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

try:  # pragma: no cover - optional dependency path varies by version
    from great_expectations.dataset import PandasDataset
except ImportError:  # pragma: no cover - fallback for GE >= 1.6
    PandasDataset = None

from ..features.build_features import (
    FeatureBuilder,
    FeatureBuilderConfig,
    SlidingWindowConfig,
)
from ..ingest.batch import BatchLoader, BatchLoaderConfig
from ..models import registry, utils


logger = logging.getLogger(__name__)
CONFIG_PATH = str((Path(__file__).resolve().parents[3] / "configs").resolve())


def _flatten_params(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(name, value))
        elif isinstance(value, list):
            flat[name] = ",".join(map(str, value))
        else:
            flat[name] = value
    return flat


def _load_configured_dataframe(cfg_section: DictConfig) -> pd.DataFrame:
    loader = BatchLoader(
        BatchLoaderConfig(path=Path(cfg_section.path), format=cfg_section.format)
    )
    return loader.load()


def _fallback_expectation(
    df: pd.DataFrame, expectation_type: str, kwargs: dict[str, Any]
) -> None:
    column = kwargs.get("column")
    if column is None:
        raise ValueError(
            f"Expectation '{expectation_type}' requires a 'column' argument."
        )
    if expectation_type == "expect_column_values_to_not_be_null":
        if df[column].isnull().any():
            logger.warning(
                "Expectation '%s' failed: column '%s' contains null values.",
                expectation_type,
                column,
            )
        return
    if expectation_type == "expect_column_values_to_be_between":
        series = df[column]
        min_value = kwargs.get("min_value")
        max_value = kwargs.get("max_value")
        mask = pd.Series(True, index=series.index, dtype=bool)
        if min_value is not None:
            mask &= series >= min_value
        if max_value is not None:
            mask &= series <= max_value
        if not mask.all():
            logger.warning(
                "Expectation '%s' failed: values in column '%s' outside [%s, %s]",
                expectation_type,
                column,
                min_value,
                max_value,
            )
        return
    logger.warning(
        "Unsupported expectation '%s'. Install great_expectations<1.6 or extend"
        " the fallback handler.",
        expectation_type,
    )


def _run_expectations(df: pd.DataFrame, expectation_cfg: DictConfig | None) -> None:
    if expectation_cfg is None or not expectation_cfg.get("expectations"):
        return
    if PandasDataset is None:
        logger.warning(
            "great_expectations.dataset.PandasDataset is unavailable; applying"
            " limited built-in validations."
        )
        for expectation in expectation_cfg.expectations:
            expectation_type = expectation["type"]
            kwargs = expectation.get("kwargs", {})
            _fallback_expectation(df, expectation_type, kwargs)
        return
    dataset = PandasDataset(df)
    for expectation in expectation_cfg.expectations:
        expectation_type = expectation["type"]
        kwargs = expectation.get("kwargs", {})
        result = getattr(dataset, expectation_type)(**kwargs)
        if not result.success:
            logger.warning(
                "Expectation '%s' failed with kwargs %s.",
                expectation_type,
                kwargs,
            )


def _prepare_features(
    cfg: DictConfig, df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    window_cfg = SlidingWindowConfig(
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        sample_rate=cfg.sample_rate,
    )
    builder_cfg = FeatureBuilderConfig(
        window=window_cfg,
        scaler=cfg.scaler,
        output_dir=Path(cfg.output_dir),
    )
    builder = FeatureBuilder(builder_cfg)
    artifacts = builder.build(df, train=True)
    feature_df = pd.read_parquet(artifacts.features_path)
    metadata = artifacts.metadata
    metadata.update({"features_path": str(artifacts.features_path)})
    return feature_df, metadata


def _align_features_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    labels = labels.copy()
    features = features.copy()

    def assign_event(row: pd.Series) -> str:
        sensor_labels = labels[labels["sensor_id"] == row["sensor_id"]]
        mask = (sensor_labels["start_time"] <= row["time_end"]) & (
            sensor_labels["end_time"] >= row["time_start"]
        )
        if mask.any():
            return sensor_labels.loc[mask, "event"].iloc[0]
        return "normal"

    features["event"] = features.apply(assign_event, axis=1)
    features["target"] = (features["event"] != "normal").astype(int)
    return features


def _log_metrics(metrics: dict[str, float], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def _log_model_card(
    output_dir: Path, metrics: dict[str, float], metadata: dict[str, Any]
) -> Path:
    card_path = output_dir / "model_card.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    with card_path.open("w", encoding="utf-8") as f:
        f.write("# Model Card\n\n")
        f.write("## Overview\n")
        f.write("This model was trained on synthetic fiber-optic sensing data.\n\n")
        f.write("## Metrics\n")
        for key, value in metrics.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n## Data Metadata\n")
        for key, value in metadata.items():
            f.write(f"- **{key}**: {value}\n")
    return card_path


def _save_feature_columns(columns: list[str], artifact_dir: Path) -> Path:
    path = artifact_dir / "feature_columns.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)
    return path


@hydra_main(config_path=CONFIG_PATH, config_name="train/default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg.train
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    raw_df = _load_configured_dataframe(cfg.data.raw)
    labels_df = _load_configured_dataframe(cfg.data.labels)
    _run_expectations(raw_df, cfg.get("expectations"))

    feature_df, metadata = _prepare_features(cfg.features, raw_df)
    dataset = _align_features_labels(feature_df, labels_df)

    target = dataset["target"]
    feature_cols = dataset.columns.difference(
        ["target", "event", "start_time", "end_time"]
    )
    features = dataset[feature_cols]

    split = utils.time_aware_split(features, target, test_size=cfg.train.test_size)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    extra_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    model = registry.get_model(cfg.model.name, **extra_kwargs)

    feature_columns = split.x_train.drop(
        columns=["sensor_id", "time_start", "time_end"], errors="ignore"
    ).columns.tolist()
    x_train = split.x_train[feature_columns].to_numpy()
    x_test = split.x_test[feature_columns].to_numpy()
    if cfg.model.name.startswith("deep/"):
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    model.fit(x_train, split.y_train.to_numpy())
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "predict_scores"):
        scores = model.predict_scores(x_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    else:
        preds = model.predict(x_test)
        y_prob = preds.astype(float)
    threshold, best_score = utils.tune_threshold(
        split.y_test.to_numpy(), y_prob, metric=cfg.train.threshold_metric
    )
    metrics = utils.classification_metrics(
        split.y_test.to_numpy(), y_prob, threshold=threshold
    )
    metrics["threshold_score"] = best_score

    artifact_dir = Path(cfg.artifacts_dir)
    metrics_path = _log_metrics(metrics, artifact_dir)
    utils.save_metrics_plots(
        split.y_test.to_numpy(), y_prob, artifact_dir / "plots", threshold
    )
    model_path = artifact_dir / "model.joblib"
    model.save(model_path)
    model_card = _log_model_card(artifact_dir, metrics, metadata)
    feature_column_path = _save_feature_columns(feature_columns, artifact_dir)

    with mlflow.start_run(run_name=cfg.train.run_name):
        params = _flatten_params("", OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifacts(str(artifact_dir / "plots"))
        mlflow.log_artifact(str(model_card))
        try:
            mlflow.sklearn.log_model(model.model, artifact_path="model")  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - optional dependency fallback
            print(f"Skipping mlflow model log: {exc}")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feature_column_path))
        mlflow.set_tag("model_name", cfg.model.name)

    print(f"Training complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
