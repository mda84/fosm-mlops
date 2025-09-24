# ADR 0001: Overall Architecture

## Status
Accepted

## Context
Pipeline monitoring requires ingesting high-frequency fiber-optic signals, producing features, training models, and deploying inference services with observability. We need a modular architecture that supports experimentation and productionization.

## Decision
Adopt a lakehouse-style architecture with the following layers:
- **Raw**: Synthetic DAS signals generated via `gen_synthetic.py`
- **Bronze/Silver**: PySpark jobs enforce schema and derive clean tables
- **Feature**: Python feature builder generates spectral features stored under `data/processed`
- **Model**: Model zoo with classical (sklearn), anomaly detection, and TensorFlow deep learning
- **Serving**: FastAPI microservice with optional TensorFlow Serving and TorchServe integrations
- **Monitoring**: Prometheus + Grafana dashboards, drift checks

Hydra manages configurations, MLflow tracks experiments, Great Expectations validates data.

## Consequences
- Enables reproducible pipelines with clear boundaries between data, features, models, and serving.
- Simplifies swapping models or adjusting thresholds via configs/MLflow.
- Additional complexity due to multiple stacks (Spark, MLflow, TF Serving) but provides realistic MLOps coverage.
