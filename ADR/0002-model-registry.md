# ADR 0002: MLflow as Model Registry

## Status
Accepted

## Context
We require experiment tracking, artifact storage, and model registry capabilities that integrate with Python ML stacks. Options considered: MLflow, Weights & Biases, SageMaker Model Registry, custom S3 bucket.

## Decision
Adopt MLflow with a local SQLite/Postgres backend and MinIO artifact store for this project. Benefits include:
- Native support for sklearn, TensorFlow, PyTorch models
- Model registry with staging/production lifecycle tags
- Easy integration with FastAPI inference service via `mlflow.pyfunc`
- Simple deployment via Docker Compose / Kubernetes

## Consequences
- Additional services (MLflow server, database, object store) to maintain
- Requires operators to manage artifact retention and access control
- Enables consistent experiment tracking and reproducible deployments
