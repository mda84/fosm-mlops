# Code Review Checklist

## Data & Features
- [ ] No data leakage (train/validation/test separation enforced)
- [ ] Scaling/normalization fit on training set only
- [ ] Synthetic data generation documented with seeds
- [ ] Great Expectations suites updated when schema changes

## Modeling
- [ ] Configurable via Hydra/YAML (no hardcoded constants)
- [ ] Threshold tuning logic validated on held-out set
- [ ] Metrics include ROC-AUC, PR curve, confusion matrix
- [ ] Imbalance handling considered (SMOTE/random oversampling)

## MLOps & Infrastructure
- [ ] Dockerfiles minimal, pinned base images, no secrets baked in
- [ ] CI pipelines updated for new steps/tests
- [ ] MLflow logging includes params, metrics, and artifacts
- [ ] Serving endpoints provide schema validation & logging

## Testing
- [ ] Unit tests cover new logic with edge cases
- [ ] Integration tests updated if interfaces change
- [ ] TensorFlow/PyTorch code guarded with optional imports
- [ ] Pre-commit hooks passing locally (`make lint`, `make test`)

## Documentation
- [ ] README / docs updated to reflect major changes
- [ ] ADR added/updated for significant decisions
- [ ] Operational Runbook adjustments for deployment changes
- [ ] Security considerations documented when handling credentials
