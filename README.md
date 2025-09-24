# Fiber-Optic Pipeline Monitoring (fosm-mlops)

An end-to-end ML/MLOps reference implementation for fiber-optic pipeline monitoring. The project spans synthetic data generation, signal processing, model training/evaluation, production serving, orchestration, and observability.

## 🧭 Architecture Overview

```mermaid
graph LR
    A[gen_synthetic.py] --> B[data lake - bronze]
    B --> C[Feature Builder]
    C --> D[Model Zoo (sklearn/tf)]
    D --> E[MLflow Registry]
    D --> F[FastAPI Service]
    E --> F
    F --> G[Prometheus/Grafana]
    D --> H[TensorFlow & TorchServe]
    B --> I[Spark Jobs]
    C --> J[Great Expectations]
```

## 🚀 Quickstart

1. **Setup environment**
   ```bash
   make setup
   ```

2. **Generate synthetic data**
   ```bash
   make data
   ```

3. **Train baseline models (XGBoost, Isolation Forest, Conv1D)**
   ```bash
   make train
   ```

4. **Evaluate and produce reports**
   ```bash
   make eval
   ```

5. **Run the FastAPI inference service**
   ```bash
   make serve
   # in another terminal
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"rows": [{"rms": 0.3, "peak_to_peak": 0.5, "spectral_centroid": 10.0, "spectral_entropy": 0.7}]}'
   ```

6. **Spin up the full stack (API, MLflow, TF-Serving, Grafana, Prometheus)**
   ```bash
   docker compose up
   ```

7. **Deploy to local Kubernetes via kind**
   ```bash
   make kind-up
   kubectl apply -f k8s/
   ```

## 📂 Repository Structure

- `gen_synthetic.py` – synthetic fiber-optic signal generator
- `src/fosm_mlops/` – python package (ingest, features, models, pipelines, serving, monitoring)
- `configs/` – Hydra/YAML configs for data, models, training, evaluation
- `scripts/` – CLI utilities (`train.py`, `evaluate.py`)
- `spark_jobs/` – PySpark batch jobs (ingest, feature rollup, anomaly scoring)
- `serve/` – TorchServe handler and serving assets
- `docs/` – operational runbooks, security notes, model card template
- `tests/` – pytest suite covering signal processing, feature engineering, models, and API
- `k8s/` – manifests for FastAPI, TF Serving, MLflow, monitoring stack, and HPAs
- `ansible/` – playbooks for provisioning & deployment
- `ADR/` – architectural decision records

## 🧪 Testing & Quality

- `ruff`, `black`, `mypy`, `pytest` via `make lint` / GitHub Actions
- `pre-commit` hooks configured (`make setup` installs)
- Synthetic datasets seeded for reproducibility
- Great Expectations validations prevent schema drift

## 📈 Monitoring & Observability

- Prometheus scrapes FastAPI and TensorFlow Serving metrics (see `docker-compose.yml` + `k8s/`)
- Grafana dashboards track latency, throughput, anomaly rates, and data drift summaries
- `src/fosm_mlops/monitoring/drift.py` provides quick population-statistics based drift detection

## 🧰 MLOps Toolchain

| Capability | Implementation |
|------------|----------------|
| Experiment tracking | MLflow (local backend + MinIO artifact store) |
| Config management | Hydra + YAML |
| Data validation | Great Expectations |
| Feature engineering | Savitzky–Golay, Butterworth, FFT/STFT, band energies |
| Models | Logistic Regression, RandomForest, XGBoost, LightGBM, Isolation Forest, One-Class SVM, Z-Score, Conv1D, LSTM, GRU, Autoencoder |
| Serving | FastAPI, TensorFlow Serving, TorchServe |
| CI/CD | GitHub Actions (lint/test, docker build, nightly training) |
| Deployment | Docker, Ansible, Kubernetes (kind) |

## 📚 Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) – workflow, coding standards, review checklist
- [docs/Operational_Runbook.md](docs/Operational_Runbook.md) – deploy/rollback runbooks & troubleshooting
- [docs/ModelCardTemplate.md](docs/ModelCardTemplate.md) – template auto-populated after evaluations
- [docs/Security.md](docs/Security.md) – handling secrets, RBAC, container hardening

## 🧪 Notebooks

Interactive notebooks for exploration live under `notebooks/`:

1. `01_eda_signal_visuals.ipynb` – raw vs filtered signals, FFT/STFT demos
2. `02_feature_benchmarks.ipynb` – sliding window feature comparisons
3. `03_model_benchmarks.ipynb` – compare classical vs. deep learning vs. anomaly models

## 🔐 Security

- No secrets in repo; use `.env` files or environment variables
- Docker/Ansible/Kubernetes manifests parameterized for credentials
- FastAPI uses Pydantic validation and structlog audit logs

## 🧭 Roadmap Ideas

- Integrate streaming feature store
- Add real-time drift alerts via Prometheus Alertmanager
- Extend to multi-task detection (event classification + regression severity)

Happy monitoring! ⚡
