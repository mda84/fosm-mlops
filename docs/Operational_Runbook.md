# Operational Runbook

## Deployment

1. **Docker Compose (local)**
   ```bash
   docker compose up -d
   ```
   - Services: FastAPI (`api`), MLflow (`mlflow` + `postgres` + `minio`), TensorFlow Serving (`tf-serving`), Prometheus, Grafana

2. **Kubernetes (kind)**
   ```bash
   make kind-up
   kubectl apply -f k8s/
   ```
   - Deployments: `fosm-api`, `tf-serving`, `mlflow`, `minio`
   - HPA scales API and TF Serving based on CPU usage

3. **Ansible (edge nodes)**
   ```bash
   ansible-playbook -i inventory.ini ansible/site.yml
   ```
   - Installs Docker, pulls images, configures systemd units

## Monitoring

- Access Grafana at `http://localhost:3000` (default creds admin/admin)
- Dashboards include: API latency, TF Serving throughput, anomaly rate trend, data drift summary
- Prometheus at `http://localhost:9090`
- Drift reports stored under `reports/drift/`

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| API returns 500 | Check container logs `docker logs fosm-api`. Ensure model artifacts exist under `models/latest`. |
| TF Serving not responding | Verify `models/tf_export/` contains SavedModel, restart `tf-serving` container. |
| MLflow artifacts missing | Confirm MinIO credentials via `.env`. Re-create bucket using `scripts/mlflow_setup.py` (TBD). |
| Great Expectations failure | Inspect `data/expectations` outputs; regenerate synthetic data or adjust expectation thresholds. |

## Rollback

1. Restore last known-good model from MLflow registry: `mlflow models serve -m models:/fosm/Production`
2. Re-deploy API with pinned `FOSM_MODEL_URI`
3. Monitor metrics for stabilization

## On-call Checklist

- [ ] API latency < 200ms p95
- [ ] Prediction throughput > 1k req/min sustained
- [ ] Drift alerts resolved within 1 hour
- [ ] Nightly training job completed (`train-eval.yml`)
