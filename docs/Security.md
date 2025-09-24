# Security Notes

## Secrets Management
- Use environment variables or external secret stores (Vault, AWS Secrets Manager) to inject credentials.
- `.env` files should be stored outside version control; sample templates can live in `config/` if needed.
- Never commit AWS/Azure/GCP credentials or private keys.

## Network & Access
- FastAPI service exposes `/healthz`, `/metadata`, `/predict`, `/stream`; apply authentication/authorization via reverse proxy or API gateway in production.
- Restrict MLflow UI access via OAuth/SSO when exposed publicly.
- Ensure Prometheus/Grafana endpoints are protected (basic auth, OAuth, VPN).

## Containers
- Base images pinned (python:3.10-slim) and updated regularly.
- Run containers as non-root where possible; sample Dockerfiles add user `mlops`.
- Enable read-only root filesystem in Kubernetes deployments for FastAPI and TF Serving.

## Data Handling
- Synthetic data used for development; replace with secure connectors for production datasets.
- Encrypt data at rest in MinIO/S3 (server-side encryption) and in transit (TLS termination at ingress).

## Logging & Auditing
- `structlog` provides structured request logs including model version/threshold.
- Centralize logs via ELK/CloudWatch; configure retention to comply with policies.
- Monitor drift metrics and anomaly thresholds for signs of tampering.

## Dependency Management
- Dependabot/GitHub Actions keep dependencies up-to-date.
- Run `pip-audit` or similar before releases.
- Pin major versions in `pyproject.toml` to avoid breaking upgrades.
