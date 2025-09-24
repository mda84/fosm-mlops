# Contributing Guide

Thank you for investing time in fosm-mlops! This repository serves as a reference MLOps implementation for fiber-optic pipeline monitoring. Contributions are welcome across data engineering, ML modeling, infrastructure, and documentation.

## Development Environment

1. Clone the repository and install dependencies:
   ```bash
   make setup
   ```
2. Install the pre-commit hooks (done automatically during `make setup`).
3. Run `make lint` and `make test` before pushing.

## Branching & Commits

- Use feature branches with descriptive names (`feature/signal-filters`, `bugfix/api-threshold`).
- Keep commits focused and include a clear message summarizing the change.
- Avoid committing data artifacts (`data/raw`, `data/processed`, `models`, `mlruns`).

## Pull Requests

- Provide a concise summary, screenshots for UI changes, and validation evidence.
- Reference related issues and ADRs.
- Ensure CI checks pass (`lint-test.yml`).
- Include tests for new functionality (unit, integration, or smoke).
- Review the [Code Review Checklist](docs/Code_Review_Checklist.md) before requesting review.

## Coding Standards

- Python 3.10+, type hints encouraged (mypy tolerant of optional deps).
- Formatting via `black`, lint with `ruff`, typed with `mypy` (`strict = false`).
- Prefer modular functions/classes with docstrings.
- Keep configuration externalized (Hydra/YAML) to avoid code changes for tuning.

## Testing Strategy

- Unit tests for signal processing, feature engineering, models, APIs.
- Synthetic data generator tests should validate label alignment & reproducibility.
- Use `pytest.importorskip` for optional heavy dependencies (TensorFlow, PySpark).

## Documentation

- Update README and docs/ where applicable (e.g., operational changes, security).
- Record architectural decisions in `ADR/` via short markdown docs.
- Keep diagrams up to date (Mermaid or referenced images stored under `docs/`).

## Release Process

1. Merge PRs into `main` only when CI passes.
2. Tag releases using semantic versioning (e.g., `v0.1.0`).
3. Publish release notes summarizing changes, migrations, and validation evidence.

Thanks for helping build resilient fiber-optic monitoring systems! 🚀
