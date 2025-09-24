.PHONY: setup lint test format train eval data serve docker docker-compose kind-up kind-down docs clean run-ge unit

PYTHON=python

setup:
$(PYTHON) -m pip install --upgrade pip
$(PYTHON) -m pip install -e .[dev]
pre-commit install

lint:
ruff check src tests
black --check src tests
mypy src

format:
black src tests
ruff check --fix src tests

unit:
pytest -m "not integration"

test:
pytest

train:
$(PYTHON) scripts/train.py model=classical/xgboost
$(PYTHON) scripts/train.py model=anomaly/isolation_forest
$(PYTHON) scripts/train.py model=deep/conv1d

train-nightly:
$(PYTHON) scripts/train.py +train.experiment_name=nightly

eval:
$(PYTHON) scripts/evaluate.py

data:
$(PYTHON) gen_synthetic.py --sensors 4 --duration 120

serve:
uvicorn fosm_mlops.serve.app:app --reload

docker:
docker build -t fosm-mlops:train -f Dockerfile .
docker build -t fosm-mlops:serve -f Dockerfile.serve .

docker-compose:
docker compose up

kind-up:
kind create cluster --name fosm-mlops || true
kubectl apply -f k8s/

kind-down:
kind delete cluster --name fosm-mlops || true

run-ge:
great_expectations checkpoint run data_expectation_suite

clean:
rm -rf data/raw/* data/processed/* data/labels/* models/ mlruns/
