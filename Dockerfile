FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY gen_synthetic.py ./

RUN pip install --upgrade pip \
    && pip install -e .[dev]

CMD ["python", "scripts/train.py"]
