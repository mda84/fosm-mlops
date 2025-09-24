"""FastAPI application for serving anomaly detection models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = structlog.get_logger(__name__)

MODEL_DIR = Path(os.getenv("FOSM_MODEL_DIR", "models/latest"))
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_PATH = MODEL_DIR / "model.joblib"


class PredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(
        ..., description="List of feature dictionaries"
    )


class PredictResponse(BaseModel):
    probabilities: list[float]
    predictions: list[int]
    threshold: float


class StreamRequest(BaseModel):
    row: dict[str, float]
    flush: bool = False


class MetadataResponse(BaseModel):
    model_name: str
    feature_columns: list[str]
    source: str


def _load_feature_columns() -> list[str]:
    if FEATURE_COLUMNS_PATH.exists():
        with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_model():
    mlflow_uri = os.getenv("MLFLOW_MODEL_URI")
    if mlflow_uri:
        import mlflow

        LOGGER.info("loading-model-from-mlflow", uri=mlflow_uri)
        return mlflow.pyfunc.load_model(mlflow_uri)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    LOGGER.info("loading-model-from-disk", path=str(MODEL_PATH))
    return joblib.load(MODEL_PATH)


class PredictionService:
    def __init__(self) -> None:
        self.model = _load_model()
        self.feature_columns = _load_feature_columns()
        self.threshold = float(os.getenv("FOSM_THRESHOLD", "0.5"))
        self.buffer: list[dict[str, float]] = []

    def _prepare(self, rows: list[dict[str, float]]) -> np.ndarray:
        df = pd.DataFrame(rows)
        if not self.feature_columns:
            self.feature_columns = sorted(df.columns.tolist())
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature columns: {sorted(missing_cols)}",
            )
        df = df[self.feature_columns]
        return df.to_numpy()

    def predict(self, rows: list[dict[str, float]]) -> PredictResponse:
        features = self._prepare(rows)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[:, 1]
        else:
            try:
                preds = self.model.predict(pd.DataFrame(rows)[self.feature_columns])
                scores = np.asarray(preds).astype(float)
            except Exception:
                scores = np.asarray(self.model.predict(features)).astype(float)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        preds = (probs >= self.threshold).astype(int)
        return PredictResponse(
            probabilities=list(map(float, probs)),
            predictions=preds.tolist(),
            threshold=self.threshold,
        )

    def stream(self, row: dict[str, float], flush: bool) -> PredictResponse | None:
        self.buffer.append(row)
        if flush or len(self.buffer) >= int(os.getenv("FOSM_STREAM_BATCH", "16")):
            rows = self.buffer
            self.buffer = []
            return self.predict(rows)
        return None


service = PredictionService()
app = FastAPI(title="Fiber Optic Pipeline Monitoring API", version="0.1.0")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_name=os.getenv("FOSM_MODEL_NAME", "fosm-model"),
        feature_columns=service.feature_columns,
        source="mlflow" if os.getenv("MLFLOW_MODEL_URI") else "local",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    LOGGER.info("predict", rows=len(request.rows))
    return service.predict(request.rows)


@app.post("/stream")
def stream(request: StreamRequest):
    result = service.stream(request.row, request.flush)
    if result is None:
        return {"status": "buffered", "buffer_size": len(service.buffer)}
    return result
