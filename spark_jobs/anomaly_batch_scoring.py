"""Run anomaly detection over daily data using Spark."""

from __future__ import annotations

from pathlib import Path

import joblib
from pyspark.sql import SparkSession


MODEL_PATH = Path("models/latest/model.joblib")


def main() -> None:
    spark = SparkSession.builder.appName("fosm-anomaly-scoring").getOrCreate()
    model = joblib.load(MODEL_PATH)
    df = spark.read.parquet("lake/gold/features")
    pandas_df = df.toPandas()
    features = pandas_df.drop(columns=["sensor_id", "window_id", "values"], errors="ignore")
    scores = model.decision_function(features.to_numpy()) if hasattr(model, "decision_function") else model.predict_proba(features.to_numpy())[:, 1]
    pandas_df["anomaly_score"] = scores
    result_df = spark.createDataFrame(pandas_df)
    result_df.write.mode("overwrite").parquet("lake/gold/anomaly_scores")
    spark.stop()


if __name__ == "__main__":
    main()
