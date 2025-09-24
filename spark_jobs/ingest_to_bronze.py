"""Spark job to enforce schema and write bronze/silver tables."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, LongType, StructField, StructType


RAW_SCHEMA = StructType(
    [
        StructField("time", DoubleType(), False),
        StructField("sensor_id", LongType(), False),
        StructField("value", DoubleType(), False),
    ]
)


def main() -> None:
    spark = SparkSession.builder.appName("fosm-ingest").getOrCreate()
    raw_df = spark.read.schema(RAW_SCHEMA).parquet("data/raw")
    raw_df.write.mode("overwrite").parquet("lake/bronze/signals")

    silver_df = raw_df.withColumn("value_zscore", (col("value") - col("value").mean()) / col("value").stddev())
    silver_df.write.mode("overwrite").parquet("lake/silver/signals")
    spark.stop()


if __name__ == "__main__":
    main()
