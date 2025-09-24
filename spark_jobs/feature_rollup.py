"""Aggregate high-rate sensor data into feature windows using PySpark."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, collect_list, col, max as spark_max, min as spark_min


WINDOW_SECONDS = 5


def main() -> None:
    spark = SparkSession.builder.appName("fosm-feature-rollup").getOrCreate()
    df = spark.read.parquet("lake/silver/signals")
    df = df.withColumn("window_id", (col("time") / WINDOW_SECONDS).cast("int"))

    rollup = (
        df.groupBy("sensor_id", "window_id")
        .agg(
            avg("value").alias("avg_value"),
            spark_max("value").alias("max_value"),
            spark_min("value").alias("min_value"),
            collect_list("value").alias("values"),
        )
        .withColumn("peak_to_peak", col("max_value") - col("min_value"))
    )
    rollup.write.mode("overwrite").parquet("lake/gold/features")
    spark.stop()


if __name__ == "__main__":
    main()
