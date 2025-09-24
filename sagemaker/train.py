"""SageMaker training stub."""

from __future__ import annotations

import argparse

import sagemaker
from sagemaker.tensorflow import TensorFlow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator = TensorFlow(
        entry_point="scripts/train.py",
        role=args.role,
        framework_version="2.11",
        py_version="py310",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        hyperparameters={"model.name": "classical/random_forest"},
        output_path=f"s3://{args.bucket}/fosm-mlops/output",
    )
    estimator.fit({"training": f"s3://{args.bucket}/fosm-mlops/training"}, wait=False)


if __name__ == "__main__":
    main()
