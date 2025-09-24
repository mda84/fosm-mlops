"""SageMaker deployment stub."""

from __future__ import annotations

import argparse

import sagemaker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy trained model to SageMaker endpoint")
    parser.add_argument("--model-data", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--endpoint-name", default="fosm-mlops-endpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = sagemaker.Session()
    model = sagemaker.tensorflow.model.TensorFlowModel(
        model_data=args.model_data,
        role=args.role,
        framework_version="2.11",
        py_version="py310",
        entry_point="scripts/train.py",
    )
    predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=args.endpoint_name)
    print(f"Deployed endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
