"""Export TensorFlow model to SavedModel directory."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def export_model(model_path: str, export_dir: str) -> None:
    model = tf.keras.models.load_model(model_path)
    export_path = Path(export_dir)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(export_path)
    print(f"Exported model to {export_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export TF model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--export-dir", required=True)
    args = parser.parse_args()
    export_model(args.model_path, args.export_dir)
