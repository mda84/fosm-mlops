"""CLI entry-point for model training."""

from fosm_mlops.pipelines.train_pipeline import main as hydra_main


if __name__ == "__main__":
    hydra_main()
