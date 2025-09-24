# Model Card Template

## Model Details
- **Model name**: {{ model_name }}
- **Version**: {{ version }}
- **Date**: {{ date }}
- **Owners**: FiberOps MLOps Team

## Intended Use
- **Primary purpose**: Detect anomalies (leaks, tampering, ground movement) from fiber-optic acoustic data
- **Input data**: Sliding window spectral features derived from DAS signals
- **Output**: Probability of anomaly + tuned classification threshold

## Training Data
- Synthetic signals generated via `gen_synthetic.py`
- Events include: normal flow, leak bursts, third-party interference, ground movement
- Sampling rate configurable (default 200Hz)

## Evaluation Data
- Held-out split via time-aware segmentation
- Metrics: ROC-AUC, Average Precision, F1/Recall threshold tuning, confusion matrix

## Ethical Considerations
- False negatives can lead to environmental damage; thresholds favor recall.
- Synthetic data approximates real behavior; validate on production data before deployment.

## Caveats & Recommendations
- Re-train when sensor configuration changes or new event types emerge.
- Monitor drift via `src/fosm_mlops/monitoring/drift.py` outputs.
- Document hyperparameters and data windows in MLflow for reproducibility.
