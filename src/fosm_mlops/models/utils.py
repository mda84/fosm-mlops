"""Model utility helpers used across training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def persist_model(model: object, path: Path | str) -> None:
    """Serialise a fitted model to disk."""

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)


def load_model(path: Path | str) -> object:
    """Load a model persisted with :func:`persist_model`."""

    return joblib.load(path)


def tune_threshold(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    metric: str = "f1",
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Search for the best probability threshold."""

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true_arr = np.asarray(y_true)
    score_arr = np.asarray(scores)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        y_pred = (score_arr >= threshold).astype(int)
        current_score = _threshold_metric(metric, y_true_arr, y_pred, score_arr)
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def _threshold_metric(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> float:
    metric = metric.lower()
    if metric == "f1":
        return metrics.f1_score(y_true, y_pred, zero_division=0)
    if metric == "precision":
        return metrics.precision_score(y_true, y_pred, zero_division=0)
    if metric == "recall":
        return metrics.recall_score(y_true, y_pred, zero_division=0)
    if metric == "roc_auc":
        return metrics.roc_auc_score(y_true, scores)
    if metric == "pr_auc":
        return metrics.average_precision_score(y_true, scores)
    msg = f"Unsupported threshold metric: {metric}"
    raise ValueError(msg)


def classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a suite of classification metrics."""

    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    y_pred = (scores_arr >= threshold).astype(int)
    metrics_dict = {
        "accuracy": metrics.accuracy_score(y_true_arr, y_pred),
        "precision": metrics.precision_score(
            y_true_arr, y_pred, zero_division=0
        ),
        "recall": metrics.recall_score(y_true_arr, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true_arr, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true_arr, scores_arr),
        "pr_auc": metrics.average_precision_score(y_true_arr, scores_arr),
    }
    return metrics_dict


def save_metrics_plots(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    output_dir: Path,
    threshold: float,
) -> None:
    """Persist ROC, PR and confusion matrix plots to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    y_pred = (scores_arr >= threshold).astype(int)

    fpr, tpr, _ = metrics.roc_curve(y_true_arr, scores_arr)
    precision, recall, _ = metrics.precision_recall_curve(y_true_arr, scores_arr)
    cm = metrics.confusion_matrix(y_true_arr, y_pred)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics.auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR AUC = {metrics.auc(recall, precision):.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.savefig(output_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
