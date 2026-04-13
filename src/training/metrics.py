"""Metrics computation for fraud detection evaluation."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsAccumulator:
    """Accumulates predictions across batches and computes epoch-level metrics."""

    all_labels: list = field(default_factory=list)
    all_probs: list = field(default_factory=list)
    all_losses: list = field(default_factory=list)

    def update(self, labels: np.ndarray, probs: np.ndarray, loss: float) -> None:
        """Add a batch of predictions."""
        self.all_labels.extend(labels.tolist())
        self.all_probs.extend(probs.tolist())
        self.all_losses.append(loss)

    def compute(self, threshold: float = 0.5) -> dict:
        """Compute comprehensive metrics.

        Returns:
            dict with: loss, auroc, auprc, f1, precision, recall,
                       confusion_matrix, classification_report, optimal_threshold.
        """
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        preds = (probs >= threshold).astype(int)

        metrics = {
            "loss": float(np.mean(self.all_losses)),
            "n_samples": len(labels),
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
        }

        # Guard against edge cases
        if len(np.unique(labels)) < 2:
            metrics.update({"auroc": 0.0, "auprc": 0.0, "f1": 0.0,
                            "precision": 0.0, "recall": 0.0})
            return metrics

        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
        metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
        metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
        metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
        metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
        metrics["classification_report"] = classification_report(
            labels, preds, target_names=["normal", "fraud"], zero_division=0
        )

        # Find optimal threshold (maximises F1)
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        metrics["optimal_threshold"] = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

        return metrics

    def reset(self) -> None:
        """Clear accumulated data."""
        self.all_labels.clear()
        self.all_probs.clear()
        self.all_losses.clear()
