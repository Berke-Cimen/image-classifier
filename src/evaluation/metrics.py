"""Evaluation metrics for image classification."""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """Calculate classification metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_predictions: List[int] = []
        self.all_labels: List[int] = []
        self.all_probabilities: List[np.ndarray] = []

    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> None:
        """Update metrics with batch results.

        Args:
            predictions: Predicted class indices (batch_size,)
            labels: Ground truth class indices (batch_size,)
            probabilities: Predicted probabilities (batch_size, num_classes)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()

        self.all_predictions.extend(predictions.flatten().tolist())
        self.all_labels.extend(labels.flatten().tolist())

        if probabilities is not None:
            self.all_probabilities.append(probabilities)

    def compute_accuracy(self) -> float:
        """Compute top-1 accuracy."""
        return accuracy_score(self.all_labels, self.all_predictions)

    def compute_top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-k accuracy.

        Args:
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy as a float
        """
        if not self.all_probabilities:
            return 0.0

        all_probs = np.vstack(self.all_probabilities)
        all_labels = np.array(self.all_labels)

        top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(all_labels):
            if label in top_k_preds[i]:
                correct += 1

        return correct / len(all_labels)

    def compute_precision(self, average: str = "macro") -> Union[float, np.ndarray]:
        """Compute precision score.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            Precision score(s)
        """
        return precision_score(
            self.all_labels,
            self.all_predictions,
            average=average if average != "micro" else "micro",
            zero_division=0,
        )

    def compute_recall(self, average: str = "macro") -> Union[float, np.ndarray]:
        """Compute recall score.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            Recall score(s)
        """
        return recall_score(
            self.all_labels,
            self.all_predictions,
            average=average if average != "micro" else "micro",
            zero_division=0,
        )

    def compute_f1(self, average: str = "macro") -> Union[float, np.ndarray]:
        """Compute F1 score.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            F1 score(s)
        """
        return f1_score(
            self.all_labels,
            self.all_predictions,
            average=average if average != "micro" else "micro",
            zero_division=0,
        )

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(self.all_labels, self.all_predictions)

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, F1."""
        per_class_precision = precision_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0,
        )
        per_class_recall = recall_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0,
        )
        per_class_f1 = f1_score(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0,
        )

        results = {}
        for i, class_name in enumerate(self.class_names):
            results[class_name] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int(np.sum(np.array(self.all_labels) == i)),
            }

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "accuracy": self.compute_accuracy(),
            "top_5_accuracy": self.compute_top_k_accuracy(k=5) if self.num_classes > 5 else None,
            "precision_macro": float(self.compute_precision("macro")),
            "recall_macro": float(self.compute_recall("macro")),
            "f1_macro": float(self.compute_f1("macro")),
            "precision_weighted": float(self.compute_precision("weighted")),
            "recall_weighted": float(self.compute_recall("weighted")),
            "f1_weighted": float(self.compute_f1("weighted")),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics including per-class."""
        summary = self.get_summary()
        summary["per_class"] = self.compute_per_class_metrics()
        summary["confusion_matrix"] = self.compute_confusion_matrix()
        return summary
