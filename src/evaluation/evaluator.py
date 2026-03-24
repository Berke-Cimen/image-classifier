"""Evaluation pipeline."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import MetricsCalculator


class Evaluator:
    """Evaluation pipeline for image classification models."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.

        Args:
            dataloader: DataLoader to evaluate on
            return_predictions: Whether to return all predictions

        Returns:
            Dictionary of metrics
        """
        metrics_calc = MetricsCalculator(
            num_classes=self.num_classes,
            class_names=self.class_names,
        )

        all_predictions: List[int] = []
        all_labels: List[int] = []
        all_probabilities: List[np.ndarray] = []
        all_paths: List[str] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                metrics_calc.update(predictions, labels, probabilities)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probabilities.append(probabilities.cpu().numpy())
                all_paths.extend(batch.get("path", [None] * len(labels)))

        results = metrics_calc.get_all_metrics()

        if return_predictions:
            results["predictions"] = {
                "labels": all_labels,
                "predictions": all_predictions,
                "probabilities": np.vstack(all_probabilities).tolist(),
                "paths": all_paths,
            }

        return results

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 10),
    ) -> plt.Figure:
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save figure
            normalize: Whether to normalize
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kw={"label": "Count" if not normalize else "Proportion"},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot per-class metrics as bar chart.

        Args:
            per_class_metrics: Per-class metrics dictionary
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[c]["precision"] for c in classes]
        recall = [per_class_metrics[c]["recall"] for c in classes]
        f1 = [per_class_metrics[c]["f1"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1 Score", alpha=0.8)

        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Union[str, Path],
        normalize_cm: bool = True,
    ) -> Dict[str, Any]:
        """Generate full evaluation report with visualizations.

        Args:
            dataloader: DataLoader to evaluate on
            output_dir: Directory to save report and figures
            normalize_cm: Whether to normalize confusion matrix

        Returns:
            Dictionary of all results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Evaluating model...")
        results = self.evaluate(dataloader, return_predictions=True)

        print("Generating confusion matrix...")
        cm = results["confusion_matrix"]
        self.plot_confusion_matrix(
            cm,
            save_path=output_dir / "confusion_matrix.png",
            normalize=normalize_cm,
        )
        self.plot_confusion_matrix(
            cm,
            save_path=output_dir / "confusion_matrix_raw.png",
            normalize=False,
        )

        print("Generating per-class metrics...")
        self.plot_per_class_metrics(
            results["per_class"],
            save_path=output_dir / "per_class_metrics.png",
        )

        print("Saving results...")
        import json

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        summary = {k: v for k, v in results.items() if k not in ["predictions", "confusion_matrix", "per_class"]}
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
        print("=" * 50)

        return results
