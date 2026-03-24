"""Batch inference utilities."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image


class BatchInferenceRunner:
    """Runner for batch inference on large datasets."""

    def __init__(
        self,
        predictor: Any,
        output_dir: Union[str, Path] = "./predictions",
        save_format: str = "csv",
    ):
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.save_format = save_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        return_probs: bool = False,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Run batch inference on list of images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size
            return_probs: Whether to return probabilities
            progress: Show progress bar

        Returns:
            DataFrame with predictions
        """
        results = []
        total = len(image_paths)

        iterator = range(0, total, batch_size)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=(total + batch_size - 1) // batch_size, desc="Batch inference")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, total)
            batch_paths = image_paths[start_idx:end_idx]

            batch_results = self.predictor.predict_batch(
                batch_paths,
                batch_size=batch_size,
                return_probs=return_probs,
            )

            for path, result in zip(batch_paths, batch_results):
                row = {
                    "image_path": str(path),
                    "predicted_class": result["class_name"],
                    "class_index": result["class_index"],
                    "confidence": result["confidence"],
                }
                if return_probs:
                    row.update(result.get("probabilities", {}))
                results.append(row)

        df = pd.DataFrame(results)
        return df

    def run_from_directory(
        self,
        directory: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        recursive: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Run inference on all images in a directory.

        Args:
            directory: Directory containing images
            extensions: Image file extensions to include
            recursive: Search subdirectories
            **kwargs: Arguments for run()

        Returns:
            DataFrame with predictions
        """
        directory = Path(directory)
        image_paths = []

        if recursive:
            for ext in extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(f"No images found in {directory}")

        return self.run(image_paths, **kwargs)

    def save_results(
        self,
        df: pd.DataFrame,
        filename: str = "predictions",
    ) -> Path:
        """Save predictions to file.

        Args:
            df: DataFrame with predictions
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        if self.save_format == "csv":
            output_path = self.output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)
        elif self.save_format == "json":
            output_path = self.output_dir / f"{filename}.json"
            df.to_json(output_path, orient="records", indent=2)
        elif self.save_format == "parquet":
            output_path = self.output_dir / f"{filename}.parquet"
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {self.save_format}")

        return output_path
