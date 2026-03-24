"""Inference pipeline for image classification."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Predictor:
    """Predictor class for image classification inference."""

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: Optional[torch.device] = None,
        use_amp: bool = False,
        warmup_steps: int = 3,
    ):
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.warmup_steps = warmup_steps

        self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=self.mean, std=self.std),
            A.ToFloat(max_value=255),
        ])

        self._is_warmed_up = False

    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference.

        Args:
            image: Image as path, PIL Image, or numpy array

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        augmented = self.transform(image=image)
        tensor = augmented["image"]

        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        return_probs: bool = False,
    ) -> Dict[str, Any]:
        """Predict class for a single image.

        Args:
            image: Image input
            return_probs: Whether to return all class probabilities

        Returns:
            Dictionary with prediction results
        """
        if not self._is_warmed_up:
            self._warmup()

        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = self.preprocess(image)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(tensor)
            else:
                outputs = self.model(tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        result = {
            "class_name": self.class_names[predicted_idx],
            "class_index": predicted_idx,
            "confidence": confidence,
        }

        if return_probs:
            result["probabilities"] = {
                name: prob.item()
                for name, prob in zip(self.class_names, probabilities[0])
            }

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Predict classes for multiple images.

        Args:
            images: List of images
            batch_size: Batch size for inference
            return_probs: Whether to return all class probabilities

        Returns:
            List of prediction dictionaries
        """
        if not self._is_warmed_up:
            self._warmup()

        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            tensors = [self.preprocess(img) for img in batch_images]
            batch_tensor = torch.cat(tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)

            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, dim=1)

            for j, (conf, idx) in enumerate(zip(confidences, predicted_indices)):
                result = {
                    "class_name": self.class_names[idx.item()],
                    "class_index": idx.item(),
                    "confidence": conf.item(),
                }
                if return_probs:
                    result["probabilities"] = {
                        name: prob.item()
                        for name, prob in zip(self.class_names, probabilities[j])
                    }
                results.append(result)

        return results

    def _warmup(self) -> None:
        """Warmup the model with dummy inputs."""
        dummy_input = torch.randn(1, 3, self.image_size, self.device)

        for _ in range(self.warmup_steps):
            with torch.no_grad():
                _ = self.model(dummy_input)

        self._is_warmed_up = True

    def get_inference_speed(self, num_iterations: int = 100) -> float:
        """Measure average inference speed in milliseconds.

        Args:
            num_iterations: Number of iterations for measurement

        Returns:
            Average inference time in ms
        """
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)

        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        avg_time_ms = (elapsed / num_iterations) * 1000

        return avg_time_ms
