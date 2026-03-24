"""Tests for inference modules."""
import numpy as np
import pytest
import torch
import torch.nn as nn


class TestPredictor:
    """Tests for Predictor class."""

    def test_predictor_initialization(self):
        """Test Predictor initialization."""
        from src.inference.predictor import Predictor

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 10, 1)

            def forward(self, x):
                return self.conv(x)

        model = MockModel()
        class_names = ["class_0", "class_1"]

        predictor = Predictor(
            model=model,
            class_names=class_names,
            image_size=224,
        )

        assert predictor.class_names == class_names
        assert predictor.image_size == 224

    def test_preprocess(self):
        """Test image preprocessing."""
        from src.inference.predictor import Predictor

        class MockModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=(2, 3))

        model = MockModel()
        predictor = Predictor(model=model, class_names=["a", "b"], image_size=224)

        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        tensor = predictor.preprocess(dummy_image)

        assert tensor.shape == (1, 3, 224, 224)
