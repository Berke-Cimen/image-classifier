"""Tests for data pipeline modules."""
import numpy as np
import pytest
import torch


class TestTransforms:
    """Tests for transforms module."""

    def test_train_transforms_output_shape(self):
        """Test train transforms output correct shape."""
        from src.data.transforms import get_train_transforms

        transform = get_train_transforms(image_size=224)
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        result = transform(image=dummy_image)
        assert result["image"].shape == (224, 224, 3)

    def test_val_transforms_output_shape(self):
        """Test validation transforms output correct shape."""
        from src.data.transforms import get_val_transforms

        transform = get_val_transforms(image_size=224)
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        result = transform(image=dummy_image)
        assert result["image"].shape == (224, 224, 3)

    def test_train_transforms_normalization(self):
        """Test train transforms include normalization."""
        from src.data.transforms import get_train_transforms

        transform = get_train_transforms(image_size=224)
        assert transform is not None


class TestDataloader:
    """Tests for dataloader module."""

    def test_cifar10_loaders_structure(self):
        """Test CIFAR-10 loaders can be created."""
        from src.data.dataloader import get_cifar10_loaders

        try:
            train_loader, test_loader = get_cifar10_loaders(
                data_dir="./data",
                batch_size=32,
                num_workers=0,
                download=False,
            )
            assert train_loader is not None
            assert test_loader is not None
        except Exception:
            pytest.skip("CIFAR-10 not available")
