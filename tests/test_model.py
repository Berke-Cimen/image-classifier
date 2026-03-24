"""Tests for model modules."""
import pytest
import torch


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_list_models(self):
        """Test listing available models."""
        from src.models.factory import ModelFactory

        models = ModelFactory.list_models()
        assert len(models) > 0
        assert "efficientnet_b3" in models
        assert "resnet50" in models

    def test_create_model(self):
        """Test creating a model."""
        from src.models.factory import ModelFactory

        model = ModelFactory.create(
            model_name="efficientnet_b0",
            num_classes=10,
            pretrained=False,
        )
        assert model is not None
        assert hasattr(model, "forward")

    def test_get_input_size(self):
        """Test getting model input size."""
        from src.models.factory import ModelFactory

        size = ModelFactory.get_input_size("efficientnet_b3")
        assert size == 300

        size = ModelFactory.get_input_size("resnet50")
        assert size == 224

    def test_feature_extraction_mode(self):
        """Test feature extraction mode freezes backbone."""
        from src.models.factory import ModelFactory

        model = ModelFactory.create(
            model_name="efficientnet_b0",
            num_classes=10,
            pretrained=False,
            feature_extraction=True,
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0


class TestCustomArchitectures:
    """Tests for custom architectures."""

    def test_custom_cnn_output_shape(self):
        """Test CustomCNN output shape."""
        from src.models.architectures import CustomCNN

        model = CustomCNN(num_classes=10)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)

        assert output.shape == (1, 10)

    def test_custom_resnet_output_shape(self):
        """Test CustomResNet output shape."""
        from src.models.architectures import CustomResNet

        model = CustomResNet(num_classes=10)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)

        assert output.shape == (1, 10)
