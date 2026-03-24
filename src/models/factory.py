"""Model factory using timm for pretrained models."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from timm.models import create_model


class ModelFactory:
    """Factory for creating image classification models."""

    # Registry of available pretrained models
    MODEL_REGISTRY = {
        # EfficientNet family
        "efficientnet_b0": {"input_size": 224, "family": "efficientnet"},
        "efficientnet_b1": {"input_size": 240, "family": "efficientnet"},
        "efficientnet_b2": {"input_size": 260, "family": "efficientnet"},
        "efficientnet_b3": {"input_size": 300, "family": "efficientnet"},
        "efficientnet_b4": {"input_size": 380, "family": "efficientnet"},
        "efficientnet_b5": {"input_size": 456, "family": "efficientnet"},
        "efficientnet_b6": {"input_size": 528, "family": "efficientnet"},
        "efficientnet_b7": {"input_size": 600, "family": "efficientnet"},
        # ResNet family
        "resnet18": {"input_size": 224, "family": "resnet"},
        "resnet34": {"input_size": 224, "family": "resnet"},
        "resnet50": {"input_size": 224, "family": "resnet"},
        "resnet101": {"input_size": 224, "family": "resnet"},
        "resnet152": {"input_size": 224, "family": "resnet"},
        # MobileNet family
        "mobilenetv2_100": {"input_size": 224, "family": "mobilenet"},
        "mobilenetv3_large_100": {"input_size": 224, "family": "mobilenet"},
        "mobilenetv3_small_100": {"input_size": 224, "family": "mobilenet"},
        # ViT family
        "vit_tiny_patch16_224": {"input_size": 224, "family": "vit"},
        "vit_small_patch16_224": {"input_size": 224, "family": "vit"},
        "vit_base_patch16_224": {"input_size": 224, "family": "vit"},
        "vit_large_patch16_224": {"input_size": 224, "family": "vit"},
        # ConvNeXt
        "convnext_tiny": {"input_size": 224, "family": "convnext"},
        "convnext_small": {"input_size": 224, "family": "convnext"},
        "convnext_base": {"input_size": 224, "family": "convnext"},
        "convnext_large": {"input_size": 224, "family": "convnext"},
    }

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names."""
        return list(cls.MODEL_REGISTRY.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in cls.MODEL_REGISTRY:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )
        return cls.MODEL_REGISTRY[model_name]

    @classmethod
    def create(
        cls,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        feature_extraction: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Create a model.

        Args:
            model_name: Name of the model (e.g., 'efficientnet_b3')
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth drop path rate
            feature_extraction: If True, freeze all layers except classifier
            **kwargs: Additional arguments for timm.create_model

        Returns:
            Model instance
        """
        info = cls.get_model_info(model_name)

        model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        if feature_extraction:
            model = cls._enable_feature_extraction(model)

        return model

    @classmethod
    def _enable_feature_extraction(cls, model: nn.Module) -> nn.Module:
        """Freeze all layers except the classifier head."""
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "head"):
            if hasattr(model.head, "classifier"):
                for param in model.head.classifier.parameters():
                    param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True

        return model

    @classmethod
    def create_with_custom_head(
        cls,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> nn.Module:
        """Create model with custom classification head.

        Args:
            model_name: Base model name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            hidden_dim: Hidden layer dimension (if None, uses model default)
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'silu')

        Returns:
            Model with custom head
        """
        model = cls.create(
            model_name=model_name,
            num_classes=0,
            pretrained=pretrained,
        )

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu6": nn.ReLU6,
        }
        act_fn = activation_map.get(activation, nn.ReLU)

        feature_dim = cls._get_feature_dim(model)
        hidden_dim = hidden_dim or feature_dim

        custom_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        model.classifier = custom_head

        return model

    @classmethod
    def _get_feature_dim(cls, model: nn.Module) -> int:
        """Get the feature dimension of a model."""
        if hasattr(model, "num_features"):
            return model.num_features

        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                return model.classifier.in_features

        if hasattr(model, "fc"):
            if isinstance(model.fc, nn.Linear):
                return model.fc.in_features

        if hasattr(model, "head"):
            if hasattr(model.head, "dense"):
                return model.head.dense.in_features

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.forward_features(dummy_input)
            if len(features.shape) == 4:
                return features.shape[1]
            elif len(features.shape) == 2:
                return features.shape[1]

        raise ValueError("Could not determine feature dimension")

    @classmethod
    def get_input_size(cls, model_name: str) -> int:
        """Get recommended input size for a model."""
        info = cls.get_model_info(model_name)
        return info["input_size"]