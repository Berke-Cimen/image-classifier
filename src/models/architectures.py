"""Custom model architectures."""
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """Simple custom CNN for image classification."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for custom architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    """Custom ResNet-like architecture."""

    def __init__(
        self,
        num_classes: int = 10,
        layers: List[int] = [2, 2, 2, 2],
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            ResidualBlock(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class HybridModel(nn.Module):
    """Hybrid model combining pretrained features with custom head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 10,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        pooling: str = "avg",
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

        self.feature_dim = self._get_feature_dim()

        hidden_dim = hidden_dim or self.feature_dim

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _get_feature_dim(self) -> int:
        if hasattr(self.backbone, "num_features"):
            return self.backbone.num_features
        if hasattr(self.backbone, "classifier"):
            if isinstance(self.backbone.classifier, nn.Linear):
                return self.backbone.classifier.in_features
        raise ValueError("Could not determine feature dimension")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        if len(features.shape) == 4:
            if self.pooling == "avg":
                features = F.adaptive_avg_pool2d(features, 1)
            elif self.pooling == "max":
                features = F.adaptive_max_pool2d(features, 1)
            features = torch.flatten(features, 1)

        logits = self.head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) == 4:
                if self.pooling == "avg":
                    features = F.adaptive_avg_pool2d(features, 1)
                elif self.pooling == "max":
                    features = F.adaptive_max_pool2d(features, 1)
                features = torch.flatten(features, 1)
        return features