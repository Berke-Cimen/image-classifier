"""Albumentations transform pipelines."""
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from pydantic import BaseModel, Field


class TransformConfig(BaseModel):
    """Transform configuration."""

    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotate_limit: int = 15
    scale_limit: float = 0.1
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_limit: float = 0.0
    saturation_limit: float = 0.0
    gaussian_blur_limit: int = 0
    random_gamma_limit: Tuple[float, float] = (80, 120)
    cutout_num_holes: int = 0
    cutout_max_size: int = 8


def get_train_transforms(
    image_size: int = 224,
    config: Optional[TransformConfig] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get training transforms with augmentation."""
    if config is None:
        config = TransformConfig()

    transforms = [
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        A.HorizontalFlip(p=config.horizontal_flip),
        A.VerticalFlip(p=config.vertical_flip),
    ]

    if config.rotate_limit > 0 or config.scale_limit > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=config.scale_limit,
                rotate_limit=config.rotate_limit,
                border_mode=0,  # BORDER_CONSTANT = 0
                value=0,
            )
        )

    color_augs = []
    if config.brightness_limit > 0 or config.contrast_limit > 0:
        color_augs.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=0.5,
            )
        )
    if config.hue_limit > 0 or config.saturation_limit > 0:
        color_augs.append(
            A.HueSaturationValue(
                hue_shift_limit=int(config.hue_limit * 255),
                sat_shift_limit=int(config.saturation_limit * 255),
                p=0.5,
            )
        )
    if config.gaussian_blur_limit > 0:
        color_augs.append(
            A.GaussianBlur(blur_limit=(3, config.gaussian_blur_limit), p=0.3)
        )

    if color_augs:
        transforms.append(A.OneOf(color_augs, p=0.5))

    if config.cutout_num_holes > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(config.cutout_num_holes, config.cutout_num_holes),
                hole_height_range=(config.cutout_max_size, config.cutout_max_size),
                hole_width_range=(config.cutout_max_size, config.cutout_max_size),
                fill=0,
                p=0.3,
            )
        )

    transforms.extend([
        A.Normalize(mean=mean, std=std),
        A.ToFloat(max_value=255),
    ])

    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get validation transforms (resize + normalize only)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        A.ToFloat(max_value=255),
    ])


def get_test_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get test transforms (same as validation)."""
    return get_val_transforms(image_size=image_size, mean=mean, std=std)
