"""DataLoader factory."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import ImageDataset, download_cifar10
from .transforms import TransformConfig, get_train_transforms, get_val_transforms


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Any] = None,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    return_path: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader with appropriate transforms.

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        transform: Custom transform (if None, uses default train transforms)
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        return_path: Whether to return image paths
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to keep workers alive

    Returns:
        DataLoader instance
    """
    if transform is None:
        transform = get_train_transforms(image_size=image_size, mean=mean, std=std)

    dataset = ImageDataset(
        data_dir=data_dir,
        transform=transform,
        return_path=return_path,
    )

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": True,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }

    return DataLoader(dataset, **dataloader_kwargs)


def create_train_val_loaders(
    data_dir: Union[str, Path],
    batch_size: int,
    val_split: float = 0.2,
    num_workers: int = 4,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    transform_config: Optional[TransformConfig] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a single directory.

    Assumes data_dir contains class subdirectories with all images mixed.
    Splits by sampling 20% of each class for validation.

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        val_split: Fraction of data for validation
        num_workers: Number of workers
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        transform_config: Augmentation config

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_dir = Path(data_dir)

    train_transform = get_train_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        config=transform_config,
    )
    val_transform = get_val_transforms(image_size=image_size, mean=mean, std=std)

    full_dataset = ImageDataset(data_dir=data_dir, transform=None)

    train_samples = []
    val_samples = []

    for class_name, class_idx in full_dataset.class_to_idx.items():
        class_images = [
            (path, class_idx)
            for path, idx in full_dataset.samples
            if idx == class_idx
        ]
        num_val = max(1, int(len(class_images) * val_split))
        train_samples.extend(class_images[num_val:])
        val_samples.extend(class_images[:num_val])

    train_dataset = ImageDataset(data_dir=data_dir, transform=train_transform)
    train_dataset.samples = train_samples

    val_dataset = ImageDataset(data_dir=data_dir, transform=val_transform)
    val_dataset.samples = val_samples

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def get_cifar10_loaders(
    data_dir: Union[str, Path] = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 32,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test loaders.

    Args:
        data_dir: Directory to save/load dataset
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size (CIFAR-10 is 32x32)
        download: Whether to download if not present

    Returns:
        Tuple of (train_loader, test_loader)
    """
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as T

    if download:
        download_cifar10(data_dir)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_path = Path(data_dir)
    train_dataset = tv_datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=train_transform,
        download=download,
    )
    test_dataset = tv_datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=val_transform,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader
