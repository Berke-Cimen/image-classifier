"""Image dataset implementations."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Folder-based image dataset for classification.

    Expected structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        return_path: bool = False,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_path = return_path
        self.extensions = extensions

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_samples_list()

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Find all classes in data directory."""
        classes = sorted(
            [
                d.name
                for d in self.data_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )
        if not classes:
            raise ValueError(f"No class directories found in {self.data_dir}")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_samples_list(self) -> List[Tuple[Path, int]]:
        """Create list of (image_path, class_index) tuples."""
        samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, class_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, int]]:
        img_path, label = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.return_path:
            return {"image": image, "label": label, "path": str(img_path)}

        return {"image": image, "label": label}

    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.classes[idx]


class ImageNetSubsetDataset(Dataset):
    """Download and use a subset of ImageNet for quick experiments."""

    # We'll implement download functionality separately
    pass


def download_cifar10(data_dir: Union[str, Path]) -> Path:
    """Download CIFAR-10 dataset.

    Args:
        data_dir: Directory to save dataset

    Returns:
        Path to extracted dataset directory
    """
    import torchvision.datasets as datasets

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 to {data_dir}...")
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)

    return data_dir / "cifar-10-batches-py"
