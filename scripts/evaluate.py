#!/usr/bin/env python
"""Evaluation script."""
import argparse
import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Image Classification Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data/raw"),
        help="Path to dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model architecture name",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Class names",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("Image Classification Evaluation")
    print("=" * 60)

    device = get_device(args.device)
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = args.model_name or checkpoint.get("model_name", "efficientnet_b3")
    num_classes = args.num_classes

    print(f"Model: {model_name}")
    print(f"Classes: {num_classes}")

    from src.models.factory import ModelFactory

    model = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = args.class_names or [f"class_{i}" for i in range(num_classes)]

    print(f"\nLoading data from: {args.data_dir}")
    from src.data.dataloader import create_dataloader
    from src.data.transforms import get_val_transforms

    val_transform = get_val_transforms(image_size=args.image_size)

    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        transform=val_transform,
        image_size=args.image_size,
    )

    print(f"Validation batches: {len(dataloader)}")

    print("\nRunning evaluation...")
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    results = evaluator.generate_report(
        dataloader=dataloader,
        output_dir=args.output_dir,
        normalize_cm=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
