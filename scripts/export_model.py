#!/usr/bin/env python
"""Export model to ONNX or TorchScript."""
import argparse
import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model_exports",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "both"],
        default="onnx",
        help="Export format",
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
        "--image-size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version",
    )
    return parser.parse_args()


def export_onnx(model, output_path, image_size, opset_version=11):
    """Export model to ONNX format."""
    import numpy as np

    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to: {output_path}")


def export_torchscript(model, output_path, image_size):
    """Export model to TorchScript format."""
    dummy_input = torch.randn(1, 3, image_size, image_size)

    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(str(output_path))

    print(f"TorchScript model saved to: {output_path}")


def main():
    """Main export function."""
    args = parse_args()

    print("=" * 60)
    print("Model Export")
    print("=" * 60)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_name = args.model_name or checkpoint.get("model_name", "efficientnet_b3")
    num_classes = args.num_classes

    print(f"Model: {model_name}")
    print(f"Classes: {num_classes}")
    print(f"Image size: {args.image_size}")

    from src.models.factory import ModelFactory

    model = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ["onnx", "both"]:
        onnx_path = output_dir / f"{model_name}.onnx"
        print(f"\nExporting to ONNX...")
        export_onnx(model, onnx_path, args.image_size, args.opset_version)

    if args.format in ["torchscript", "both"]:
        ts_path = output_dir / f"{model_name}.pt"
        print(f"\nExporting to TorchScript...")
        export_torchscript(model, ts_path, args.image_size)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
