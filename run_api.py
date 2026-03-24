#!/usr/bin/env python
"""Run API server locally."""
import argparse
import os
import sys
from pathlib import Path

import torch
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Image Classification API")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "efficientnet_b3"),
        help="Model name (default: efficientnet_b3)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.getenv("MODEL_WEIGHTS", "pretrained"),
        help="Model weights path or 'pretrained' (default: pretrained)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("NUM_CLASSES", "10")),
        help="Number of classes (default: 10)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.getenv("IMAGE_SIZE", "300")),
        help="Input image size (default: 300)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="API host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="API port (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of workers (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "auto"),
        help="Device to use: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Class names (default: class_0, class_1, ...)",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device with auto-detection.

    Args:
        device_str: 'cuda', 'cpu', or 'auto'

    Returns:
        torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """Get device information string.

    Args:
        device: torch device

    Returns:
        Device info string
    """
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda ({gpu_name}, {gpu_memory:.1f}GB)"
    return "cpu"


def create_class_names(num_classes: int, custom_names=None) -> list:
    """Create class names list.

    Args:
        num_classes: Number of classes
        custom_names: Optional list of custom names

    Returns:
        List of class names
    """
    if custom_names:
        return custom_names
    return [f"class_{i}" for i in range(num_classes)]


def load_model(model_name: str, weights: str, num_classes: int, device: torch.device):
    """Load model with weights.

    Args:
        model_name: Model architecture name
        weights: Path to weights or 'pretrained'
        num_classes: Number of classes
        device: Target device

    Returns:
        Loaded model
    """
    from src.models.factory import ModelFactory

    print(f"Loading model: {model_name}")
    pretrained = weights == "pretrained"
    model = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )

    if weights != "pretrained" and Path(weights).exists():
        print(f"Loading weights from: {weights}")
        checkpoint = torch.load(weights, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Image Classification API")
    print("=" * 60)

    device = get_device(args.device)
    print(f"Device: {get_device_info(device)}")

    device_str = str(device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model: {args.model}")
    print(f"Classes: {args.num_classes}")
    print(f"Image size: {args.image_size}")

    try:
        model = load_model(args.model, args.weights, args.num_classes, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    class_names = create_class_names(args.num_classes, args.classes)

    print("\nInitializing predictor...")
    from src.inference.predictor import Predictor

    predictor = Predictor(
        model=model,
        class_names=class_names,
        image_size=args.image_size,
        device=device,
        warmup_steps=3,
    )

    print("Warming up model...")
    import numpy as np
    dummy_img = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
    _ = predictor.predict(dummy_img)
    print("Warmup complete")

    from api.main import create_app

    app = create_app(
        model=model,
        class_names=class_names,
        predictor=predictor,
        device=device,
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )

    print("\n" + "=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
