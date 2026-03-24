#!/usr/bin/env python
"""Training script."""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Image Classification Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data/raw"),
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    """Main training function."""
    args = parse_args()

    from src.utils.helpers import set_seed
    set_seed(args.seed)

    print("=" * 60)
    print("Image Classification Training")
    print("=" * 60)

    train_config = load_config(args.config)
    model_config = load_config(args.model_config)

    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"\nLoading model: {model_config['model']['name']}")
    from src.models.factory import ModelFactory

    model = ModelFactory.create(
        model_name=model_config["model"]["name"],
        num_classes=model_config["model"]["num_classes"],
        pretrained=model_config["model"]["pretrained"],
        drop_rate=model_config["model"].get("drop_rate", 0.0),
        drop_path_rate=model_config["model"].get("drop_path_rate", 0.0),
    )

    image_size = model_config["image"]["size"]
    mean = tuple(model_config["image"]["mean"])
    std = tuple(model_config["image"]["std"])

    print(f"\nLoading data from: {args.data_dir}")
    from src.data.dataloader import create_train_val_loaders
    from src.data.transforms import TransformConfig

    transform_config = TransformConfig(**train_config.get("augmentation", {}).get("train", {}))

    train_loader, val_loader = create_train_val_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size or train_config["training"]["batch_size"],
        val_split=0.2,
        num_workers=4,
        image_size=image_size,
        mean=mean,
        std=std,
        transform_config=transform_config,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nSetting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr or train_config["training"]["learning_rate"],
        weight_decay=train_config["training"]["weight_decay"],
    )

    from src.training.scheduler import create_scheduler

    scheduler = create_scheduler(
        scheduler_type=train_config["training"]["scheduler"]["type"],
        optimizer=optimizer,
        num_epochs=args.epochs or train_config["training"]["num_epochs"],
        warmup_epochs=train_config["training"]["warmup_epochs"],
        min_lr=train_config["training"]["scheduler"]["min_lr"],
    )

    print("\nInitializing trainer...")
    from src.training.trainer import Trainer
    from src.training.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ProgressLogger,
    )

    callbacks = [
        ModelCheckpoint(
            output_dir=args.output_dir,
            monitor="val_accuracy",
            mode="max",
            save_best=True,
            save_last=True,
        ),
        CSVLogger(output_dir=args.output_dir),
        ProgressLogger(verbose=True),
    ]

    if train_config.get("early_stopping"):
        callbacks.append(
            EarlyStopping(
                patience=train_config["early_stopping"]["patience"],
                min_delta=train_config["early_stopping"]["min_delta"],
                mode="max",
            )
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        callbacks=callbacks,
        use_amp=True,
    )

    print(f"\nStarting training for {args.epochs or train_config['training']['num_epochs']} epochs...")
    print(f"Output directory: {args.output_dir}")

    history = trainer.train(num_epochs=args.epochs or train_config["training"]["num_epochs"])

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {trainer.best_metric:.2f}%")


if __name__ == "__main__":
    main()
