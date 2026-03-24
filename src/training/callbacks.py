"""Training callbacks."""
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class Callback:
    """Base callback class."""

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Check if training should stop."""
        if self.best_metric is None:
            self.best_metric = metric
            return

        if self.mode == "max":
            improved = metric > self.best_metric + self.min_delta
        else:
            improved = metric < self.best_metric - self.min_delta

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training")


class ModelCheckpoint(Callback):
    """Model checkpoint callback."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.verbose = verbose
        self.best_metric = None

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Save checkpoint if needed."""
        if self.save_best:
            if self.best_metric is None:
                self.best_metric = metric
                should_save = True
            else:
                if self.mode == "max":
                    should_save = metric > self.best_metric
                else:
                    should_save = metric < self.best_metric

            if should_save:
                self.best_metric = metric
                trainer.save_checkpoint("best.pth")
                if self.verbose:
                    print(f"Saved best model with {self.monitor}: {metric:.4f}")

        if self.save_last:
            trainer.save_checkpoint("last.pth")


class LRSchedulerCallback(Callback):
    """Learning rate scheduler callback."""

    def __init__(
        self,
        scheduler: Any,
        step_after_epoch: bool = True,
        verbose: bool = False,
    ):
        self.scheduler = scheduler
        self.step_after_epoch = step_after_epoch
        self.verbose = verbose

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Step the scheduler."""
        if self.step_after_epoch:
            if hasattr(self.scheduler, "step"):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            if self.verbose:
                lr = trainer.optimizer.param_groups[0]["lr"]
                print(f"Learning rate: {lr:.6f}")


class CSVLogger(Callback):
    """CSV logger callback."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        filename: str = "training_log.csv",
    ):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        self.file_exists = False

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Log metrics to CSV."""
        log_path = self.output_dir / self.filename
        row = {
            "epoch": trainer.epoch,
            "train_loss": trainer.history["train_loss"][-1] if trainer.history["train_loss"] else 0,
            "train_acc": trainer.history["train_acc"][-1] if trainer.history["train_acc"] else 0,
            "val_loss": trainer.history["val_loss"][-1] if trainer.history["val_loss"] else 0,
            "val_acc": trainer.history["val_acc"][-1] if trainer.history["val_acc"] else 0,
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            writer.writerow(row)


class ProgressLogger(Callback):
    """Progress logger callback."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        """Log progress."""
        if self.verbose:
            epoch = trainer.epoch
            train_loss = trainer.history["train_loss"][-1] if trainer.history["train_loss"] else 0
            train_acc = trainer.history["train_acc"][-1] if trainer.history["train_acc"] else 0
            val_loss = trainer.history["val_loss"][-1] if trainer.history["val_loss"] else 0
            val_acc = trainer.history["val_acc"][-1] if trainer.history["val_acc"] else 0

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )