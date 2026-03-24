"""Training loop implementation."""
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..utils.helpers import get_device, save_checkpoint


class Trainer:
    """Trainer class for image classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        output_dir: Union[str, Path] = "./outputs",
        callbacks: Optional[List[Any]] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: Optional[float] = None,
        log_interval: int = 10,
        save_best_only: bool = True,
        **kwargs,
    ):
        self.model = model.to(device or get_device())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.AdamW(model.parameters())
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or []
        self.use_amp = use_amp and self.device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.log_interval = log_interval
        self.save_best_only = save_best_only

        self.scaler = GradScaler() if self.use_amp else None
        self.epoch = 0
        self.best_metric = 0.0
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for num_epochs."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()

            train_loss, train_acc = self._train_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if self.val_loader:
                val_loss, val_acc = self._validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                metric = val_acc
                self.best_metric = max(self.best_metric, metric)
            else:
                metric = train_acc

            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                elif hasattr(self.scheduler, 'step'):
                    self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(self, metric=metric)

            if not self.save_best_only or metric >= self.best_metric:
                self.save_checkpoint("last.pth")

        return self.history

    def _train_epoch(self) -> tuple:
        """Train for one epoch."""
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def _validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def save_checkpoint(self, filename: str = "checkpoint.pth") -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.epoch,
            {"best_metric": self.best_metric},
            self.output_dir / filename,
        )

    def load_checkpoint(self, filename: str = "checkpoint.pth") -> None:
        """Load model checkpoint."""
        checkpoint_path = self.output_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.history = checkpoint.get("history", self.history)

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])