"""Learning rate schedulers."""
from typing import Any, Optional, Union

import torch
from torch import optim


def create_scheduler(
    scheduler_type: str,
    optimizer: optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: Optional[int] = None,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs,
) -> Any:
    """Create a learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler ('cosine', 'step', 'multistep', 'exponential', 'plateau')
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        steps_per_epoch: Number of training steps per epoch (for OneCycleLR)
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        **kwargs: Additional arguments

    Returns:
        Scheduler instance
    """
    warmup_steps = warmup_epochs * (steps_per_epoch or 100)

    if scheduler_type == "cosine":
        if steps_per_epoch:
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                total_steps=num_epochs * steps_per_epoch,
                pct_start=warmup_steps / (num_epochs * steps_per_epoch),
                anneal_strategy="cos",
                final_div_factor=optimizer.param_groups[0]["lr"] / min_lr,
            )
        else:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=min_lr,
            )

    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif scheduler_type == "multistep":
        milestones = kwargs.get("milestones", [30, 60, 90])
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    elif scheduler_type == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    elif scheduler_type == "plateau":
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 10)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupScheduler:
    """Wrapper scheduler with linear warmup."""

    def __init__(
        self,
        scheduler: Any,
        warmup_steps: int,
        min_lr: float = 0.0,
    ):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0

    def step(self) -> None:
        """Step the scheduler."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            lr = self._get_warmup_lr()
            for param_group in self.scheduler.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            self.scheduler.step()

    def _get_warmup_lr(self) -> float:
        """Get learning rate for current warmup step."""
        base_lr = self.scheduler.optimizer.param_groups[0]["lr"]
        return self.min_lr + (base_lr - self.min_lr) * (self.step_count / self.warmup_steps)

    @property
    def state_dict(self) -> dict:
        """Return state dict."""
        return {
            "scheduler": self.scheduler.state_dict(),
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.warmup_steps = state_dict["warmup_steps"]
        self.step_count = state_dict["step_count"]