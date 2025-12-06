"""Training utilities for image classification."""

import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Get optimizer with weight decay.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Optimizer instance
    """
    # Don't apply weight decay to bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    steps_per_epoch: int = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler (cosine, step, plateau)
        epochs: Number of epochs
        steps_per_epoch: Number of steps per epoch (for step scheduler)

    Returns:
        Scheduler instance
    """
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool = True,
    scaler: GradScaler = None,
) -> tuple[float, float]:
    """Train model for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{100.*correct/total:.2f}%"})

    accuracy = 100.0 * correct / total
    return losses.avg, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
) -> tuple[float, float]:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validation")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{100.*correct/total:.2f}%"})

    accuracy = 100.0 * correct / total
    return losses.avg, accuracy


def train_fold(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    scheduler_type: str,
    device: str,
    use_amp: bool,
    save_path: Path,
    fold: int,
) -> dict:
    """Train model for one fold.

    Args:
        model: Model to train
        train_loader: Training dataloader
        valid_loader: Validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        scheduler_type: Type of scheduler
        device: Device to use
        use_amp: Whether to use AMP
        save_path: Path to save model
        fold: Fold number

    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr, weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_type, epochs)
    scaler = GradScaler() if use_amp else None

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )

        # Validate
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device, use_amp)

        # Update scheduler
        if scheduler_type == "plateau":
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")

        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / f"model_fold{fold}.pth")
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

    return history
