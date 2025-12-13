"""Training script for biomass regression with grid search support.

Based on pj_kenpin's training framework, adapted for regression tasks.
"""

# isort: off
# config must be imported first to setup paths via rootutils
import config  # noqa: F401
# isort: on

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold
from timm.utils import ModelEmaV3
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data import (
    DualInputBiomassDataset,
    build_post_split_transform,
    build_pre_split_transform,
    build_pre_split_transform_valid,
    convert_long_to_wide,
)
from src.loss_function import build_loss_function
from src.metric import TARGET_COLS_PRED, weighted_r2_score_3targets
from src.model import build_model
from src.seed import seed_everything

# Keys that should NOT be swept even if they are lists
NON_SWEEP_PATHS: set[tuple[str, ...]] = {
    ("experiment", "notes"),
    ("dataset", "target_cols"),
    ("loss", "weights"),
    ("augmentation", "normalize", "mean"),
    ("augmentation", "normalize", "std"),
    ("augmentation", "train"),
}


@dataclass(frozen=True)
class SweepParam:
    """Parameter to sweep in grid search."""

    path: tuple[str, ...]
    values: Sequence[Any]


def load_yaml_with_includes(path: Path) -> dict[str, Any]:
    """Load YAML with __include__ support."""
    cfg = OmegaConf.load(str(path))
    data = OmegaConf.to_container(cfg, resolve=True) or {}

    includes = data.pop("__include__", [])
    result: dict[str, Any] = {}

    if isinstance(includes, str):
        includes = [includes]

    for inc in includes or []:
        include_path = (path.parent / inc).resolve()
        if not include_path.exists():
            raise FileNotFoundError(f"Include file not found: {include_path}")
        include_cfg = load_yaml_with_includes(include_path)
        result = merge_dicts(result, include_cfg)

    result = merge_dicts(result, data)
    return result


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def collect_sweep_params(cfg: dict[str, Any]) -> list[SweepParam]:
    """Collect parameters to sweep from config."""
    params: list[SweepParam] = []

    def _is_non_sweep_path(path: tuple[str, ...]) -> bool:
        if path in NON_SWEEP_PATHS:
            return True
        # Check prefix matches
        for non_sweep in NON_SWEEP_PATHS:
            if len(path) >= len(non_sweep) and path[: len(non_sweep)] == non_sweep:
                return True
        return False

    def _collect(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _collect(v, path + (k,))
        elif isinstance(node, list):
            if _is_non_sweep_path(path):
                return
            if not node:
                return
            # Check if it's a list of values (not nested structures)
            if any(isinstance(item, (dict, list)) for item in node):
                return
            params.append(SweepParam(path=path, values=node))

    _collect(cfg, tuple())
    return params


def apply_sweep_values(cfg: dict[str, Any], assignments: Sequence[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
    """Apply sweep values to config."""
    updated = deepcopy(cfg)
    for path, value in assignments:
        target = updated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return updated


def make_run_descriptor(assignments: Sequence[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
    """Create run descriptor from assignments."""
    return {"/".join(path): value for path, value in assignments}


def sanitize_name(name: str) -> str:
    """Sanitize string for use in filenames."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    group_col: str = "site",
) -> pd.DataFrame:
    """Create fold column using GroupKFold.

    GroupKFold ensures that samples from the same group (e.g., same site)
    are not split across train and validation sets. This provides a more
    realistic estimate of model performance on unseen sites.

    Args:
        df: DataFrame with group column
        n_folds: Number of folds
        group_col: Column name for grouping (default: "site")

    Returns:
        DataFrame with fold column added
    """
    df = df.copy()
    df["fold"] = -1

    gkf = GroupKFold(n_splits=n_folds)
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold

    return df


def plot_training_curves(
    all_histories: list[dict[str, list[float]]],
    output_dir: Path,
    folds: list[int] | None = None,
) -> None:
    """Plot and save training curves for all folds.

    Args:
        all_histories: List of history dicts, one per fold
        output_dir: Directory to save the plot
        folds: List of fold numbers (for labels)
    """
    ensure_dir(output_dir)

    if not all_histories:
        return

    # Colors for each fold
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    n_folds = len(all_histories)
    if folds is None:
        folds = list(range(n_folds))

    # Plot each fold
    for i, (history, fold) in enumerate(zip(all_histories, folds)):
        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors[i % len(colors)]

        # Train loss
        axes[0].plot(epochs, history["train_loss"], color=color, alpha=0.7, label=f"fold{fold}")

        # Val loss
        axes[1].plot(epochs, history["val_loss"], color=color, alpha=0.7, label=f"fold{fold}")

        # Val R²
        if "val_r2" in history:
            axes[2].plot(epochs, history["val_r2"], color=color, alpha=0.7, label=f"fold{fold}")

    # Calculate and plot mean curves
    if n_folds > 1:
        mean_train_loss = np.mean([h["train_loss"] for h in all_histories], axis=0)
        mean_val_loss = np.mean([h["val_loss"] for h in all_histories], axis=0)
        epochs = range(1, len(mean_train_loss) + 1)

        axes[0].plot(epochs, mean_train_loss, color="black", linewidth=2, linestyle="--", label="mean")
        axes[1].plot(epochs, mean_val_loss, color="black", linewidth=2, linestyle="--", label="mean")

        if "val_r2" in all_histories[0]:
            mean_val_r2 = np.mean([h["val_r2"] for h in all_histories], axis=0)
            axes[2].plot(epochs, mean_val_r2, color="black", linewidth=2, linestyle="--", label="mean")

    # Labels and titles
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Weighted R²")
    axes[2].set_title("Validation R²")
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=100)
    plt.close()


def configure_logger(log_path: Path) -> logging.Logger:
    """Configure logger for a run."""
    logger = logging.getLogger(f"train.{log_path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    ema: ModelEmaV3 | None,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    global_step: int,
    use_amp: bool,
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    train_loss_sum = 0.0
    train_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
    for batch in pbar:
        targets = batch["targets"].to(device)
        images_left = batch["image_left"].to(device)
        images_right = batch["image_right"].to(device)
        batch_size = images_left.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images_left, images_right)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        if ema is not None:
            ema.update(model, step=global_step)

        train_samples += batch_size
        train_loss_sum += float(loss.item()) * batch_size

        pbar.set_postfix({"loss": f"{train_loss_sum / train_samples:.4f}"})

    train_loss = train_loss_sum / max(train_samples, 1)
    return train_loss, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model."""
    model.eval()
    val_loss_sum = 0.0
    val_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False)
    for batch in pbar:
        targets = batch["targets"].to(device)
        images_left = batch["image_left"].to(device)
        images_right = batch["image_right"].to(device)
        batch_size = images_left.size(0)

        preds = model(images_left, images_right)
        loss = criterion(preds, targets)

        val_loss_sum += float(loss.item()) * batch_size
        val_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    val_loss = val_loss_sum / max(val_samples, 1)

    # Calculate R² score
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_r2 = weighted_r2_score_3targets(all_targets, all_preds)

    return val_loss, val_r2, all_preds, all_targets


def train_single_fold(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    fold: int,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Train a single fold."""
    # Extract config sections
    trainer_cfg = cfg.get("trainer", {})
    dataset_cfg = cfg.get("dataset", {})
    optimization_cfg = cfg.get("optimization", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    aug_cfg = cfg.get("augmentation", {})

    # Get parameters
    img_size = dataset_cfg.get("img_size", 224)
    if isinstance(img_size, list):
        img_size = img_size[0]
    in_chans = dataset_cfg.get("in_chans", 3)
    target_cols = dataset_cfg.get("target_cols", TARGET_COLS_PRED)

    # Split data
    train_fold_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    valid_fold_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    logger.info(f"Fold {fold}: Train={len(train_fold_df)}, Valid={len(valid_fold_df)}")

    # Build transforms and datasets
    normalize_cfg = aug_cfg.get("normalize", {})
    train_aug_cfg = aug_cfg.get("train", {})
    image_dir_cfg = dataset_cfg.get("image_dir")
    image_dir = Path(image_dir_cfg) if image_dir_cfg else config.get_image_dir()
    image_col = "image_path"

    # Pre-split + post-split transforms
    coarse_dropout_cfg = train_aug_cfg.get("coarse_dropout", {})
    gaussian_blur_cfg = train_aug_cfg.get("gaussian_blur", {})
    pre_split_transform = build_pre_split_transform(
        horizontal_flip_p=train_aug_cfg.get("horizontal_flip", {}).get("p", 0.5),
        vertical_flip_p=train_aug_cfg.get("vertical_flip", {}).get("p", 0.5),
        brightness_limit=train_aug_cfg.get("random_brightness_contrast", {}).get("brightness_limit", 0.2),
        contrast_limit=train_aug_cfg.get("random_brightness_contrast", {}).get("contrast_limit", 0.2),
        brightness_contrast_p=train_aug_cfg.get("random_brightness_contrast", {}).get("p", 0.25),
        coarse_dropout_p=coarse_dropout_cfg.get("p", 0.2),
        coarse_dropout_num_holes_range=tuple(coarse_dropout_cfg.get("num_holes_range", [10, 20])),
        coarse_dropout_hole_height_range=tuple(coarse_dropout_cfg.get("hole_height_range", [20, 100])),
        coarse_dropout_hole_width_range=tuple(coarse_dropout_cfg.get("hole_width_range", [20, 100])),
        gaussian_blur_p=gaussian_blur_cfg.get("p", 0.3),
        gaussian_blur_limit=tuple(gaussian_blur_cfg.get("blur_limit", [3, 7])),
    )
    post_split_transform = build_post_split_transform(
        img_size=img_size,
        normalize_mean=normalize_cfg.get("mean"),
        normalize_std=normalize_cfg.get("std"),
    )

    train_dataset = DualInputBiomassDataset(
        df=train_fold_df,
        image_dir=image_dir,
        pre_split_transform=pre_split_transform,
        post_split_transform=post_split_transform,
        target_cols=target_cols,
        image_col=image_col,
        is_train=True,
    )
    valid_dataset = DualInputBiomassDataset(
        df=valid_fold_df,
        image_dir=image_dir,
        pre_split_transform=build_pre_split_transform_valid(),  # No augmentation
        post_split_transform=post_split_transform,
        target_cols=target_cols,
        image_col=image_col,
        is_train=True,
    )

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_cfg.get("train_batch_size", 32),
        shuffle=True,
        num_workers=trainer_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=trainer_cfg.get("val_batch_size", 64),
        shuffle=False,
        num_workers=trainer_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Build model
    model = build_model(
        model_name=model_cfg.get("backbone", "tf_efficientnetv2_b0.in1k"),
        num_outputs=model_cfg.get("num_outputs", 3),
        pretrained=model_cfg.get("pretrained", True),
        in_chans=in_chans,
        dropout=model_cfg.get("dropout", 0.2),
        hidden_size=model_cfg.get("hidden_size", 512),
        device=device,
    )

    # Build loss function
    loss_weights = loss_cfg.get("weights", {})
    weights_list = [loss_weights.get(col, 1.0) for col in target_cols]
    criterion = build_loss_function(
        weights=weights_list,
        beta=loss_cfg.get("beta", 1.0),
    )

    # Optimizer and scheduler
    num_epochs = trainer_cfg.get("num_epochs", 30)
    total_steps = len(train_loader) * num_epochs
    lr = optimization_cfg.get("lr", 1e-4)
    weight_decay = optimization_cfg.get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_rate = optimization_cfg.get("warmup_rate", 0.1)
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP
    use_amp = trainer_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # EMA
    use_ema = optimization_cfg.get("use_ema", True)
    ema = None
    if use_ema:
        ema_decay = optimization_cfg.get("ema_decay", 0.995)
        ema_start_ratio = optimization_cfg.get("ema_start_ratio", 0.1)
        update_after_step = int(total_steps * ema_start_ratio)
        ema = ModelEmaV3(model, decay=ema_decay, update_after_step=update_after_step)
        logger.info(f"EMA enabled: decay={ema_decay}")

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    best_val_r2 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    global_step = 0

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            ema,
            device,
            epoch,
            num_epochs,
            global_step,
            use_amp,
        )

        # Validate (use EMA model if available)
        eval_model = ema.module if ema is not None else model
        val_loss, val_r2, _, _ = validate(
            eval_model,
            valid_loader,
            criterion,
            device,
            epoch,
            num_epochs,
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        logger.info(
            f"Epoch {epoch}/{num_epochs} - train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_r2: {val_r2:.5f}"
        )

        # Save best model (by val_loss to avoid spike selection)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_epoch = epoch
            save_model = ema.module if ema is not None else model
            torch.save(save_model.state_dict(), weights_dir / f"best_fold{fold}.pth")
            logger.info(f"Saved best model (val_loss={val_loss:.5f}, R²={val_r2:.5f})")

    # Save last model
    save_model = ema.module if ema is not None else model
    torch.save(save_model.state_dict(), weights_dir / f"last_fold{fold}.pth")

    return {
        "fold": fold,
        "best_val_r2": best_val_r2,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def train_single_run(
    cfg: dict[str, Any],
    run_dir: Path,
    run_descriptor: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train a single run (possibly multiple folds)."""
    # Setup directories
    ensure_dir(run_dir)
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    # Save config
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(run_dir / "config.yaml"))

    # Setup logger
    logger = configure_logger(run_dir / "logs" / "train.log")
    logger.info("=" * 50)
    logger.info("Run parameters:")
    logger.info(json.dumps(run_descriptor, ensure_ascii=False, indent=2))

    # Get config sections
    experiment_cfg = cfg.get("experiment", {})
    dataset_cfg = cfg.get("dataset", {})
    trainer_cfg = cfg.get("trainer", {})

    # Seed
    seed = experiment_cfg.get("seed", 42)
    seed_everything(seed)
    logger.info(f"Seed: {seed}")

    # Load and prepare data
    train_csv_cfg = dataset_cfg.get("train_csv")
    train_csv = Path(train_csv_cfg) if train_csv_cfg else config.get_train_csv_path()
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    logger.info(f"Loading data from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    # Create site column for GroupKFold (before Wide conversion)
    train_df["site"] = train_df["State"] + "_" + train_df["Sampling_Date"]

    # Convert Long to Wide format
    train_df = convert_long_to_wide(train_df)
    logger.info(f"Data loaded: {len(train_df)} images")

    # Create folds using GroupKFold
    n_folds = trainer_cfg.get("n_folds", 5)
    group_col = trainer_cfg.get("group_col", "site")
    train_df = create_folds(train_df, n_folds=n_folds, group_col=group_col)
    logger.info(f"Created {n_folds} folds using GroupKFold (group_col={group_col})")

    # Determine which folds to train
    fold_spec = trainer_cfg.get("fold", None)
    if fold_spec is None:
        folds_to_train = list(range(n_folds))
    else:
        folds_to_train = [fold_spec] if isinstance(fold_spec, int) else fold_spec

    logger.info(f"Training folds: {folds_to_train}")

    # Train each fold
    fold_results = []
    all_histories = []

    for fold in folds_to_train:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"FOLD {fold}")
        logger.info("=" * 50)

        result = train_single_fold(cfg, train_df, fold, run_dir, device, logger)
        fold_results.append(result)
        all_histories.append(result["history"])

        logger.info(f"Fold {fold} - Best val_loss: {result['best_val_loss']:.5f}, R²: {result['best_val_r2']:.5f} (epoch {result['best_epoch']})")

    # Calculate average metrics
    avg_r2 = np.mean([r["best_val_r2"] for r in fold_results])
    avg_loss = np.mean([r["best_val_loss"] for r in fold_results])

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Average val_loss: {avg_loss:.5f}")
    logger.info(f"Average R²: {avg_r2:.5f}")

    # Plot combined training curves (all folds + mean)
    if all_histories:
        plot_training_curves(all_histories, run_dir / "plots", folds=folds_to_train)

    # Save metrics
    metrics_path = run_dir / "logs" / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_val_loss", "best_val_r2", "best_epoch"])
        for r in fold_results:
            writer.writerow([r["fold"], r["best_val_loss"], r["best_val_r2"], r["best_epoch"]])

    # Return summary
    summary = {
        "avg_val_loss": avg_loss,
        "avg_val_r2": avg_r2,
        "fold_results": fold_results,
    }
    summary.update(run_descriptor)
    return summary


def create_run_name(index: int, descriptor: dict[str, Any]) -> str:
    """Create run name from descriptor."""
    backbone = descriptor.get("model/backbone")
    backbone_slug = sanitize_name(str(backbone)) if backbone else f"run{index:03d}"

    other_parts = [
        f"{sanitize_name(k.split('/')[-1])}-{sanitize_name(str(v))}"
        for k, v in descriptor.items()
        if k != "model/backbone"
    ]
    suffix = "__".join(other_parts) if other_parts else ""

    if suffix:
        return f"{index:03d}_{backbone_slug}__{suffix}"
    return f"{index:03d}_{backbone_slug}"


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train biomass regression models.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    # Load config (resolve relative to EXP_DIR, not cwd which may be changed by rootutils)
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = config.EXP_DIR / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_cfg = load_yaml_with_includes(config_path)
    sweep_params = collect_sweep_params(base_cfg)

    # Generate grid search combinations
    assignments_list: list[list[tuple[tuple[str, ...], Any]]] = []
    if sweep_params:
        path_list = [param.path for param in sweep_params]
        value_list = [param.values for param in sweep_params]
        for combo in product(*value_list):
            assignments_list.append(list(zip(path_list, combo, strict=False)))
    else:
        assignments_list.append([])

    # Create experiment directory
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    experiment_cfg = base_cfg.get("experiment", {})
    experiment_name = experiment_cfg.get("name", "exp")
    output_dir_cfg = experiment_cfg.get("output_dir")
    output_root = Path(output_dir_cfg).expanduser() if output_dir_cfg else config.OUTPUT_DIR
    experiment_dir = output_root / f"{timestamp}_{sanitize_name(str(experiment_name))}"
    ensure_dir(experiment_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Total runs: {len(assignments_list)}")

    # Run all combinations
    summaries: list[dict[str, Any]] = []

    for idx, assignments in enumerate(assignments_list, start=1):
        run_cfg = apply_sweep_values(base_cfg, assignments)
        descriptor = make_run_descriptor(assignments)
        run_name = create_run_name(idx, descriptor) if descriptor else f"{idx:03d}"
        run_dir = experiment_dir / run_name

        print(f"\n[Run {idx}/{len(assignments_list)}] {run_name}")

        summary = train_single_run(run_cfg, run_dir, descriptor, device)
        summary["run_number"] = idx
        summary["run_name"] = run_name
        summaries.append(summary)

    # Find best run (by avg_val_loss)
    best_summary = min(summaries, key=lambda x: x.get("avg_val_loss", float("inf")))

    # Save summary
    summary_csv = experiment_dir / "summary.csv"
    if summaries:
        summaries_sorted = sorted(summaries, key=lambda x: x.get("avg_val_loss", float("inf")))

        fieldnames = ["avg_val_loss", "avg_val_r2", "run_name", "run_number"]
        other_keys = sorted({k for s in summaries_sorted for k in s if k not in fieldnames + ["fold_results"]})
        fieldnames.extend(other_keys)

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in summaries_sorted:
                writer.writerow(row)

        summary_json = {
            "runs": [{k: v for k, v in s.items() if k != "fold_results"} for s in summaries],
            "best_run": {
                "run_name": best_summary.get("run_name"),
                "avg_val_loss": best_summary.get("avg_val_loss"),
                "avg_val_r2": best_summary.get("avg_val_r2"),
            },
        }
        with (experiment_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print(f"Experiment outputs: {experiment_dir}")
    print(f"Summary: {summary_csv}")
    print(f"Best run: {best_summary.get('run_name')} (val_loss={best_summary.get('avg_val_loss'):.5f}, R²={best_summary.get('avg_val_r2'):.5f})")


if __name__ == "__main__":
    main()
