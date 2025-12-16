"""Recalculate CV scores using 5 targets instead of 3.

This script recalculates the CV R² scores for exp002-005 using the full 5-target
metric instead of the 3-target metric that was mistakenly used during training.

The 5 targets are:
- Dry_Total_g (predicted, weight=0.5)
- GDM_g (predicted, weight=0.2)
- Dry_Green_g (predicted, weight=0.1)
- Dry_Dead_g (derived: Dry_Total_g - GDM_g, weight=0.1)
- Dry_Clover_g (derived: GDM_g - Dry_Green_g, weight=0.1)

Usage:
    python src/recalculate_cv_5targets.py
    python src/recalculate_cv_5targets.py --exp 002 003  # specific experiments only
"""

import argparse
import csv
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

# ============================================================================
# Constants
# ============================================================================
TARGET_COLS_PRED = ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
TARGET_COLS_ALL = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = {
    "Dry_Total_g": 0.5,
    "GDM_g": 0.2,
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
}

DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
COMP_DATASET_DIR = INPUT_DIR / "csiro-biomass"

# Experiment output directories
EXP_OUTPUT_DIRS = {
    "002": OUTPUT_DIR / "002" / "1" / "20251213_001219_exp002",
    "003": OUTPUT_DIR / "003" / "1" / "20251213_223443_exp003",
    "004": OUTPUT_DIR / "004" / "1" / "20251215_000838_exp004",
    "005": OUTPUT_DIR / "005" / "1" / "20251216_005641_exp005",
}


# ============================================================================
# Metric Functions (from experiments/*/src/metric.py)
# ============================================================================
def derive_all_targets(preds_3: np.ndarray) -> np.ndarray:
    """Derive all 5 target values from 3 predicted values.

    Args:
        preds_3: Array of shape [N, 3] with columns [Dry_Total_g, GDM_g, Dry_Green_g]

    Returns:
        Array of shape [N, 5] with columns [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    Dry_Total_g = preds_3[:, 0]
    GDM_g = preds_3[:, 1]
    Dry_Green_g = preds_3[:, 2]

    # Derive remaining targets
    Dry_Dead_g = np.maximum(Dry_Total_g - GDM_g, 0)
    Dry_Clover_g = np.maximum(GDM_g - Dry_Green_g, 0)

    # Return in TARGET_COLS_ALL order
    return np.stack([Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g], axis=1)


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str] | None = None,
) -> float:
    """Calculate globally weighted R² score (competition metric)."""
    if target_cols is None:
        target_cols = TARGET_COLS_ALL

    weights = np.array([TARGET_WEIGHTS[col] for col in target_cols])
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    n_samples = y_true.shape[0]
    w_flat = np.tile(weights, n_samples)

    y_bar_w = np.average(y_true_flat, weights=w_flat)
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_bar_w) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def weighted_r2_score_full(y_true_3: np.ndarray, y_pred_3: np.ndarray) -> float:
    """Calculate full weighted R² by deriving all 5 targets."""
    y_true_5 = derive_all_targets(y_true_3)
    y_pred_5 = derive_all_targets(y_pred_3)
    return weighted_r2_score(y_true_5, y_pred_5, target_cols=TARGET_COLS_ALL)


# ============================================================================
# Data Functions (from experiments/*/src/data.py)
# ============================================================================
def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Long format CSV to Wide format (1 row per image)."""
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    if "target" not in df.columns:
        df_wide = df[["image_id", "image_path"]].drop_duplicates().reset_index(drop=True)
        return df_wide

    meta_cols = [c for c in df.columns if c not in ["sample_id", "target_name", "target", "image_id"]]

    df_wide = df.pivot_table(
        index=["image_id"] + meta_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    df_wide.columns.name = None
    return df_wide


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    group_col: str = "site",
) -> pd.DataFrame:
    """Create fold column using GroupKFold."""
    df = df.copy()
    df["fold"] = -1

    gkf = GroupKFold(n_splits=n_folds)
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold

    return df


class DualInputBiomassDataset(Dataset):
    """Dataset for dual-input (left-right split) biomass prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path | str,
        post_split_transform: A.Compose | None = None,
        target_cols: list[str] = TARGET_COLS_PRED,
        image_col: str = "image_path",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.post_split_transform = post_split_transform
        self.target_cols = target_cols
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_path = self.image_dir / row[self.image_col]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split into left and right halves (no augmentation for validation)
        w = image.shape[1]
        mid = w // 2
        image_left = image[:, :mid, :]
        image_right = image[:, mid:, :]

        if self.post_split_transform is not None:
            image_left = self.post_split_transform(image=image_left)["image"]
            image_right = self.post_split_transform(image=image_right)["image"]

        result = {
            "image_left": image_left,
            "image_right": image_right,
            "image_path": str(row[self.image_col]),
        }

        if "image_id" in row:
            result["image_id"] = row["image_id"]

        if all(col in row for col in self.target_cols):
            targets = [row[col] for col in self.target_cols]
            result["targets"] = torch.tensor(targets, dtype=torch.float32)

        return result


def build_post_split_transform(
    img_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> A.Compose:
    """Build post-split transform for validation."""
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    transforms = [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


# ============================================================================
# Model Functions (from experiments/*/src/model.py)
# ============================================================================
try:
    import timm
except ImportError:
    print("timm not installed. Please install it with: pip install timm")
    sys.exit(1)


class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, p_trainable: bool = True):
        super().__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x


def _ensure_bchw(x: torch.Tensor, num_features: int | None) -> torch.Tensor:
    """Ensure tensor is in BCHW format."""
    if x.dim() == 4 and (num_features is not None) and x.shape[-1] == num_features:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


class DualInputRegressionNet(nn.Module):
    """Dual input regression model for left-right split images."""

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_b0.in1k",
        num_outputs: int = 3,
        pretrained: bool = True,
        in_chans: int = 3,
        dropout: float = 0.1,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        self.num_features = self.backbone.num_features

        self.global_pool = GeM(p_trainable=True)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.backbone.forward_features(x)
        features = _ensure_bchw(features, self.num_features)
        pooled = self.global_pool(features).view(batch_size, -1)
        return pooled

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        feat_left = self._extract_features(x_left)
        feat_right = self._extract_features(x_right)
        feat_concat = torch.cat([feat_left, feat_right], dim=1)
        output = self.head(feat_concat)
        return output


def load_model(
    model_path: str | Path,
    model_name: str,
    num_outputs: int = 3,
    in_chans: int = 3,
    dropout: float = 0.2,
    hidden_size: int = 512,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Load trained model from checkpoint."""
    model = DualInputRegressionNet(
        model_name=model_name,
        num_outputs=num_outputs,
        pretrained=False,
        in_chans=in_chans,
        dropout=dropout,
        hidden_size=hidden_size,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ============================================================================
# Inference Function
# ============================================================================
@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Inference", leave=False):
        images_left = batch["image_left"].to(device)
        images_right = batch["image_right"].to(device)
        targets = batch["targets"]

        preds = model(images_left, images_right)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


# ============================================================================
# Main Processing
# ============================================================================
def process_single_run(
    run_dir: Path,
    train_df: pd.DataFrame,
    image_dir: Path,
    device: torch.device,
    n_folds: int = 5,
) -> dict[int, float]:
    """Process a single run and return 5-target R² for each fold."""
    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"  Warning: config.yaml not found in {run_dir}")
        return {}

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract model config
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentation", {})

    model_name = model_cfg.get("backbone", "tf_efficientnetv2_b0.in1k")
    num_outputs = model_cfg.get("num_outputs", 3)
    in_chans = dataset_cfg.get("in_chans", 3)
    img_size = dataset_cfg.get("img_size", 224)
    dropout = model_cfg.get("dropout", 0.1)
    hidden_size = model_cfg.get("hidden_size", 512)

    normalize_cfg = aug_cfg.get("normalize", {})
    normalize_mean = normalize_cfg.get("mean")
    normalize_std = normalize_cfg.get("std")

    # Build transform
    post_split_transform = build_post_split_transform(
        img_size=img_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    results = {}

    for fold in range(n_folds):
        weight_path = run_dir / "weights" / f"best_fold{fold}.pth"
        if not weight_path.exists():
            print(f"  Warning: {weight_path} not found, skipping fold {fold}")
            continue

        # Load model
        model = load_model(
            model_path=weight_path,
            model_name=model_name,
            num_outputs=num_outputs,
            in_chans=in_chans,
            dropout=dropout,
            hidden_size=hidden_size,
            device=device,
        )

        # Create validation dataset for this fold
        valid_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        valid_dataset = DualInputBiomassDataset(
            df=valid_df,
            image_dir=image_dir,
            post_split_transform=post_split_transform,
            target_cols=TARGET_COLS_PRED,
            image_col="image_path",
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Run inference
        preds, targets = run_inference(model, valid_loader, device)

        # Calculate 5-target R²
        r2_5target = weighted_r2_score_full(targets, preds)
        results[fold] = r2_5target

        # Clean up
        del model
        torch.cuda.empty_cache()

    return results


def process_experiment(
    exp_name: str,
    exp_dir: Path,
    train_df: pd.DataFrame,
    image_dir: Path,
    device: torch.device,
) -> None:
    """Process all runs in an experiment."""
    print(f"\n{'='*60}")
    print(f"Processing experiment: {exp_name}")
    print(f"Directory: {exp_dir}")
    print(f"{'='*60}")

    if not exp_dir.exists():
        print(f"  Error: Directory not found: {exp_dir}")
        return

    # Find all run directories
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()])

    if not run_dirs:
        print(f"  No runs found in {exp_dir}")
        return

    print(f"  Found {len(run_dirs)} runs")

    # Store results for summary_fix.csv
    run_results: dict[str, float] = {}

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n  Processing run: {run_name}")

        # Check if metrics.csv exists
        metrics_path = run_dir / "logs" / "metrics.csv"
        if not metrics_path.exists():
            print(f"    Warning: metrics.csv not found, skipping")
            continue

        # Read existing metrics
        existing_metrics = pd.read_csv(metrics_path)

        # Process the run
        r2_results = process_single_run(
            run_dir=run_dir,
            train_df=train_df,
            image_dir=image_dir,
            device=device,
            n_folds=5,
        )

        if not r2_results:
            print(f"    Warning: No results obtained, skipping")
            continue

        # Add 5-target R² column
        existing_metrics["best_val_r2_5target"] = existing_metrics["fold"].map(r2_results)

        # Reorder columns
        cols = ["fold", "best_val_loss", "best_val_r2", "best_val_r2_5target", "best_epoch"]
        existing_cols = [c for c in cols if c in existing_metrics.columns]
        existing_metrics = existing_metrics[existing_cols]

        # Save metrics_fix.csv
        output_path = run_dir / "logs" / "metrics_fix.csv"
        existing_metrics.to_csv(output_path, index=False)
        print(f"    Saved: {output_path}")

        # Print summary
        avg_r2_3 = existing_metrics["best_val_r2"].mean()
        avg_r2_5 = existing_metrics["best_val_r2_5target"].mean()
        print(f"    Avg R² (3-target): {avg_r2_3:.5f}")
        print(f"    Avg R² (5-target): {avg_r2_5:.5f}")

        # Store for summary_fix.csv
        run_results[run_name] = avg_r2_5

    # Create summary_fix.csv
    summary_path = exp_dir / "summary.csv"
    if summary_path.exists() and run_results:
        summary_df = pd.read_csv(summary_path)

        # Add avg_val_r2_5target column
        summary_df["avg_val_r2_5target"] = summary_df["run_name"].map(run_results)

        # Reorder columns to place avg_val_r2_5target after avg_val_r2
        cols = list(summary_df.columns)
        if "avg_val_r2" in cols and "avg_val_r2_5target" in cols:
            idx = cols.index("avg_val_r2") + 1
            cols.remove("avg_val_r2_5target")
            cols.insert(idx, "avg_val_r2_5target")
            summary_df = summary_df[cols]

        # Sort by avg_val_r2_5target (descending, best first)
        summary_df = summary_df.sort_values("avg_val_r2_5target", ascending=False)

        # Save summary_fix.csv
        summary_fix_path = exp_dir / "summary_fix.csv"
        summary_df.to_csv(summary_fix_path, index=False)
        print(f"\n  Saved summary: {summary_fix_path}")


def main():
    parser = argparse.ArgumentParser(description="Recalculate CV scores using 5 targets")
    parser.add_argument(
        "--exp",
        nargs="+",
        default=["002", "003", "004", "005"],
        help="Experiment names to process (default: 002 003 004 005)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load and prepare training data
    train_csv = COMP_DATASET_DIR / "train.csv"
    if not train_csv.exists():
        print(f"Error: train.csv not found at {train_csv}")
        sys.exit(1)

    print(f"Loading training data from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    # Create site column for GroupKFold
    train_df["site"] = train_df["State"] + "_" + train_df["Sampling_Date"]

    # Convert to Wide format
    train_df = convert_long_to_wide(train_df)
    train_df = train_df.sort_values(by=["image_id"]).reset_index(drop=True)
    print(f"Loaded {len(train_df)} images")

    # Create folds using GroupKFold
    train_df = create_folds(train_df, n_folds=5, group_col="site")
    print("Created 5 folds using GroupKFold")

    # Process each experiment
    for exp_name in args.exp:
        if exp_name not in EXP_OUTPUT_DIRS:
            print(f"Warning: Unknown experiment {exp_name}, skipping")
            continue

        exp_dir = EXP_OUTPUT_DIRS[exp_name]
        process_experiment(
            exp_name=exp_name,
            exp_dir=exp_dir,
            train_df=train_df,
            image_dir=COMP_DATASET_DIR,
            device=device,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
