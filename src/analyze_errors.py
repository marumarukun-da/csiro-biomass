"""Error analysis script for biomass prediction models (exp008 only).

This script generates PDF reports comparing predictions vs ground truth
for each fold's validation data.

Usage:
    python src/analyze_errors.py --run_dir data/output/008/1/.../001_convnextv2_...
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold
from torch import nn
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================
TARGET_COLS_ALL = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = {
    "Dry_Total_g": 0.5,
    "GDM_g": 0.2,
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
}
TARGET_COLS_PRED = ["Dry_Dead_g", "Dry_Green_g", "Dry_Clover_g"]

METADATA_COLS = ["State", "Species", "Sampling_Date", "Pre_GSHH_NDVI", "Height_Ave_cm"]


# =============================================================================
# Evaluation Metrics
# =============================================================================
def derive_all_targets(preds_3: np.ndarray) -> np.ndarray:
    """Derive all 5 target values from 3 predicted values."""
    Dry_Dead_g = preds_3[:, 0]
    Dry_Green_g = preds_3[:, 1]
    Dry_Clover_g = preds_3[:, 2]

    GDM_g = Dry_Green_g + Dry_Clover_g
    Dry_Total_g = Dry_Dead_g + GDM_g

    return np.stack([Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g], axis=1)


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str] | None = None,
) -> float:
    """Calculate globally weighted R2 score (competition metric)."""
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
    """Calculate full weighted R2 by deriving all 5 targets."""
    y_true_5 = derive_all_targets(y_true_3)
    y_pred_5 = derive_all_targets(y_pred_3)
    return weighted_r2_score(y_true_5, y_pred_5, target_cols=TARGET_COLS_ALL)


# =============================================================================
# Data Processing
# =============================================================================
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


# =============================================================================
# Model Definition (exp008)
# =============================================================================
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


class DualInputRegressionNet(nn.Module):
    """Dual-input regression network for biomass prediction (exp008)."""

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

        # Shared backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        self.num_features = self.backbone.num_features

        # Shared pooling
        self.global_pool = GeM(p_trainable=True)

        # Regression head
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
        # Handle BHWC format (e.g., Swin)
        if features.dim() == 4 and features.shape[-1] == self.num_features:
            features = features.permute(0, 3, 1, 2).contiguous()
        pooled = self.global_pool(features).view(batch_size, -1)
        return pooled

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        feat_left = self._extract_features(x_left)
        feat_right = self._extract_features(x_right)
        feat_concat = torch.cat([feat_left, feat_right], dim=1)
        output = self.head(feat_concat)
        output = F.softplus(output)
        return output


def load_model(
    model_path: str,
    model_name: str = "tf_efficientnetv2_b0.in1k",
    num_outputs: int = 3,
    in_chans: int = 3,
    dropout: float = 0.1,
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
    model = model.to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# =============================================================================
# Data Loading and Transforms
# =============================================================================
def build_val_transform(
    img_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> Compose:
    """Build validation transform."""
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    return Compose(
        [
            Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def load_and_preprocess_image(
    image_path: Path,
    transform: Compose,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load image and preprocess for model input."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image = image.copy()

    w = image.shape[1]
    mid = w // 2
    image_left = image[:, :mid, :]
    image_right = image[:, mid:, :]

    image_left = transform(image=image_left)["image"]
    image_right = transform(image=image_right)["image"]

    return image_left, image_right, original_image


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def predict_fold(
    model: nn.Module,
    df: pd.DataFrame,
    image_dir: Path,
    transform: Compose,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run inference on validation data."""
    model.eval()
    all_preds = []
    all_images = []

    for idx in tqdm(range(len(df)), desc="Predicting", leave=False):
        row = df.iloc[idx]
        image_path = image_dir / row["image_path"]

        image_left, image_right, original_image = load_and_preprocess_image(image_path, transform)

        image_left = image_left.unsqueeze(0).to(device)
        image_right = image_right.unsqueeze(0).to(device)

        # exp008: forward returns tensor directly, not dict
        pred = model(image_left, image_right).cpu().numpy()[0]

        all_preds.append(pred)
        all_images.append(original_image)

    return np.array(all_preds), all_images


# =============================================================================
# PDF Generation
# =============================================================================
def create_sample_figure(
    ax: plt.Axes,
    image: np.ndarray,
    image_id: str,
    pred: np.ndarray,
    target: np.ndarray,
    metadata: dict,
) -> None:
    """Create visualization for a single sample."""
    ax.imshow(image)
    ax.axis("off")

    pred_5 = derive_all_targets(pred.reshape(1, -1))[0]
    target_5 = derive_all_targets(target.reshape(1, -1))[0]

    text_lines = [f"ID: {image_id}"]
    text_lines.append("")

    text_lines.append("[Metadata]")
    for col in METADATA_COLS:
        if col in metadata:
            text_lines.append(f"  {col}: {metadata[col]}")
    text_lines.append("")

    text_lines.append("[Predictions vs Ground Truth]")
    for i, col in enumerate(TARGET_COLS_ALL):
        error = pred_5[i] - target_5[i]
        text_lines.append(f"  {col}: {pred_5[i]:.2f} vs {target_5[i]:.2f} (err: {error:+.2f})")

    text = "\n".join(text_lines)
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def generate_pdf(
    df: pd.DataFrame,
    predictions: np.ndarray,
    images: list[np.ndarray],
    output_path: Path,
    fold: int,
    r2_score: float,
) -> None:
    """Generate PDF report for a fold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(df)
    n_pages = (n_samples + 1) // 2

    with PdfPages(output_path) as pdf:
        for page_idx in tqdm(range(n_pages), desc=f"Generating PDF fold {fold}", leave=False):
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f"Fold {fold} - Weighted R2: {r2_score:.5f} (Page {page_idx + 1}/{n_pages})", fontsize=12)

            for row_idx in range(2):
                sample_idx = page_idx * 2 + row_idx
                ax = axes[row_idx]

                if sample_idx >= n_samples:
                    ax.axis("off")
                    continue

                row = df.iloc[sample_idx]
                image = images[sample_idx]
                pred = predictions[sample_idx]
                target = np.array([row[col] for col in TARGET_COLS_PRED])

                metadata = {col: row[col] for col in METADATA_COLS if col in row}

                create_sample_figure(
                    ax=ax,
                    image=image,
                    image_id=row["image_id"],
                    pred=pred,
                    target=target,
                    metadata=metadata,
                )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, right=0.65)
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis for biomass prediction models (exp008).")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory (contains config.yaml and weights/)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (default: auto-detect)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    print(f"Loaded config from: {config_path}")

    # Detect data paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        project_root = run_dir
        while project_root.name != "data" and project_root != project_root.parent:
            project_root = project_root.parent
        project_root = project_root.parent
        data_dir = project_root / "data"

    input_dir = data_dir / "input" / "csiro-biomass"
    train_csv_path = input_dir / "train.csv"
    image_dir = input_dir

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

    model_name = run_dir.name
    output_dir = data_dir / "output" / "pdf" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    print(f"Loading data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    train_df["site"] = train_df["State"] + "_" + train_df["Sampling_Date"]
    train_df = convert_long_to_wide(train_df)
    train_df = train_df.sort_values(by=["image_id"]).reset_index(drop=True)
    print(f"Loaded {len(train_df)} images")

    # Create folds
    n_folds = cfg.trainer.get("n_folds", 5)
    group_col = cfg.trainer.get("group_col", "site")
    train_df = create_folds(train_df, n_folds=n_folds, group_col=group_col)
    print(f"Created {n_folds} folds using GroupKFold (group_col={group_col})")

    # Load metrics.csv for comparison
    metrics_csv_path = run_dir / "logs" / "metrics.csv"
    recorded_metrics = {}
    if metrics_csv_path.exists():
        metrics_df = pd.read_csv(metrics_csv_path)
        for _, row in metrics_df.iterrows():
            recorded_metrics[int(row["fold"])] = row["best_val_r2"]
        print(f"Loaded recorded metrics from: {metrics_csv_path}")

    # Build transform
    img_size = cfg.dataset.get("img_size", 224)
    if isinstance(img_size, list):
        img_size = img_size[0]
    normalize_cfg = cfg.augmentation.get("normalize", {})
    transform = build_val_transform(
        img_size=img_size,
        normalize_mean=normalize_cfg.get("mean"),
        normalize_std=normalize_cfg.get("std"),
    )

    # Get model config
    model_cfg = cfg.get("model", {})

    # Process each fold
    weights_dir = run_dir / "weights"
    print(f"\nProcessing {n_folds} folds...")

    for fold in range(n_folds):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold}")
        print("=" * 50)

        weight_path = weights_dir / f"best_fold{fold}.pth"
        if not weight_path.exists():
            print(f"  Weights not found: {weight_path}, skipping...")
            continue

        val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        print(f"  Validation samples: {len(val_df)}")

        model = load_model(
            model_path=str(weight_path),
            model_name=model_cfg.get("backbone", "tf_efficientnetv2_b0.in1k"),
            num_outputs=model_cfg.get("num_outputs", 3),
            in_chans=cfg.dataset.get("in_chans", 3),
            dropout=model_cfg.get("dropout", 0.1),
            hidden_size=model_cfg.get("hidden_size", 512),
            device=device,
        )
        print(f"  Loaded model from: {weight_path}")

        predictions, images = predict_fold(
            model=model,
            df=val_df,
            image_dir=image_dir,
            transform=transform,
            device=device,
        )

        targets = val_df[TARGET_COLS_PRED].values
        r2_score = weighted_r2_score_full(targets, predictions)
        print(f"  Calculated R2: {r2_score:.5f}")

        if fold in recorded_metrics:
            recorded_r2 = recorded_metrics[fold]
            diff = abs(r2_score - recorded_r2)
            status = "OK" if diff < 0.0001 else "MISMATCH!"
            print(f"  Recorded R2:   {recorded_r2:.5f} ({status}, diff={diff:.6f})")

        pdf_path = output_dir / f"{fold}.pdf"
        generate_pdf(
            df=val_df,
            predictions=predictions,
            images=images,
            output_path=pdf_path,
            fold=fold,
            r2_score=r2_score,
        )

        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 50}")
    print(f"Done! PDFs saved to: {output_dir}")


if __name__ == "__main__":
    main()
