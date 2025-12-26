"""Inference script for DINOv3 head model with backbone feature extraction.

Supports:
- Multi-fold head model ensemble
- Test Time Augmentation (TTA) with 4 flip variants
- On-the-fly DINOv3 backbone feature extraction
- Automatic submission creation
"""

# isort: off
# config must be imported first to setup paths via rootutils
import config  # noqa: F401
# isort: on

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.backbone import build_backbone
from src.head_model import load_head_model
from src.metric import derive_all_targets
from src.seed import seed_everything


def build_tta_transforms() -> list[tuple[str, A.Compose]]:
    """Build TTA transforms (flip variants).

    Returns:
        List of (name, transform) tuples for TTA.
    """
    return [
        ("original", A.Compose([])),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)])),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)])),
        ("hvflip", A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])),
    ]


def preprocess_image(
    image: np.ndarray,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Preprocess image for DINOv3 backbone.

    Args:
        image: RGB image [H, W, C] uint8
        mean: ImageNet mean
        std: ImageNet std

    Returns:
        Normalized tensor [1, C, H, W]
    """
    img_normalized = (image / 255.0 - np.array(mean)) / np.array(std)
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def load_head_models(
    run_dir: Path,
    folds: list[int],
    device: torch.device,
    weight_type: str = "best",
) -> tuple[list[torch.nn.Module], dict]:
    """Load head models from all folds.

    Args:
        run_dir: Directory containing trained models
        folds: List of fold numbers to load
        device: Device to load models to
        weight_type: Type of weights to load ("best" or "last")

    Returns:
        Tuple of (list of loaded head models, head config dict)
    """
    if weight_type not in ("best", "last"):
        raise ValueError(f"weight_type must be 'best' or 'last', got '{weight_type}'")

    models = []
    head_config = None

    # Load config
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(config_path))
        cfg = OmegaConf.to_container(cfg, resolve=True)
        head_cfg = cfg.get("head", {})
        backbone_cfg = cfg.get("backbone", {})
    else:
        head_cfg = {}
        backbone_cfg = {}

    head_config = {
        "hidden_dim": backbone_cfg.get("hidden_dim", 1280),
        "num_patches": backbone_cfg.get("num_patches", 3600),
        "shared_hidden": head_cfg.get("shared_hidden", 512),
        "dropout": head_cfg.get("dropout", 0.1),
    }

    for fold in folds:
        weights_dir = run_dir / "weights"

        if weight_type == "best":
            primary_path = weights_dir / f"best_fold{fold}.pth"
            fallback_path = weights_dir / f"last_fold{fold}.pth"
        else:
            primary_path = weights_dir / f"last_fold{fold}.pth"
            fallback_path = weights_dir / f"best_fold{fold}.pth"

        weight_path = primary_path
        if not weight_path.exists():
            if fallback_path.exists():
                weight_path = fallback_path
                print(f"Warning: {primary_path.name} not found, using {fallback_path.name}")
            else:
                raise FileNotFoundError(f"Cannot find weights for fold {fold} in {weights_dir}")

        model = load_head_model(
            model_path=str(weight_path),
            hidden_dim=head_config["hidden_dim"],
            num_patches=head_config["num_patches"],
            shared_hidden=head_config["shared_hidden"],
            dropout=head_config["dropout"],
            device=str(device),
        )
        models.append(model)
        print(f"Loaded head model from {weight_path}")

    return models, head_config


@torch.inference_mode()
def predict_single_image(
    image: np.ndarray,
    backbone: torch.nn.Module,
    head_models: list[torch.nn.Module],
    tta_transforms: list[tuple[str, A.Compose]],
    device: torch.device,
    img_size: int = 960,
) -> np.ndarray:
    """Predict for a single image using DINOv3 backbone and head models.

    Args:
        image: Original image [H, W, C] in RGB
        backbone: DINOv3 backbone for feature extraction
        head_models: List of head models for ensemble
        tta_transforms: List of (name, transform) for TTA
        device: Device for inference
        img_size: Target image size for DINOv3

    Returns:
        Averaged predictions [num_outputs] (Dry_Total_g, GDM_g, Dry_Green_g)
    """
    all_preds = []

    # Resize to target size
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    for _, tta_transform in tta_transforms:
        # Apply TTA transform
        aug_image = tta_transform(image=image_resized)["image"]

        # Preprocess and move to device
        img_tensor = preprocess_image(aug_image).to(device)

        # Extract features with backbone
        cls_token, patch_tokens = backbone(img_tensor)

        # Predict with each head model
        for head_model in head_models:
            main_pred, _, _, _ = head_model(cls_token, patch_tokens)
            all_preds.append(main_pred.cpu().numpy()[0])

    # Average all predictions
    return np.mean(all_preds, axis=0)


def run_inference(
    test_df: pd.DataFrame,
    image_dir: Path,
    backbone: torch.nn.Module,
    head_models: list[torch.nn.Module],
    tta_transforms: list[tuple[str, A.Compose]],
    device: torch.device,
    img_size: int = 960,
    image_col: str = "image_path",
) -> dict[str, np.ndarray]:
    """Run inference on test set.

    Args:
        test_df: Test DataFrame
        image_dir: Directory containing images
        backbone: DINOv3 backbone
        head_models: List of head models for ensemble
        tta_transforms: TTA transforms
        device: Device for inference
        img_size: Target image size
        image_col: Column name for image path

    Returns:
        Dict mapping image_path to 5 target predictions
    """
    predictions = {}

    unique_images = test_df[image_col].unique()

    for image_path in tqdm(unique_images, desc="Inference"):
        full_path = image_dir / image_path
        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict (3 values: Dry_Total_g, GDM_g, Dry_Green_g)
        pred_3 = predict_single_image(image, backbone, head_models, tta_transforms, device, img_size)

        # Derive all 5 targets from 3 predictions
        pred_5 = derive_all_targets(pred_3.reshape(1, -1))[0]

        predictions[image_path] = pred_5

    return predictions


def create_submission(
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create submission file.

    Args:
        test_df: Test DataFrame with columns [sample_id, image_path, target_name]
        predictions: Dict mapping image_path to 5 target predictions

    Returns:
        Submission DataFrame with columns [sample_id, target]
    """
    target_to_idx = {
        "Dry_Green_g": 0,
        "Dry_Dead_g": 1,
        "Dry_Clover_g": 2,
        "GDM_g": 3,
        "Dry_Total_g": 4,
    }

    submission_rows = []
    for _, row in test_df.iterrows():
        sample_id = row["sample_id"]
        image_path = row["image_path"]
        target_name = row["target_name"]

        pred = predictions[image_path]
        idx = target_to_idx[target_name]
        value = pred[idx]

        submission_rows.append(
            {
                "sample_id": sample_id,
                "target": value,
            }
        )

    return pd.DataFrame(submission_rows)


def main():
    parser = argparse.ArgumentParser(description="DINOv3 inference for biomass prediction")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name",
    )
    parser.add_argument("--folds", type=str, default="all", help="Folds to use (e.g., '0,1,2' or 'all')")
    parser.add_argument("--img_size", type=int, default=960, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--weight_type",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Weight type to use",
    )
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate image size
    if args.img_size % 16 != 0:
        raise ValueError(f"Image size must be divisible by 16, got {args.img_size}")

    # Setup paths
    run_dir = config.OUTPUT_DIR / args.experiment_dir / args.run_name
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Run directory: {run_dir}")

    # Determine folds
    if args.folds == "all":
        folds = list(range(5))
    else:
        folds = [int(f) for f in args.folds.split(",")]

    print(f"Using folds: {folds}")

    # Load DINOv3 backbone
    print("Loading DINOv3 backbone...")
    backbone = build_backbone(pretrained=True, device=str(device))
    print(f"Backbone loaded: {backbone.MODEL_NAME}")

    # Load head models
    print(f"Weight type: {args.weight_type}")
    head_models, _ = load_head_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
        weight_type=args.weight_type,
    )

    # Setup TTA transforms
    tta_transforms = build_tta_transforms()
    print(f"Using {len(tta_transforms)} TTA transforms")

    # Load test data
    test_csv = config.get_test_csv_path()
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # Run inference
    predictions = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        backbone=backbone,
        head_models=head_models,
        tta_transforms=tta_transforms,
        device=device,
        img_size=args.img_size,
    )

    # Create submission
    submission_df = create_submission(test_df, predictions)

    # Save submission
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Save predictions in wide format
    pred_path = output_dir / "predictions.csv"
    all_targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    pred_df = pd.DataFrame.from_dict(predictions, orient="index", columns=all_targets)
    pred_df.index.name = "image_path"
    pred_df.to_csv(pred_path)
    print(f"Predictions saved to {pred_path}")


# Kaggle notebook inference function
def kaggle_inference(
    run_name: str,
    folds: list[int] | None = None,
    img_size: int = 960,
    weight_type: str = "best",
) -> pd.DataFrame:
    """Inference function for Kaggle notebook.

    Args:
        run_name: Run name
        folds: List of fold numbers (default: all 5 folds)
        img_size: Image size
        weight_type: Type of weights to load

    Returns:
        Submission DataFrame
    """
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if folds is None:
        folds = list(range(5))

    # Validate image size
    if img_size % 16 != 0:
        raise ValueError(f"Image size must be divisible by 16, got {img_size}")

    print(f"Weight type: {weight_type}")

    # Load backbone
    print("Loading DINOv3 backbone...")
    backbone = build_backbone(pretrained=True, device=str(device))

    # Load head models from artifact directory
    run_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / run_name
    head_models, _ = load_head_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
        weight_type=weight_type,
    )

    # Setup TTA
    tta_transforms = build_tta_transforms()

    # Load test data
    test_df = pd.read_csv(config.get_test_csv_path())

    # Run inference
    predictions = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        backbone=backbone,
        head_models=head_models,
        tta_transforms=tta_transforms,
        device=device,
        img_size=img_size,
    )

    # Create submission
    submission_df = create_submission(test_df, predictions)

    return submission_df


if __name__ == "__main__":
    main()
