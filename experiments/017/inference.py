"""Inference script for regression heads with DINOv3 backbone feature extraction.

Supports:
- Multi-fold ensemble for any head type (SVR, Ridge, Lasso, etc.)
- Test Time Augmentation (TTA) with 4 flip variants
- On-the-fly DINOv3 backbone feature extraction
"""

# isort: off
# config must be imported first to setup paths via rootutils
import config  # noqa: F401
# isort: on

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.backbone import build_backbone, load_backbone_weights
from src.feature_engine import FeatureEngine
from src.heads import MultiTargetHead, load_head
from src.metric import derive_all_targets
from src.seed import seed_everything


def load_backbone(
    weights_path: Path | None = None,
    device: torch.device = torch.device("cuda"),
) -> torch.nn.Module:
    """Load DINOv3 backbone.

    Args:
        weights_path: Path to saved backbone weights. If None, download from timm.
        device: Device to place model on.

    Returns:
        DINOv3Backbone instance.
    """
    if weights_path is not None and weights_path.exists():
        print(f"Loading backbone from local weights: {weights_path}")
        return load_backbone_weights(str(weights_path), device=str(device))
    else:
        print("Loading backbone from timm (online)...")
        return build_backbone(pretrained=True, device=str(device))


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


def detect_head_type(model_dir: Path) -> str:
    """Detect head type from model files in directory.

    Args:
        model_dir: Directory containing trained models.

    Returns:
        Detected head type string.
    """
    weights_dir = model_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    # Look for model files
    model_files = list(weights_dir.glob("*_fold*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {weights_dir}")

    # Extract head type from filename (e.g., "ridge_fold0.pkl" -> "ridge")
    filename = model_files[0].stem  # e.g., "ridge_fold0"
    head_type = filename.rsplit("_fold", 1)[0]

    return head_type


def load_feature_engines(
    model_dir: Path,
    folds: list[int],
) -> list[FeatureEngine | None]:
    """Load feature engines from all folds if they exist.

    Args:
        model_dir: Directory containing trained models.
        folds: List of fold numbers to load.

    Returns:
        List of loaded feature engines (or None if not found for each fold).
    """
    engines = []

    for fold in folds:
        engine_path = model_dir / "weights" / f"feature_engine_fold{fold}.pkl"
        if engine_path.exists():
            engine = FeatureEngine.load(engine_path)
            engines.append(engine)
            print(f"Loaded feature engine from {engine_path}")
        else:
            engines.append(None)

    return engines


def load_head_models(
    model_dir: Path,
    folds: list[int],
    head_type: str | None = None,
) -> tuple[list[MultiTargetHead], str]:
    """Load head models from all folds.

    Args:
        model_dir: Directory containing trained models.
        folds: List of fold numbers to load.
        head_type: Head type (auto-detected if None).

    Returns:
        Tuple of (list of loaded models, head_type).
    """
    if head_type is None:
        head_type = detect_head_type(model_dir)

    models = []

    for fold in folds:
        model_path = model_dir / "weights" / f"{head_type}_fold{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = load_head(model_path)
        models.append(model)
        print(f"Loaded {head_type} model from {model_path}")

    return models, head_type


@torch.inference_mode()
def predict_single_image(
    image: np.ndarray,
    backbone: torch.nn.Module,
    head_models: list[MultiTargetHead],
    tta_transforms: list[tuple[str, A.Compose]],
    device: torch.device,
    img_size: int = 960,
    feature_engines: list[FeatureEngine | None] | None = None,
) -> np.ndarray:
    """Predict for a single image using DINOv3 backbone and head models.

    Args:
        image: Original image [H, W, C] in RGB
        backbone: DINOv3 backbone for feature extraction
        head_models: List of head models for fold ensemble
        tta_transforms: List of (name, transform) for TTA
        device: Device for inference
        img_size: Target image size for DINOv3
        feature_engines: List of feature engines for each fold (or None)

    Returns:
        Averaged predictions [3] (Dry_Total_g, GDM_g, Dry_Green_g)
    """
    from src.coverage import calculate_coverage

    all_preds = []

    # Calculate coverage once (before resize, using original image)
    coverage_raw, coverage_log = calculate_coverage(image)

    # Resize to target size
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    for _, tta_transform in tta_transforms:
        # Apply TTA transform
        aug_image = tta_transform(image=image_resized)["image"]

        # Preprocess and extract features
        img_tensor = preprocess_image(aug_image).to(device)

        with torch.no_grad():
            cls_token, patch_tokens = backbone(img_tensor)

        # Convert to features
        cls_np = cls_token.cpu().numpy()[0]
        patch_mean = patch_tokens.cpu().numpy()[0].mean(axis=0)
        # Concatenate DINOv3 features + coverage features
        features = np.concatenate([cls_np, patch_mean, [coverage_raw, coverage_log]]).reshape(1, -1)

        # Predict with each head model (and optionally apply feature engine)
        for i, head_model in enumerate(head_models):
            # Apply feature engine if available
            if feature_engines is not None and feature_engines[i] is not None:
                features_transformed = feature_engines[i].transform(features)
            else:
                features_transformed = features

            pred = head_model.predict(features_transformed)[0]
            all_preds.append(pred)

    # Average all predictions
    return np.mean(all_preds, axis=0)


def run_inference(
    test_df: pd.DataFrame,
    image_dir: Path,
    backbone: torch.nn.Module,
    head_models: list[MultiTargetHead],
    tta_transforms: list[tuple[str, A.Compose]],
    device: torch.device,
    img_size: int = 960,
    image_col: str = "image_path",
    feature_engines: list[FeatureEngine | None] | None = None,
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
        feature_engines: List of feature engines for each fold (or None)

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
        pred_3 = predict_single_image(image, backbone, head_models, tta_transforms, device, img_size, feature_engines)

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
    parser = argparse.ArgumentParser(description="Inference for biomass prediction")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory name (e.g., 20241228_123456_exp017_ridge)",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default=None,
        help="Head type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Folds to use (e.g., '0,1,2' or 'all')",
    )
    parser.add_argument("--img_size", type=int, default=960, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default=None,
        help="Path to backbone weights (for offline inference)",
    )
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate image size
    if args.img_size % 16 != 0:
        raise ValueError(f"Image size must be divisible by 16, got {args.img_size}")

    # Setup paths
    model_dir = config.OUTPUT_DIR / args.experiment_dir
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Model directory: {model_dir}")

    # Determine folds
    if args.folds == "all":
        folds = list(range(5))
    else:
        folds = [int(f) for f in args.folds.split(",")]

    print(f"Using folds: {folds}")

    # Load DINOv3 backbone
    backbone_weights_path = Path(args.backbone_weights) if args.backbone_weights else None
    backbone = load_backbone(weights_path=backbone_weights_path, device=device)
    print(f"Backbone loaded: {backbone.MODEL_NAME}")

    # Load head models
    head_models, head_type = load_head_models(model_dir, folds, args.head_type)
    print(f"Loaded {len(head_models)} {head_type} models")

    # Load feature engines (if available)
    feature_engines = load_feature_engines(model_dir, folds)
    if any(engine is not None for engine in feature_engines):
        print("Using feature engines for preprocessing (PCA/PLS)")
    else:
        feature_engines = None

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
        feature_engines=feature_engines,
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
    experiment_dir: str,
    folds: list[int] | None = None,
    img_size: int = 960,
    head_type: str | None = None,
) -> pd.DataFrame:
    """Inference function for Kaggle notebook.

    Args:
        experiment_dir: Experiment directory name
        folds: List of fold numbers (default: all 5 folds)
        img_size: Image size
        head_type: Head type (auto-detected if None)

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

    # Load backbone from artifact directory (for offline inference)
    artifact_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME)
    backbone_weights_path = artifact_dir / "backbone.pth"
    backbone = load_backbone(weights_path=backbone_weights_path, device=device)
    print(f"Backbone loaded: {backbone.MODEL_NAME}")

    # Load head models from artifact directory
    model_dir = artifact_dir / experiment_dir
    head_models, detected_head_type = load_head_models(model_dir, folds, head_type)
    print(f"Loaded {len(head_models)} {detected_head_type} models")

    # Load feature engines (if available)
    feature_engines = load_feature_engines(model_dir, folds)
    if any(engine is not None for engine in feature_engines):
        print("Using feature engines for preprocessing (PCA/PLS)")
    else:
        feature_engines = None

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
        feature_engines=feature_engines,
    )

    # Create submission
    submission_df = create_submission(test_df, predictions)

    return submission_df


if __name__ == "__main__":
    main()
