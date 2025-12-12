"""Inference script for biomass prediction.

Supports:
- Multi-fold model ensemble
- Test Time Augmentation (TTA) applied to original image before split
- Dual input (left-right split) model
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
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.data import TARGET_COLS_PRED, build_tta_pre_split_transforms
from src.metric import derive_all_targets
from src.model import build_model


def build_post_split_transform(
    img_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> A.Compose:
    """Build post-split transform (resize, normalize, to_tensor)."""
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def load_models(
    run_dir: Path,
    folds: list[int],
    device: torch.device,
) -> tuple[list[torch.nn.Module], dict]:
    """Load models from all folds.

    Expected directory structure (created by train.py):
        run_dir/
        ├── weights/
        │   ├── best_fold0.pth
        │   ├── best_fold1.pth
        │   └── ...
        └── config.yaml

    Args:
        run_dir: Directory containing trained models (weights/, config.yaml)
        folds: List of fold numbers to load
        device: Device to load models to

    Returns:
        Tuple of (list of loaded models, model config dict)
    """
    models = []
    model_config = None

    # Load config (same for all folds)
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(config_path))
        model_cfg = OmegaConf.to_container(cfg, resolve=True).get("model", {})
    else:
        # Fallback to JSON config
        json_path = run_dir / "config.json"
        if json_path.exists():
            with open(json_path) as f:
                model_cfg = json.load(f)
        else:
            model_cfg = {}

    model_config = model_cfg

    for fold in folds:
        # Build model
        model = build_model(
            model_name=model_cfg.get("backbone", "tf_efficientnetv2_b0.in1k"),
            num_outputs=model_cfg.get("num_outputs", 3),
            pretrained=False,
            dropout=model_cfg.get("dropout", 0.2),
            hidden_size=model_cfg.get("hidden_size", 512),
            device=device,
        )

        # Load weights from run_dir/weights/
        weights_dir = run_dir / "weights"
        weight_path = weights_dir / f"best_fold{fold}.pth"

        if not weight_path.exists():
            # Try alternative paths
            alt_paths = [
                weights_dir / f"last_fold{fold}.pth",
                run_dir / f"best_fold{fold}.pth",
            ]
            for alt in alt_paths:
                if alt.exists():
                    weight_path = alt
                    break

        if not weight_path.exists():
            raise FileNotFoundError(f"Cannot find weights for fold {fold} in {weights_dir}")

        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

        print(f"Loaded model from {weight_path}")

    return models, model_config


@torch.inference_mode()
def predict_single_image(
    image: np.ndarray,
    models: list[torch.nn.Module],
    tta_transforms: list[tuple[str, A.Compose]],
    post_split_transform: A.Compose,
    device: torch.device,
) -> np.ndarray:
    """Predict for a single image using dual input model.

    TTA is applied to the original image BEFORE splitting.

    Args:
        image: Original image [H, W, C] in RGB (e.g., 1000x2000)
        models: List of models for ensemble
        tta_transforms: List of (name, transform) for TTA (applied before split)
        post_split_transform: Transform for each half (resize, normalize, to_tensor)
        device: Device for inference

    Returns:
        Averaged predictions [num_outputs]
    """
    all_preds = []

    for _, tta_transform in tta_transforms:
        # 1. Apply TTA to entire image
        aug_image = tta_transform(image=image)["image"]

        # 2. Split into left and right halves
        mid = aug_image.shape[1] // 2
        image_left = aug_image[:, :mid, :]
        image_right = aug_image[:, mid:, :]

        # 3. Apply post-split transform to each half
        left_tensor = post_split_transform(image=image_left)["image"].unsqueeze(0).to(device)
        right_tensor = post_split_transform(image=image_right)["image"].unsqueeze(0).to(device)

        # 4. Predict with each model
        for model in models:
            pred = model(left_tensor, right_tensor)
            all_preds.append(pred.cpu().numpy()[0])

    # Average all predictions
    return np.mean(all_preds, axis=0)


def run_inference(
    test_df: pd.DataFrame,
    image_dir: Path,
    models: list[torch.nn.Module],
    tta_transforms: list[tuple[str, A.Compose]],
    post_split_transform: A.Compose,
    device: torch.device,
    image_col: str = "image_path",
) -> pd.DataFrame:
    """Run inference on test set.

    Args:
        test_df: Test DataFrame (unique images)
        image_dir: Directory containing images
        models: List of models for ensemble
        tta_transforms: TTA transforms (applied before split)
        post_split_transform: Post-split transform
        device: Device for inference
        image_col: Column name for image path

    Returns:
        DataFrame with predictions (3 columns)
    """
    predictions = []
    image_ids = []

    # Get unique images
    unique_images = test_df[image_col].unique()

    for image_path in tqdm(unique_images, desc="Inference"):
        # Load image
        full_path = image_dir / image_path
        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict
        pred = predict_single_image(image, models, tta_transforms, post_split_transform, device)
        predictions.append(pred)

        # Extract image_id from path
        image_id = Path(image_path).stem
        image_ids.append(image_id)

    # Create predictions DataFrame
    pred_df = pd.DataFrame(predictions, columns=TARGET_COLS_PRED)
    pred_df["image_id"] = image_ids
    pred_df["image_path"] = unique_images

    return pred_df


def create_submission(
    pred_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Create submission file in Long format.

    Args:
        pred_df: DataFrame with predictions (Wide format, 3 columns)
        sample_submission: Sample submission DataFrame

    Returns:
        Submission DataFrame in Long format
    """
    # Derive all 5 targets from 3 predictions
    pred_3 = pred_df[TARGET_COLS_PRED].values
    pred_5 = derive_all_targets(pred_3)

    # Create full predictions DataFrame
    all_targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    full_pred_df = pd.DataFrame(pred_5, columns=all_targets)
    full_pred_df["image_id"] = pred_df["image_id"].values

    # Convert to Long format
    submission_rows = []
    for _, row in full_pred_df.iterrows():
        image_id = row["image_id"]
        for target in all_targets:
            sample_id = f"{image_id}__{target}"
            submission_rows.append(
                {
                    "sample_id": sample_id,
                    "target": row[target],
                }
            )

    submission_df = pd.DataFrame(submission_rows)

    # Ensure correct order (match sample_submission)
    submission_df = submission_df.set_index("sample_id")
    submission_df = submission_df.loc[sample_submission["sample_id"]]
    submission_df = submission_df.reset_index()

    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Inference for biomass prediction")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory name (e.g., '20251212_111833_exp001')",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name (e.g., '001_tf_efficientnetv2_b0_in1k__img_size-224__lr-0_001')",
    )
    parser.add_argument("--folds", type=str, default="all", help="Folds to use (e.g., '0,1,2' or 'all')")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup paths
    # Local: config.OUTPUT_DIR / experiment_dir / run_name
    run_dir = config.OUTPUT_DIR / args.experiment_dir / args.run_name
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Run directory: {run_dir}")

    # Determine folds
    if args.folds == "all":
        folds = list(range(5))  # 5-fold (0-4)
    else:
        folds = [int(f) for f in args.folds.split(",")]

    print(f"Using folds: {folds}")

    # Load models
    models, _ = load_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
    )

    # Setup transforms (TTA always enabled)
    tta_transforms = build_tta_pre_split_transforms()
    post_split_transform = build_post_split_transform(args.img_size)

    print(f"Using {len(tta_transforms)} TTA transforms")

    # Load test data
    test_csv = config.get_test_csv_path()
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # Run inference
    pred_df = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        models=models,
        tta_transforms=tta_transforms,
        post_split_transform=post_split_transform,
        device=device,
    )

    # Create submission
    sample_submission = pd.read_csv(config.get_sample_submission_path())
    submission_df = create_submission(pred_df, sample_submission)

    # Save submission
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Also save predictions in wide format for analysis
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")


# Kaggle notebook inference function
def kaggle_inference(
    run_name: str,
    folds: list[int] | None = None,
    img_size: int = 224,
) -> pd.DataFrame:
    """Inference function for Kaggle notebook.

    On Kaggle, the run directory structure is:
        /kaggle/input/{competition}-artifacts/other/{exp_name}/1/{run_name}/
        ├── weights/
        │   ├── best_fold0.pth
        │   └── ...
        └── config.yaml

    Args:
        run_name: Run name (e.g., '001_tf_efficientnetv2_b0_in1k__img_size-224__lr-0_001')
        folds: List of fold numbers (default: all 5 folds, 0-4)
        img_size: Image size

    Returns:
        Submission DataFrame
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if folds is None:
        folds = list(range(5))  # 5-fold (0-4)

    # Load models from artifact directory
    # Kaggle: ARTIFACT_EXP_DIR / run_name (no experiment_dir needed)
    run_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / run_name

    models, _ = load_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
    )

    # Setup transforms (TTA always enabled)
    tta_transforms = build_tta_pre_split_transforms()
    post_split_transform = build_post_split_transform(img_size)

    # Load test data
    test_df = pd.read_csv(config.get_test_csv_path())

    # Run inference
    pred_df = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        models=models,
        tta_transforms=tta_transforms,
        post_split_transform=post_split_transform,
        device=device,
    )

    # Create submission
    sample_submission = pd.read_csv(config.get_sample_submission_path())
    submission_df = create_submission(pred_df, sample_submission)

    return submission_df


if __name__ == "__main__":
    main()
