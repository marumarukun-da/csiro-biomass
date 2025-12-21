"""Inference script for biomass prediction.

Supports:
- Multi-fold model ensemble
- Test Time Augmentation (TTA) applied to original image before split
- Dual input (left-right split) density map model
- Automatic submission creation
- Density map visualization
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

from src.data import build_tta_pre_split_transforms
from src.metric import derive_all_targets
from src.model import build_model
from src.seed import seed_everything


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
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def load_models(
    run_dir: Path,
    folds: list[int],
    device: torch.device,
    weight_type: str = "best",
) -> tuple[list[torch.nn.Module], dict]:
    """Load models from all folds.

    Expected directory structure (created by train.py):
        run_dir/
        ├── weights/
        │   ├── best_fold0.pth
        │   ├── best_fold1.pth
        │   ├── last_fold0.pth
        │   ├── last_fold1.pth
        │   └── ...
        └── config.yaml

    Args:
        run_dir: Directory containing trained models (weights/, config.yaml)
        folds: List of fold numbers to load
        device: Device to load models to
        weight_type: Type of weights to load ("best" or "last")

    Returns:
        Tuple of (list of loaded models, model config dict)
    """
    if weight_type not in ("best", "last"):
        raise ValueError(f"weight_type must be 'best' or 'last', got '{weight_type}'")

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
        # Build model (DensityMapModel)
        model = build_model(
            backbone=model_cfg.get("backbone", "maxvit_small_tf_512.in1k"),
            decoder_channels=model_cfg.get("decoder_channels", [512, 256, 128, 64]),
            num_outputs=model_cfg.get("num_outputs", 3),
            pretrained=False,
            device=device,
        )

        # Load weights from run_dir/weights/
        weights_dir = run_dir / "weights"

        # Primary and fallback paths based on weight_type
        if weight_type == "best":
            primary_path = weights_dir / f"best_fold{fold}.pth"
            fallback_paths = [
                weights_dir / f"last_fold{fold}.pth",
                run_dir / f"best_fold{fold}.pth",
            ]
        else:  # weight_type == "last"
            primary_path = weights_dir / f"last_fold{fold}.pth"
            fallback_paths = [
                weights_dir / f"best_fold{fold}.pth",
                run_dir / f"last_fold{fold}.pth",
            ]

        weight_path = primary_path
        if not weight_path.exists():
            # Try fallback paths
            for alt in fallback_paths:
                if alt.exists():
                    weight_path = alt
                    print(f"Warning: {primary_path.name} not found, using {alt.name}")
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
    visualize_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run inference on test set.

    Args:
        test_df: Test DataFrame
        image_dir: Directory containing images
        models: List of models for ensemble
        tta_transforms: TTA transforms (applied before split)
        post_split_transform: Post-split transform
        device: Device for inference
        image_col: Column name for image path
        visualize_dir: Directory to save density map visualizations (None = no visualization)

    Returns:
        Dict mapping image_path to 5 target predictions
        Order: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    predictions = {}

    # Setup visualization directory
    if visualize_dir is not None:
        visualize_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving density map visualizations to: {visualize_dir}")

    # Get unique images
    unique_images = test_df[image_col].unique()

    for image_path in tqdm(unique_images, desc="Inference"):
        # Load image
        full_path = image_dir / image_path
        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict (3 values: Dry_Dead_g, Dry_Green_g, Dry_Clover_g)
        pred_3 = predict_single_image(image, models, tta_transforms, post_split_transform, device)

        # Derive all 5 targets from 3 predictions
        # Input: [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]
        # Output: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
        pred_5 = derive_all_targets(pred_3.reshape(1, -1))[0]

        predictions[image_path] = pred_5

        # Visualize density maps (using first model, no TTA)
        if visualize_dir is not None:
            _, left_density, right_density = predict_with_density_maps(
                image, models[0], post_split_transform, device
            )
            save_path = visualize_dir / f"{Path(image_path).stem}_density.png"
            visualize_density_map(left_density, right_density, save_path)

    return predictions


def create_submission(
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create submission file using test_df's sample_id directly.

    Args:
        test_df: Test DataFrame with columns [sample_id, image_path, target_name]
        predictions: Dict mapping image_path to 5 target predictions
                    Order: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]

    Returns:
        Submission DataFrame with columns [sample_id, target]
    """
    # Map target_name to prediction index
    # derive_all_targets returns: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    target_to_idx = {
        "Dry_Green_g": 0,
        "Dry_Dead_g": 1,
        "Dry_Clover_g": 2,
        "GDM_g": 3,
        "Dry_Total_g": 4,
    }

    submission_rows = []
    for _, row in test_df.iterrows():
        sample_id = row["sample_id"]  # Use test_df's sample_id directly
        image_path = row["image_path"]
        target_name = row["target_name"]  # Use test_df's target_name directly

        # Get prediction for this image
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
    parser.add_argument(
        "--weight_type",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Weight type to use: 'best' (highest R²) or 'last' (final epoch)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save density map visualizations for each image",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(42)

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
    print(f"Weight type: {args.weight_type}")
    models, _ = load_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
        weight_type=args.weight_type,
    )

    # Setup transforms (TTA always enabled)
    tta_transforms = build_tta_pre_split_transforms()
    post_split_transform = build_post_split_transform(args.img_size)

    print(f"Using {len(tta_transforms)} TTA transforms")

    # Load test data
    test_csv = config.get_test_csv_path()
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # Setup visualization directory if requested
    visualize_dir = output_dir / "density_maps" if args.visualize else None

    # Run inference (returns dict: image_path -> 5 predictions)
    predictions = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        models=models,
        tta_transforms=tta_transforms,
        post_split_transform=post_split_transform,
        device=device,
        visualize_dir=visualize_dir,
    )

    # Create submission using test_df's sample_id directly
    submission_df = create_submission(test_df, predictions)

    # Save submission
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Also save predictions in wide format for analysis
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
    img_size: int = 224,
    weight_type: str = "best",
) -> pd.DataFrame:
    """Inference function for Kaggle notebook.

    On Kaggle, the run directory structure is:
        /kaggle/input/{competition}-artifacts/other/{exp_name}/1/{run_name}/
        ├── weights/
        │   ├── best_fold0.pth
        │   ├── last_fold0.pth
        │   └── ...
        └── config.yaml

    Args:
        run_name: Run name (e.g., '001_tf_efficientnetv2_b0_in1k__img_size-224__lr-0_001')
        folds: List of fold numbers (default: all 5 folds, 0-4)
        img_size: Image size
        weight_type: Type of weights to load ("best" or "last")

    Returns:
        Submission DataFrame
    """
    # Set seed for reproducibility
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if folds is None:
        folds = list(range(5))  # 5-fold (0-4)

    print(f"Weight type: {weight_type}")

    # Load models from artifact directory
    # Kaggle: ARTIFACT_EXP_DIR / run_name (no experiment_dir needed)
    run_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / run_name

    models, _ = load_models(
        run_dir=run_dir,
        folds=folds,
        device=device,
        weight_type=weight_type,
    )

    # Setup transforms (TTA always enabled)
    tta_transforms = build_tta_pre_split_transforms()
    post_split_transform = build_post_split_transform(img_size)

    # Load test data
    test_df = pd.read_csv(config.get_test_csv_path())

    # Run inference (returns dict: image_path -> 5 predictions)
    predictions = run_inference(
        test_df=test_df,
        image_dir=config.get_image_dir(),
        models=models,
        tta_transforms=tta_transforms,
        post_split_transform=post_split_transform,
        device=device,
    )

    # Create submission using test_df's sample_id directly
    submission_df = create_submission(test_df, predictions)

    return submission_df


def visualize_density_map(
    left_density: np.ndarray,
    right_density: np.ndarray,
    save_path: Path,
    target_names: list[str] | None = None,
) -> None:
    """Visualize density maps as heatmaps.

    Args:
        left_density: Left image density map [num_outputs, H, W]
        right_density: Right image density map [num_outputs, H, W]
        save_path: Path to save the visualization
        target_names: Names for each target channel
    """
    import matplotlib.pyplot as plt

    if target_names is None:
        target_names = ["Dead", "Green", "Clover"]

    num_outputs = left_density.shape[0]
    fig, axes = plt.subplots(2, num_outputs, figsize=(5 * num_outputs, 10))

    for i, name in enumerate(target_names):
        # Left density map
        im_left = axes[0, i].imshow(left_density[i], cmap="hot", vmin=0)
        axes[0, i].set_title(f"Left {name}: {left_density[i].sum():.2f}g")
        axes[0, i].axis("off")
        plt.colorbar(im_left, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Right density map
        im_right = axes[1, i].imshow(right_density[i], cmap="hot", vmin=0)
        axes[1, i].set_title(f"Right {name}: {right_density[i].sum():.2f}g")
        axes[1, i].axis("off")
        plt.colorbar(im_right, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.suptitle(f"Total: {(left_density.sum() + right_density.sum()):.2f}g", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def predict_with_density_maps(
    image: np.ndarray,
    model: torch.nn.Module,
    post_split_transform: A.Compose,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict for a single image and return density maps.

    Args:
        image: Original image [H, W, C] in RGB
        model: Single model (not ensemble)
        post_split_transform: Transform for each half
        device: Device for inference

    Returns:
        Tuple of (predictions [3], left_density [3, H, W], right_density [3, H, W])
    """
    # Split into left and right halves
    mid = image.shape[1] // 2
    image_left = image[:, :mid, :]
    image_right = image[:, mid:, :]

    # Apply post-split transform
    left_tensor = post_split_transform(image=image_left)["image"].unsqueeze(0).to(device)
    right_tensor = post_split_transform(image=image_right)["image"].unsqueeze(0).to(device)

    # Predict with density maps
    pred, left_density, right_density = model(left_tensor, right_tensor, return_density_maps=True)

    return (
        pred.cpu().numpy()[0],
        left_density.cpu().numpy()[0],
        right_density.cpu().numpy()[0],
    )


if __name__ == "__main__":
    main()
