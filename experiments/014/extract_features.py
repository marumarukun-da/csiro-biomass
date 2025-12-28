"""Script to extract and save DINOv3 features for all training images."""

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

from src.backbone import build_backbone, save_backbone_weights
from src.data import convert_long_to_wide


def build_augmentations() -> list[tuple[str, A.Compose]]:
    """Build 20 augmentation patterns.

    Pattern structure:
    - 0: original
    - 1: hflip
    - 2: vflip
    - 3: hvflip
    - 4-7: original + color variants (brightness, contrast, hsv, gamma)
    - 8-11: hflip + color variants
    - 12-15: vflip + color variants
    - 16-19: hvflip + color variants

    Returns:
        List of (name, transform) tuples.
    """
    # Base flip transforms
    flip_transforms = [
        ("original", A.Compose([])),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)])),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)])),
        ("hvflip", A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])),
    ]

    # Color transforms
    color_transforms = [
        ("brightness", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)),
        ("contrast", A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)),
        ("hsv", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0)),
        ("gamma", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ]

    augmentations = []

    # First 4: flip only
    for flip_name, flip_transform in flip_transforms:
        augmentations.append((flip_name, flip_transform))

    # Remaining 16: flip + color
    for flip_name, flip_transform in flip_transforms:
        for color_name, color_transform in color_transforms:
            combined = A.Compose(
                [
                    *flip_transform.transforms,
                    color_transform,
                ]
            )
            augmentations.append((f"{flip_name}_{color_name}", combined))

    return augmentations


def extract_features_for_image(
    image_path: Path,
    backbone: torch.nn.Module,
    augmentations: list[tuple[str, A.Compose]],
    img_size: int = 960,
    device: torch.device = torch.device("cuda"),
) -> dict[str, np.ndarray]:
    """Extract features for a single image with all augmentations.

    Args:
        image_path: Path to image file.
        backbone: DINOv3 backbone model.
        augmentations: List of (name, transform) tuples.
        img_size: Target image size (must be divisible by 16).
        device: Device for inference.

    Returns:
        Dict with cls_i and patches_i for each augmentation i.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Normalize parameters (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    features = {}

    for i, (aug_name, transform) in enumerate(augmentations):
        # Apply augmentation
        augmented = transform(image=image)["image"]

        # Normalize and convert to tensor
        img_normalized = (augmented / 255.0 - mean) / std
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Extract features
        cls_token, patch_tokens = backbone(img_tensor)

        # Store as numpy arrays
        features[f"cls_{i}"] = cls_token.cpu().numpy().squeeze(0)
        features[f"patches_{i}"] = patch_tokens.cpu().numpy().squeeze(0)

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--img_size", type=int, default=960, help="Image size (must be divisible by 16)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers (currently only 1 supported)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images that already have features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate image size
    if args.img_size % 16 != 0:
        raise ValueError(f"Image size must be divisible by 16, got {args.img_size}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print("Loading DINOv3 backbone...")
    backbone = build_backbone(pretrained=True, device=str(device))
    print(f"Backbone loaded: {backbone.MODEL_NAME}")
    print(f"  - Hidden dim: {backbone.hidden_dim}")
    print(f"  - Num patches for {args.img_size}x{args.img_size}: {backbone.get_num_patches(args.img_size)}")

    # Save backbone weights for offline inference (Kaggle submission)
    backbone_save_path = output_dir.parent / "backbone.pth"
    if not backbone_save_path.exists():
        save_backbone_weights(backbone, str(backbone_save_path))
        print(f"Backbone weights saved to {backbone_save_path}")
    else:
        print(f"Backbone weights already exist at {backbone_save_path}")

    # Build augmentations
    augmentations = build_augmentations()
    print(f"Using {len(augmentations)} augmentation patterns:")
    for i, (name, _) in enumerate(augmentations):
        print(f"  [{i}] {name}")

    # Load training data
    train_csv = config.get_train_csv_path()
    train_df = pd.read_csv(train_csv)
    train_df = convert_long_to_wide(train_df)

    image_dir = config.get_image_dir()

    # Calculate expected storage
    hidden_dim = backbone.hidden_dim
    num_patches = backbone.get_num_patches(args.img_size)
    num_augs = len(augmentations)
    bytes_per_image = (hidden_dim + num_patches * hidden_dim) * 4 * num_augs  # float32
    total_bytes = bytes_per_image * len(train_df)
    print("\nExpected storage (uncompressed):")
    print(f"  - Per image: {bytes_per_image / 1024 / 1024:.1f} MB")
    print(f"  - Total: {total_bytes / 1024 / 1024 / 1024:.1f} GB")

    # Extract features for each image
    print(f"\nExtracting features for {len(train_df)} images...")
    skipped = 0
    processed = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_id = row["image_id"]
        image_path = image_dir / row["image_path"]

        output_path = output_dir / f"{image_id}.npz"

        # Skip if already exists
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        # Extract features
        try:
            features = extract_features_for_image(
                image_path=image_path,
                backbone=backbone,
                augmentations=augmentations,
                img_size=args.img_size,
                device=device,
            )

            # Save as compressed npz
            np.savez_compressed(output_path, **features)
            processed += 1

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    print("\nFeature extraction complete!")
    print(f"  - Processed: {processed}")
    print(f"  - Skipped (existing): {skipped}")
    print(f"  - Output directory: {output_dir}")


if __name__ == "__main__":
    main()
