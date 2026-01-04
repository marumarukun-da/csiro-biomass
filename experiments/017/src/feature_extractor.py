"""Online feature extraction using DINOv3 backbone.

This module provides functions to extract features from images on-the-fly
using the DINOv3 backbone, instead of loading from precomputed .npz files.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .backbone import DINOv3Backbone, build_backbone, load_backbone_weights
from .coverage import calculate_coverage


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


def load_backbone(
    weights_path: Path | str | None = None,
    device: torch.device | str = "cuda",
) -> DINOv3Backbone:
    """Load DINOv3 backbone.

    Args:
        weights_path: Path to saved backbone weights. If None, download from timm.
        device: Device to place model on.

    Returns:
        DINOv3Backbone instance.
    """
    device_str = str(device)
    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            print(f"Loading backbone from local weights: {weights_path}")
            return load_backbone_weights(str(weights_path), device=device_str)

    print("Loading backbone from timm (online)...")
    return build_backbone(pretrained=True, device=device_str)


@torch.inference_mode()
def extract_features_single(
    image: np.ndarray,
    backbone: nn.Module,
    device: torch.device,
    img_size: int = 960,
) -> np.ndarray:
    """Extract features from a single image.

    Args:
        image: Original image [H, W, C] in RGB
        backbone: DINOv3 backbone
        device: Device for inference
        img_size: Target image size for DINOv3

    Returns:
        Feature vector [2562] (CLS 1280 + PatchMean 1280 + coverage 2)
    """
    # Calculate coverage from original image (before resize)
    coverage_raw, coverage_log = calculate_coverage(image)

    # Resize to target size
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Preprocess and extract features
    img_tensor = preprocess_image(image_resized).to(device)

    with torch.no_grad():
        cls_token, patch_tokens = backbone(img_tensor)

    # Convert to numpy
    cls_np = cls_token.cpu().numpy()[0]  # [1280]
    patch_mean = patch_tokens.cpu().numpy()[0].mean(axis=0)  # [1280]

    # Concatenate: CLS + PatchMean + coverage
    features = np.concatenate([cls_np, patch_mean, [coverage_raw, coverage_log]])

    return features


def extract_all_features(
    df: pd.DataFrame,
    image_dir: Path | str,
    backbone: nn.Module,
    device: torch.device,
    img_size: int = 960,
    image_col: str = "image_path",
) -> np.ndarray:
    """Extract features from all images in DataFrame.

    Args:
        df: DataFrame with image_path column.
        image_dir: Base directory containing images.
        backbone: DINOv3 backbone.
        device: Device for inference.
        img_size: Target image size.
        image_col: Column name for image path.

    Returns:
        Feature matrix [N, 2562]
    """
    image_dir = Path(image_dir)
    features_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        image_path = image_dir / row[image_col]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feat = extract_features_single(image, backbone, device, img_size)
        features_list.append(feat)

    return np.stack(features_list, axis=0)


def extract_all_features_with_tta(
    df: pd.DataFrame,
    image_dir: Path | str,
    backbone: nn.Module,
    device: torch.device,
    img_size: int = 960,
    image_col: str = "image_path",
    tta_transforms: list[tuple[str, A.Compose]] | None = None,
) -> np.ndarray:
    """Extract features from all images with TTA (Test Time Augmentation).

    For training, we typically don't use TTA, but this is provided for
    consistency with inference.

    Args:
        df: DataFrame with image_path column.
        image_dir: Base directory containing images.
        backbone: DINOv3 backbone.
        device: Device for inference.
        img_size: Target image size.
        image_col: Column name for image path.
        tta_transforms: List of (name, transform) for TTA. If None, no TTA.

    Returns:
        Feature matrix [N, 2562] (averaged over TTA if provided)
    """
    if tta_transforms is None:
        return extract_all_features(df, image_dir, backbone, device, img_size, image_col)

    image_dir = Path(image_dir)
    features_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features (TTA)"):
        image_path = image_dir / row[image_col]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate coverage once (before resize)
        coverage_raw, coverage_log = calculate_coverage(image)

        # Resize to target size
        image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

        tta_features = []
        for _, tta_transform in tta_transforms:
            # Apply TTA transform
            aug_image = tta_transform(image=image_resized)["image"]

            # Preprocess and extract features
            img_tensor = preprocess_image(aug_image).to(device)

            with torch.no_grad():
                cls_token, patch_tokens = backbone(img_tensor)

            cls_np = cls_token.cpu().numpy()[0]
            patch_mean = patch_tokens.cpu().numpy()[0].mean(axis=0)

            feat = np.concatenate([cls_np, patch_mean, [coverage_raw, coverage_log]])
            tta_features.append(feat)

        # Average TTA features
        features_list.append(np.mean(tta_features, axis=0))

    return np.stack(features_list, axis=0)
