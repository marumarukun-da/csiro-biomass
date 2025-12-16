"""Dataset and DataLoader utilities for biomass prediction."""

from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# Target columns we predict
TARGET_COLS_PRED = ["Dry_Total_g", "GDM_g", "Dry_Green_g"]


def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Long format CSV to Wide format (1 row per image).

    Long format (train.csv):
        sample_id, image_path, ..., target_name, target

    Wide format:
        image_id, image_path, ..., Dry_Total_g, GDM_g, Dry_Green_g, ...

    Args:
        df: DataFrame in Long format

    Returns:
        DataFrame in Wide format
    """
    # Extract image_id from sample_id (remove target suffix)
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    # test.csv: target列がない場合
    if "target" not in df.columns:
        df_wide = df[["image_id", "image_path"]].drop_duplicates().reset_index(drop=True)
        return df_wide

    # train.csv: target列がある場合
    meta_cols = [c for c in df.columns if c not in ["sample_id", "target_name", "target", "image_id"]]

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=["image_id"] + meta_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    # Flatten column names
    df_wide.columns.name = None

    return df_wide


class DualInputBiomassDataset(Dataset):
    """Dataset for dual-input (left-right split) biomass prediction.

    Processing flow:
    1. Load original image (2000 x 1000)
    2. Apply pre_split_transform to entire image (flip, color augmentation)
    3. Split into left and right halves (1000 x 1000 each)
    4. Apply post_split_transform to each half (resize, normalize, to_tensor)

    This ensures physical consistency - the same augmentation is applied
    to both halves since they come from the same original image.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path | str,
        pre_split_transform: A.Compose | None = None,
        post_split_transform: A.Compose | None = None,
        target_cols: list[str] = TARGET_COLS_PRED,
        image_col: str = "image_path",
        is_train: bool = True,
    ):
        """Initialize DualInputBiomassDataset.

        Args:
            df: DataFrame with image paths and targets (Wide format)
            image_dir: Base directory for images
            pre_split_transform: Transforms applied to entire image before splitting
                                 (flip, color augmentation - NO resize)
            post_split_transform: Transforms applied to each half after splitting
                                  (resize, normalize, to_tensor)
            target_cols: List of target column names to predict
            image_col: Column name for image path
            is_train: Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.pre_split_transform = pre_split_transform
        self.post_split_transform = post_split_transform
        self.target_cols = target_cols
        self.image_col = image_col
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample with left and right images.

        Returns:
            dict with keys:
            - image_left: Tensor [C, H, W]
            - image_right: Tensor [C, H, W]
            - targets: Tensor [num_targets] (if is_train)
            - image_id: str
            - image_path: str
        """
        row = self.df.iloc[idx]

        # Load original image [H=1000, W=2000, C=3]
        image_path = self.image_dir / row[self.image_col]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Apply pre-split transform to entire image (flip, color augmentation)
        if self.pre_split_transform is not None:
            augmented = self.pre_split_transform(image=image)
            image = augmented["image"]

        # 2. Split into left and right halves
        w = image.shape[1]
        mid = w // 2
        image_left = image[:, :mid, :]  # [H, W//2, C]
        image_right = image[:, mid:, :]  # [H, W//2, C]

        # 3. Apply post-split transform to each half (resize, normalize, to_tensor)
        if self.post_split_transform is not None:
            image_left = self.post_split_transform(image=image_left)["image"]
            image_right = self.post_split_transform(image=image_right)["image"]

        # Build result
        result = {
            "image_left": image_left,
            "image_right": image_right,
            "image_path": str(row[self.image_col]),
        }

        # Add image_id if available
        if "image_id" in row:
            result["image_id"] = row["image_id"]

        # Add targets if training
        if self.is_train and all(col in row for col in self.target_cols):
            targets = [row[col] for col in self.target_cols]
            result["targets"] = torch.tensor(targets, dtype=torch.float32)

        return result


def build_pre_split_transform(
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    brightness_contrast_p: float = 0.25,
    coarse_dropout_p: float = 0.2,
    coarse_dropout_num_holes_range: tuple[int, int] = (10, 20),
    coarse_dropout_hole_height_range: tuple[int, int] = (20, 100),
    coarse_dropout_hole_width_range: tuple[int, int] = (20, 100),
    gaussian_blur_p: float = 0.3,
    gaussian_blur_limit: tuple[int, int] = (3, 7),
    hue_shift_limit: int = 10,
    sat_shift_limit: int = 20,
    val_shift_limit: int = 20,
    hue_saturation_value_p: float = 0.5,
    gamma_limit: tuple[int, int] = (60, 140),
    random_gamma_p: float = 0.5,
) -> A.Compose:
    """Build pre-split augmentation (applied to entire image before splitting).

    This includes:
    - Flip augmentations (applied consistently to both halves)
    - Color augmentations (applied consistently to both halves)
    - CoarseDropout (random rectangular regions dropped)
    - GaussianBlur (random blur effect)
    - HueSaturationValue (color variation for robustness)
    - RandomGamma (exposure variation simulation)
    - NO resize (resize happens after split)

    Args:
        horizontal_flip_p: Probability of horizontal flip
        vertical_flip_p: Probability of vertical flip
        brightness_limit: Brightness adjustment limit
        contrast_limit: Contrast adjustment limit
        brightness_contrast_p: Probability of brightness/contrast adjustment
        coarse_dropout_p: Probability of coarse dropout
        coarse_dropout_num_holes_range: Range of number of holes (min, max)
        coarse_dropout_hole_height_range: Range of hole height (min, max)
        coarse_dropout_hole_width_range: Range of hole width (min, max)
        gaussian_blur_p: Probability of gaussian blur
        gaussian_blur_limit: Blur kernel size range (min, max)
        hue_shift_limit: Hue shift limit for HueSaturationValue
        sat_shift_limit: Saturation shift limit for HueSaturationValue
        val_shift_limit: Value shift limit for HueSaturationValue
        hue_saturation_value_p: Probability of HueSaturationValue
        gamma_limit: Gamma limit range (min, max) for RandomGamma
        random_gamma_p: Probability of RandomGamma

    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.HorizontalFlip(p=horizontal_flip_p),
        A.VerticalFlip(p=vertical_flip_p),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=brightness_contrast_p,
        ),
        A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=hue_saturation_value_p,
        ),
        A.RandomGamma(
            gamma_limit=gamma_limit,
            p=random_gamma_p,
        ),
        A.CoarseDropout(
            num_holes_range=coarse_dropout_num_holes_range,
            hole_height_range=coarse_dropout_hole_height_range,
            hole_width_range=coarse_dropout_hole_width_range,
            p=coarse_dropout_p,
        ),
        A.GaussianBlur(
            blur_limit=gaussian_blur_limit,
            p=gaussian_blur_p,
        ),
    ]

    return A.Compose(transforms)


def build_post_split_transform(
    img_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
    is_train: bool = False,
    random_rotate90_p: float = 0.3,
) -> A.Compose:
    """Build post-split transform (applied to each half after splitting).

    This includes:
    - RandomRotate90 (training only, applied independently to each half)
    - Resize to target size
    - Normalization
    - Convert to tensor

    Args:
        img_size: Target image size
        normalize_mean: Normalization mean (default: ImageNet)
        normalize_std: Normalization std (default: ImageNet)
        is_train: Whether this is for training (enables RandomRotate90)
        random_rotate90_p: Probability of RandomRotate90 (training only)

    Returns:
        Albumentations Compose object
    """
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    transforms = []

    # RandomRotate90 only for training (applied before resize)
    if is_train:
        transforms.append(A.RandomRotate90(p=random_rotate90_p))

    transforms.extend(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=normalize_mean, std=normalize_std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def build_pre_split_transform_valid() -> A.Compose:
    """Build pre-split transform for validation (no augmentation).

    Returns:
        Albumentations Compose object (identity transform)
    """
    return A.Compose([])


def build_tta_pre_split_transforms() -> list[tuple[str, A.Compose]]:
    """Build TTA transforms applied to entire image before splitting.

    Returns list of (name, transform) tuples:
    - original: No transform
    - hflip: Horizontal flip (swaps left and right!)
    - vflip: Vertical flip

    Returns:
        List of (name, Albumentations Compose) tuples
    """
    return [
        ("original", A.Compose([])),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)])),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)])),
    ]


def create_dual_input_dataloader(
    df: pd.DataFrame,
    image_dir: Path | str,
    pre_split_transform: A.Compose | None,
    post_split_transform: A.Compose,
    batch_size: int,
    target_cols: list[str] = TARGET_COLS_PRED,
    image_col: str = "image_path",
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for dual-input training or validation.

    Args:
        df: DataFrame with image paths and targets
        image_dir: Base directory for images
        pre_split_transform: Transforms applied before splitting
        post_split_transform: Transforms applied after splitting
        batch_size: Batch size
        target_cols: Target column names
        image_col: Image path column name
        is_train: Whether training data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = DualInputBiomassDataset(
        df=df,
        image_dir=image_dir,
        pre_split_transform=pre_split_transform,
        post_split_transform=post_split_transform,
        target_cols=target_cols,
        image_col=image_col,
        is_train=is_train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
