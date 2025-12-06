"""Dataset and DataLoader utilities for image classification."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """Custom Dataset for image classification.

    Args:
        df: DataFrame with image paths and labels
        image_dir: Directory containing images
        transforms: Albumentations transforms
        is_train: Whether this is training data
    """

    def __init__(
        self,
        df,
        image_dir: Path,
        transforms=None,
        is_train: bool = True,
        image_col: str = "image_path",
        label_col: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.is_train = is_train
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = self.image_dir / row[self.image_col]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get label
        if self.is_train:
            label = row[self.label_col]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image


def create_dataloader(
    df,
    image_dir: Path,
    transforms,
    batch_size: int,
    num_workers: int = 4,
    is_train: bool = True,
    image_col: str = "image_path",
    label_col: str = "label",
) -> DataLoader:
    """Create DataLoader for training or validation.

    Args:
        df: DataFrame with image paths and labels
        image_dir: Directory containing images
        transforms: Albumentations transforms
        batch_size: Batch size
        num_workers: Number of workers for data loading
        is_train: Whether this is training data
        image_col: Column name for image paths
        label_col: Column name for labels

    Returns:
        DataLoader instance
    """
    dataset = ImageDataset(
        df=df,
        image_dir=image_dir,
        transforms=transforms,
        is_train=is_train,
        image_col=image_col,
        label_col=label_col,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
