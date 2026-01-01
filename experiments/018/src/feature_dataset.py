"""Dataset for precomputed DINOv3 features."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# State mapping (same as existing)
STATE_TO_IDX = {"Tas": 0, "NSW": 1, "WA": 2, "Vic": 3}


class PrecomputedFeatureDataset(Dataset):
    """Dataset that loads precomputed CLS and patch features from .npz files.

    Each .npz file contains features for multiple augmentation patterns:
        - cls_0, cls_1, ..., cls_19: CLS tokens for each augmentation
        - patches_0, patches_1, ..., patches_19: Patch tokens for each augmentation

    During training, randomly selects one augmentation pattern per sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: Path | str,
        num_aug_patterns: int = 20,
        target_cols: list[str] | None = None,
        aux_target_cols: list[str] | None = None,
        state_col: str = "State",
        height_col: str = "Height_Ave_cm",
        is_train: bool = True,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with image metadata and targets (Wide format).
            feature_dir: Directory containing .npz feature files.
            num_aug_patterns: Number of augmentation patterns per image.
            target_cols: Main target columns [Dry_Total_g, GDM_g, Dry_Green_g].
            aux_target_cols: Auxiliary target columns [Dry_Dead_g, Dry_Clover_g].
            state_col: Column name for state classification.
            height_col: Column name for height regression.
            is_train: Whether this is training data.
        """
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.num_aug_patterns = num_aug_patterns
        self.target_cols = target_cols or ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
        self.aux_target_cols = aux_target_cols or ["Dry_Dead_g", "Dry_Clover_g"]
        self.state_col = state_col
        self.height_col = height_col
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load features from .npz
        feature_path = self.feature_dir / f"{image_id}.npz"
        data = np.load(feature_path)

        # Select augmentation pattern
        if self.is_train:
            aug_idx = random.randint(0, self.num_aug_patterns - 1)
        else:
            aug_idx = 0  # Use original for validation

        cls_token = torch.tensor(data[f"cls_{aug_idx}"], dtype=torch.float32)
        patch_tokens = torch.tensor(data[f"patches_{aug_idx}"], dtype=torch.float32)

        result = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "image_id": image_id,
        }

        # Add targets
        if self.is_train:
            # Main targets
            main_targets = [row[col] for col in self.target_cols]
            result["main_targets"] = torch.tensor(main_targets, dtype=torch.float32)

            # Auxiliary targets (Dead, Clover)
            aux_targets = [row[col] for col in self.aux_target_cols]
            result["aux_targets"] = torch.tensor(aux_targets, dtype=torch.float32)

            # State label
            if self.state_col in row:
                result["state_label"] = STATE_TO_IDX[row[self.state_col]]

            # Height value
            if self.height_col in row:
                result["height_value"] = float(row[self.height_col])

        return result


class InferenceFeatureDataset(Dataset):
    """Dataset for inference with precomputed features (TTA support).

    Loads multiple augmentation patterns for TTA ensemble.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: Path | str,
        tta_indices: list[int] | None = None,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with image metadata.
            feature_dir: Directory containing .npz feature files.
            tta_indices: Indices of TTA augmentations to use (default: [0,1,2,3]).
        """
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.tta_indices = tta_indices or [0, 1, 2, 3]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        feature_path = self.feature_dir / f"{image_id}.npz"
        data = np.load(feature_path)

        # Load all TTA features
        cls_tokens = []
        patch_tokens_list = []
        for tta_idx in self.tta_indices:
            cls_tokens.append(data[f"cls_{tta_idx}"])
            patch_tokens_list.append(data[f"patches_{tta_idx}"])

        return {
            "cls_tokens": torch.tensor(np.stack(cls_tokens), dtype=torch.float32),
            "patch_tokens": torch.tensor(np.stack(patch_tokens_list), dtype=torch.float32),
            "image_id": image_id,
        }


def create_feature_dataloader(
    df: pd.DataFrame,
    feature_dir: Path | str,
    batch_size: int,
    num_aug_patterns: int = 20,
    target_cols: list[str] | None = None,
    aux_target_cols: list[str] | None = None,
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for precomputed features.

    Args:
        df: DataFrame with image metadata and targets.
        feature_dir: Directory containing .npz feature files.
        batch_size: Batch size.
        num_aug_patterns: Number of augmentation patterns per image.
        target_cols: Main target columns.
        aux_target_cols: Auxiliary target columns.
        is_train: Whether training data.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    dataset = PrecomputedFeatureDataset(
        df=df,
        feature_dir=feature_dir,
        num_aug_patterns=num_aug_patterns,
        target_cols=target_cols,
        aux_target_cols=aux_target_cols,
        is_train=is_train,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
