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

# State to index mapping for auxiliary classification task
STATE_TO_IDX = {"Tas": 0, "NSW": 1, "WA": 2, "Vic": 3}
NUM_STATES = len(STATE_TO_IDX)


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
