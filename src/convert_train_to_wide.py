"""Convert train.csv from Long format to Wide format."""

from pathlib import Path

import pandas as pd


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
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    if "target" not in df.columns:
        df_wide = df[["image_id", "image_path"]].drop_duplicates().reset_index(drop=True)
        return df_wide

    meta_cols = [c for c in df.columns if c not in ["sample_id", "target_name", "target", "image_id"]]

    df_wide = df.pivot_table(
        index=["image_id"] + meta_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    df_wide.columns.name = None

    return df_wide


def main():
    input_path = Path("data/input/csiro-biomass/train.csv")
    output_path = Path("data/input/csiro-biomass/train_wide.csv")

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    print("Converting to wide format...")
    df_wide = convert_long_to_wide(df)
    print(f"  Shape: {df_wide.shape}")

    print(f"Saving to {output_path}...")
    df_wide.to_csv(output_path, index=False)
    print("Done!")

    print("\nColumns:")
    print(df_wide.columns.tolist())
    print("\nFirst 3 rows:")
    print(df_wide.head(3))


if __name__ == "__main__":
    main()
