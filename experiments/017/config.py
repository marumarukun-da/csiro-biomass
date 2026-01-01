"""Configuration for experiment 001.

This module handles path configuration for both local and Kaggle environments,
maintaining compatibility with the existing Kaggle submission workflow.
"""

import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent.resolve()
# EXP_NAME should always be "001" regardless of where the code is located
# (e.g., on Kaggle, code may be in /kaggle/working/ but EXP_NAME should still be "001")
EXP_NAME = "017"


# ---------- # DIRECTORIES # ---------- #
IS_KAGGLE_ENV = os.getenv("KAGGLE_DATA_PROXY_TOKEN") is not None
KAGGLE_COMPETITION_NAME = os.getenv("KAGGLE_COMPETITION_NAME", "csiro-biomass")

if not IS_KAGGLE_ENV:
    import rootutils

    ROOT_DIR = rootutils.setup_root(
        ".",
        indicator="pyproject.toml",
        cwd=True,
        pythonpath=True,
    )

    # Add experiment directory to sys.path AFTER rootutils
    # This ensures experiments/001/src takes priority over /workspace/src
    sys.path.insert(0, str(EXP_DIR))

    INPUT_DIR = ROOT_DIR / "data" / "input"
    # ARTIFACT_DIR / EXP_NAME / 1 でのアクセスを想定
    ARTIFACT_DIR = ROOT_DIR / "data" / "output"
    # 当該 code の生成物の出力先. kaggle code とパスを合わせるために 1 を付与
    OUTPUT_DIR = ARTIFACT_DIR / EXP_NAME / "1"

    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "marumarukun")
    ARTIFACTS_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-artifacts/other/{EXP_NAME}"
    CODES_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-codes-{EXP_NAME}"
else:
    ROOT_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    # 当該 code 以外の生成物が格納されている場所 (Model として使用できる)
    # ARTIFACT_DIR / EXP_NAME / 1 でアクセス可能
    ARTIFACT_DIR = INPUT_DIR / f"{KAGGLE_COMPETITION_NAME}-artifacts".lower() / "other"
    OUTPUT_DIR = ROOT_DIR  # 当該 code の生成物の出力先

COMP_DATASET_DIR = INPUT_DIR / KAGGLE_COMPETITION_NAME

# Create output directory (INPUT_DIR is read-only on Kaggle)
if not IS_KAGGLE_ENV:
    for d in [INPUT_DIR, OUTPUT_DIR]:
        d.mkdir(exist_ok=True, parents=True)
else:
    # On Kaggle, only create OUTPUT_DIR (working directory)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 対象の exp の artifact が格納されている場所を返す
ARTIFACT_EXP_DIR = lambda exp_name: ARTIFACT_DIR / exp_name / "1"  # noqa: E731


# ---------- # HELPER FUNCTIONS # ---------- #
def get_train_csv_path() -> Path:
    """Get path to training CSV."""
    return COMP_DATASET_DIR / "train.csv"


def get_test_csv_path() -> Path:
    """Get path to test CSV."""
    return COMP_DATASET_DIR / "test.csv"


def get_sample_submission_path() -> Path:
    """Get path to sample submission CSV."""
    return COMP_DATASET_DIR / "sample_submission.csv"


def get_image_dir() -> Path:
    """Get path to image directory."""
    return COMP_DATASET_DIR
