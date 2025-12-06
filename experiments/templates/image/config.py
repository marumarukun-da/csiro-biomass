import os
from pathlib import Path

EXP_NAME = Path(__file__).parent.name


# ---------- # DIRECTORIES # ---------- #
IS_KAGGLE_ENV = os.getenv("KAGGLE_DATA_PROXY_TOKEN") is not None
KAGGLE_COMPETITION_NAME = os.getenv(
    "KAGGLE_COMPETITION_NAME", "csiro-biomass"
)

if not IS_KAGGLE_ENV:
    import rootutils

    ROOT_DIR = rootutils.setup_root(
        ".",
        indicator="pyproject.toml",
        cwd=True,
        pythonpath=True,
    )
    INPUT_DIR = ROOT_DIR / "data" / "input"
    ARTIFACT_DIR = ROOT_DIR / "data" / "output"
    OUTPUT_DIR = ARTIFACT_DIR / EXP_NAME / "1"

    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "marumarukun")
    ARTIFACTS_HANDLE = (
        f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-artifacts/other/{EXP_NAME}"
    )
    CODES_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-codes"
else:
    ROOT_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    ARTIFACT_DIR = INPUT_DIR / f"{KAGGLE_COMPETITION_NAME}-artifacts".lower() / "other"
    OUTPUT_DIR = ROOT_DIR

COMP_DATASET_DIR = INPUT_DIR / KAGGLE_COMPETITION_NAME

for d in [INPUT_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True, parents=True)

ARTIFACT_EXP_DIR = lambda exp_name: ARTIFACT_DIR / exp_name / "1"  # noqa


# ---------- # IMAGE CONFIG # ---------- #
class CFG:
    # General
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "label"  # Update with your target column

    # Paths
    DATA_PATH = COMP_DATASET_DIR
    OUTPUT_PATH = OUTPUT_DIR
    MODEL_PATH = OUTPUT_DIR / "models"

    # Model
    MODEL_NAME = "tf_efficientnet_b0_ns"  # timm model name
    PRETRAINED = True
    NUM_CLASSES = 2  # Update with number of classes

    # Training
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    SCHEDULER = "cosine"  # cosine, step, plateau

    # Image
    IMG_SIZE = 224

    # Hardware
    DEVICE = "cuda"  # cuda or cpu
    NUM_WORKERS = 4
    USE_AMP = True  # Automatic Mixed Precision

    # Inference
    TTA = False  # Test Time Augmentation
