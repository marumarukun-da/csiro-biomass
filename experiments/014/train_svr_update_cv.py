"""Training script for SVR model using precomputed DINOv3 features.

CV strategy: StratifiedKFold based on sampling_month + State.
"""

# isort: off
# config must be imported first to setup paths via rootutils
import config  # noqa: F401
# isort: on

import argparse
import json
import logging
import warnings
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.data import convert_long_to_wide
from src.metric import TARGET_COLS_PRED, weighted_r2_score_full
from src.seed import seed_everything
from src.svr_model import MultiTargetSVR, load_all_features

warnings.filterwarnings("ignore")


def configure_logger(log_path: Path) -> logging.Logger:
    """Configure logger for a run."""
    logger = logging.getLogger(f"train_svr.{log_path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    stratify_col: str = "stratify_group",
    random_state: int = 42,
) -> pd.DataFrame:
    """Create fold column using StratifiedKFold.

    Args:
        df: DataFrame with data.
        n_folds: Number of folds.
        stratify_col: Column name to use for stratification.
        random_state: Random state for reproducibility.

    Returns:
        DataFrame with fold column added.
    """
    df = df.copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(df, df[stratify_col])):
        df.loc[val_idx, "fold"] = fold

    return df


def grid_search_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict,
    logger: logging.Logger,
) -> tuple[MultiTargetSVR, dict, float]:
    """Grid search for best SVR parameters.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        param_grid: Dictionary with parameter lists to search.
        logger: Logger instance.

    Returns:
        Tuple of (best_model, best_params, best_score).
    """
    best_score = -float("inf")
    best_params = None
    best_model = None

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)

    logger.info(f"Grid search: {total_combinations} combinations")

    for combo in tqdm(product(*param_values), total=total_combinations, desc="GridSearch"):
        params = dict(zip(param_names, combo))

        # Train SVR with these parameters
        svr = MultiTargetSVR(
            kernel=params.get("kernel", "rbf"),
            C=params["C"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
        )
        svr.fit(X_train, y_train)

        # Evaluate on validation set
        preds = svr.predict(X_val)
        score = weighted_r2_score_full(y_val, preds)

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = svr

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best score: {best_score:.5f}")

    return best_model, best_params, best_score


def train_svr_cv(
    cfg: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Train SVR with cross-validation.

    Args:
        cfg: Configuration dictionary.
        output_dir: Output directory for models and logs.
        logger: Logger instance.

    Returns:
        Dictionary with results.
    """
    # Extract config
    dataset_cfg = cfg.get("dataset", {})
    svr_cfg = cfg.get("svr", {})
    trainer_cfg = cfg.get("trainer", {})
    experiment_cfg = cfg.get("experiment", {})

    seed = experiment_cfg.get("seed", 42)
    seed_everything(seed)

    feature_dir = Path(dataset_cfg.get("feature_dir"))
    aug_idx = dataset_cfg.get("aug_idx", 0)
    target_cols = dataset_cfg.get("target_cols", TARGET_COLS_PRED)
    n_folds = trainer_cfg.get("n_folds", 5)

    # Build parameter grid
    param_grid = {
        "C": svr_cfg.get("C", [1.0]),
        "gamma": svr_cfg.get("gamma", ["scale"]),
        "epsilon": svr_cfg.get("epsilon", [0.1]),
    }

    # Ensure lists
    for key in param_grid:
        if not isinstance(param_grid[key], list):
            param_grid[key] = [param_grid[key]]

    logger.info(f"Feature directory: {feature_dir}")
    logger.info(f"Augmentation index: {aug_idx}")
    logger.info(f"Target columns: {target_cols}")
    logger.info(f"N folds: {n_folds}")
    logger.info(f"Parameter grid: {param_grid}")

    # Load data
    train_csv = config.get_train_csv_path()
    logger.info(f"Loading data from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    # Create stratification column for StratifiedKFold
    # Extract month from Sampling_Date
    train_df["sampling_month"] = pd.to_datetime(train_df["Sampling_Date"]).dt.month
    train_df["stratify_group"] = train_df["sampling_month"].astype(str) + "_" + train_df["State"]

    # Convert to wide format
    train_df = convert_long_to_wide(train_df)
    train_df = train_df.sort_values(by=["image_id"]).reset_index(drop=True)
    logger.info(f"Data loaded: {len(train_df)} images")

    # Log stratification group distribution
    stratify_col = "stratify_group"
    group_counts = train_df[stratify_col].value_counts().sort_index()
    logger.info(f"Stratification groups ({stratify_col}):")
    for group, count in group_counts.items():
        logger.info(f"  {group}: {count}")

    # Create folds
    train_df = create_folds(train_df, n_folds=n_folds, stratify_col=stratify_col, random_state=seed)
    logger.info(f"Created {n_folds} folds using StratifiedKFold (stratify_col={stratify_col}, seed={seed})")

    # Load all features
    logger.info("Loading features...")
    X_all = load_all_features(train_df, feature_dir, aug_idx=aug_idx)
    logger.info(f"Features shape: {X_all.shape}")

    # Extract targets
    y_all = train_df[target_cols].values
    logger.info(f"Targets shape: {y_all.shape}")

    # Create output directories
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation
    fold_results = []
    oof_preds = np.zeros_like(y_all)

    for fold in range(n_folds):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"FOLD {fold}")
        logger.info("=" * 50)

        # Split data
        train_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        X_train, X_val = X_all[train_mask], X_all[val_mask]
        y_train, y_val = y_all[train_mask], y_all[val_mask]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Grid search
        best_model, best_params, best_score = grid_search_svr(X_train, y_train, X_val, y_val, param_grid, logger)

        # Save model
        model_path = weights_dir / f"svr_fold{fold}.pkl"
        best_model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Store OOF predictions
        val_preds = best_model.predict(X_val)
        oof_preds[val_mask] = val_preds

        fold_results.append(
            {
                "fold": fold,
                "best_params": best_params,
                "val_r2": best_score,
            }
        )

    # Calculate overall CV score
    cv_score = weighted_r2_score_full(y_all, oof_preds)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Overall CV Score (Weighted R²): {cv_score:.5f}")
    logger.info("=" * 50)

    # Per-fold summary
    for r in fold_results:
        logger.info(f"Fold {r['fold']}: R²={r['val_r2']:.5f}, params={r['best_params']}")

    avg_r2 = np.mean([r["val_r2"] for r in fold_results])
    logger.info(f"Average fold R²: {avg_r2:.5f}")

    # Save OOF predictions
    oof_df = train_df[["image_id"]].copy()
    for i, col in enumerate(target_cols):
        oof_df[f"{col}_pred"] = oof_preds[:, i]
        oof_df[f"{col}_true"] = y_all[:, i]
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    logger.info(f"OOF predictions saved to {output_dir / 'oof_predictions.csv'}")

    # Save fold results
    results_path = output_dir / "fold_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "cv_score": cv_score,
                "avg_fold_r2": avg_r2,
                "fold_results": fold_results,
            },
            f,
            indent=2,
            default=str,
        )

    return {
        "cv_score": cv_score,
        "avg_fold_r2": avg_r2,
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SVR model with DINOv3 features (StratifiedKFold CV)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = config.EXP_DIR / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Create output directory
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.get("experiment", {}).get("name", "exp014_svr_stratified")
    output_dir = config.OUTPUT_DIR / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(output_dir / "config.yaml"))

    # Setup logger
    logger = configure_logger(output_dir / "train.log")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")

    # Train
    results = train_svr_cv(cfg, output_dir, logger)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Output directory: {output_dir}")
    print(f"CV Score (Weighted R²): {results['cv_score']:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
