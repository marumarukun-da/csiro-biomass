"""Training script for regression heads using precomputed DINOv3 features.

Supports multiple head types: SVR, Ridge, Lasso, ElasticNet, BayesianRidge,
KernelRidge, and GPR.
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
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.data import convert_long_to_wide
from src.feature_engine import FeatureEngine
from src.heads import MultiTargetHead, create_head, get_available_heads
from src.metric import TARGET_COLS_PRED, weighted_r2_score_full
from src.seed import seed_everything
from src.svr_model import load_all_features

warnings.filterwarnings("ignore")


def configure_logger(log_path: Path) -> logging.Logger:
    """Configure logger for a run."""
    logger = logging.getLogger(f"train.{log_path.stem}")
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
    group_col: str = "site",
    stratify_col: str = "State",
    seed: int = 38,
) -> pd.DataFrame:
    """Create fold column using StratifiedGroupKFold."""
    df = df.copy()
    df["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df[stratify_col], groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold

    return df


def grid_search_head(
    head_type: str,
    head_base_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict,
    logger: logging.Logger,
) -> tuple[MultiTargetHead, dict, float]:
    """Grid search for best head parameters.

    Args:
        head_type: Type of head model.
        head_base_params: Base parameters for head (not searched).
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

    if not param_names:
        # No grid search, just train with base params
        head = create_head(head_type, **head_base_params)
        head.fit(X_train, y_train)
        preds = head.predict(X_val)
        score = weighted_r2_score_full(y_val, preds)
        return head, head_base_params.copy(), score

    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)

    logger.info(f"Grid search: {total_combinations} combinations")

    for combo in tqdm(product(*param_values), total=total_combinations, desc="GridSearch"):
        params = dict(zip(param_names, combo))

        # Merge with base params
        full_params = {**head_base_params, **params}

        # Train head with these parameters
        head = create_head(head_type, **full_params)
        head.fit(X_train, y_train)

        # Evaluate on validation set
        preds = head.predict(X_val)
        score = weighted_r2_score_full(y_val, preds)

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = head

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best score: {best_score:.5f}")

    return best_model, best_params, best_score


def build_param_grid(head_cfg: dict) -> tuple[dict, dict]:
    """Build parameter grid and base params from head config.

    Args:
        head_cfg: Head configuration dictionary.

    Returns:
        Tuple of (base_params, param_grid).
        base_params: Parameters that are not lists (fixed).
        param_grid: Parameters that are lists (to be searched).
    """
    base_params = {}
    param_grid = {}

    for key, value in head_cfg.items():
        if key == "type":
            continue
        if isinstance(value, list):
            param_grid[key] = value
        else:
            base_params[key] = value

    return base_params, param_grid


def train_cv(
    cfg: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Train head model with cross-validation.

    Args:
        cfg: Configuration dictionary.
        output_dir: Output directory for models and logs.
        logger: Logger instance.

    Returns:
        Dictionary with results.
    """
    # Extract config
    dataset_cfg = cfg.get("dataset", {})
    head_cfg = cfg.get("head", {})
    trainer_cfg = cfg.get("trainer", {})
    experiment_cfg = cfg.get("experiment", {})
    preprocessing_cfg = cfg.get("preprocessing", {})

    seed = experiment_cfg.get("seed", 42)
    seed_everything(seed)

    feature_dir = Path(dataset_cfg.get("feature_dir"))
    aug_idx = dataset_cfg.get("aug_idx", 0)
    target_cols = dataset_cfg.get("target_cols", TARGET_COLS_PRED)
    n_folds = trainer_cfg.get("n_folds", 5)
    group_col = trainer_cfg.get("group_col", "site")

    # Get head type and parameters
    head_type = head_cfg.get("type", "svr")
    base_params, param_grid = build_param_grid(head_cfg)

    # Get preprocessing parameters (PCA/PLS)
    pca_cfg = preprocessing_cfg.get("pca", {})
    pls_cfg = preprocessing_cfg.get("pls", {})
    n_pca = pca_cfg.get("n_components", None)
    n_pls = pls_cfg.get("n_components", None)
    use_preprocessing = n_pca is not None or n_pls is not None

    logger.info(f"Head type: {head_type}")
    logger.info(f"Available heads: {get_available_heads()}")
    logger.info(f"Feature directory: {feature_dir}")
    logger.info(f"Augmentation index: {aug_idx}")
    logger.info(f"Target columns: {target_cols}")
    logger.info(f"N folds: {n_folds}")
    logger.info(f"Base params: {base_params}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Preprocessing: PCA n_components={n_pca}, PLS n_components={n_pls}")

    # Load data
    train_csv = config.get_train_csv_path()
    logger.info(f"Loading data from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    # Create site column for GroupKFold
    train_df["site"] = train_df["State"] + "_" + train_df["Sampling_Date"]

    # Convert to wide format
    train_df = convert_long_to_wide(train_df)
    train_df = train_df.sort_values(by=["image_id"]).reset_index(drop=True)
    logger.info(f"Data loaded: {len(train_df)} images")

    # Create folds
    train_df = create_folds(train_df, n_folds=n_folds, group_col=group_col)
    logger.info(f"Created {n_folds} folds using StratifiedGroupKFold (group_col={group_col}, seed=38)")

    # Load all features (with coverage)
    image_dir = config.get_image_dir()
    logger.info("Loading features with coverage...")
    X_all = load_all_features(train_df, feature_dir, image_dir=image_dir, aug_idx=aug_idx)
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

        # Apply preprocessing (PCA/PLS) if configured
        if use_preprocessing:
            feature_engine = FeatureEngine(n_pca=n_pca, n_pls=n_pls, scale=True)
            feature_engine.fit(X_train, y_train)
            X_train_transformed = feature_engine.transform(X_train)
            X_val_transformed = feature_engine.transform(X_val)
            logger.info(f"Feature engine: {feature_engine}")
            logger.info(f"Transformed features: {X_train.shape} -> {X_train_transformed.shape}")

            # Save feature engine
            engine_path = weights_dir / f"feature_engine_fold{fold}.pkl"
            feature_engine.save(engine_path)
            logger.info(f"Feature engine saved to {engine_path}")
        else:
            X_train_transformed = X_train
            X_val_transformed = X_val

        # Grid search
        best_model, best_params, best_score = grid_search_head(
            head_type, base_params, X_train_transformed, y_train, X_val_transformed, y_val, param_grid, logger
        )

        # Save model
        model_path = weights_dir / f"{head_type}_fold{fold}.pkl"
        best_model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Store OOF predictions
        val_preds = best_model.predict(X_val_transformed)
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
                "head_type": head_type,
                "cv_score": cv_score,
                "avg_fold_r2": avg_r2,
                "fold_results": fold_results,
            },
            f,
            indent=2,
            default=str,
        )

    return {
        "head_type": head_type,
        "cv_score": cv_score,
        "avg_fold_r2": avg_r2,
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train regression head with DINOv3 features")
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
    head_type = cfg.get("head", {}).get("type", "svr")
    experiment_name = cfg.get("experiment", {}).get("name", f"exp017_{head_type}")
    output_dir = config.OUTPUT_DIR / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(output_dir / "config.yaml"))

    # Setup logger
    logger = configure_logger(output_dir / "train.log")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")

    # Train
    results = train_cv(cfg, output_dir, logger)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Head type: {results['head_type']}")
    print(f"Output directory: {output_dir}")
    print(f"CV Score (Weighted R²): {results['cv_score']:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
