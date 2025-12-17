"""Evaluation metrics for biomass prediction."""

import numpy as np

# Target columns and their weights (from competition)
TARGET_COLS_ALL = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = {
    "Dry_Total_g": 0.5,
    "GDM_g": 0.2,
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
}

# Columns we actually predict (others are derived)
TARGET_COLS_PRED = ["Dry_Dead_g", "Dry_Green_g", "Dry_Clover_g"]


def derive_all_targets(preds_3: np.ndarray) -> np.ndarray:
    """Derive all 5 target values from 3 predicted values.

    Args:
        preds_3: Array of shape [N, 3] with columns [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]

    Returns:
        Array of shape [N, 5] with columns [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    Dry_Dead_g = preds_3[:, 0]
    Dry_Green_g = preds_3[:, 1]
    Dry_Clover_g = preds_3[:, 2]

    # Derive remaining targets
    # GDM_g = Dry_Green_g + Dry_Clover_g
    # Dry_Total_g = Dry_Dead_g + GDM_g = Dry_Dead_g + Dry_Green_g + Dry_Clover_g
    GDM_g = Dry_Green_g + Dry_Clover_g
    Dry_Total_g = Dry_Dead_g + GDM_g

    # Return in TARGET_COLS_ALL order
    return np.stack([Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g], axis=1)


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str] | None = None,
) -> float:
    """Calculate globally weighted R² score (competition metric).

    This metric computes R² across all (sample, target) pairs with target-specific weights.

    Args:
        y_true: Ground truth values, shape [N, num_targets]
        y_pred: Predicted values, shape [N, num_targets]
        target_cols: List of target column names (for weight lookup).
                     If None, uses TARGET_COLS_ALL.

    Returns:
        Weighted R² score
    """
    if target_cols is None:
        target_cols = TARGET_COLS_ALL

    # Get weights for each target
    weights = np.array([TARGET_WEIGHTS[col] for col in target_cols])

    # Flatten arrays: [N, T] -> [N*T]
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Create weight array: repeat weights for each sample
    # weights shape: [T] -> tile to [N*T]
    n_samples = y_true.shape[0]
    w_flat = np.tile(weights, n_samples)

    # Weighted mean of y_true
    y_bar_w = np.average(y_true_flat, weights=w_flat)

    # Weighted sum of squared residuals
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)

    # Weighted total sum of squares
    ss_tot = np.sum(w_flat * (y_true_flat - y_bar_w) ** 2)

    # Handle edge case
    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def weighted_r2_score_3targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate weighted R² for 3 predicted targets only.

    This is useful during training when we only predict 3 values.

    Args:
        y_true: Ground truth values, shape [N, 3] with [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]
        y_pred: Predicted values, shape [N, 3]

    Returns:
        Weighted R² score for 3 targets
    """
    return weighted_r2_score(y_true, y_pred, target_cols=TARGET_COLS_PRED)


def weighted_r2_score_full(
    y_true_3: np.ndarray,
    y_pred_3: np.ndarray,
) -> float:
    """Calculate full weighted R² by deriving all 5 targets.

    Args:
        y_true_3: Ground truth for 3 targets, shape [N, 3]
        y_pred_3: Predictions for 3 targets, shape [N, 3]

    Returns:
        Weighted R² score for all 5 targets
    """
    # Derive all 5 targets
    y_true_5 = derive_all_targets(y_true_3)
    y_pred_5 = derive_all_targets(y_pred_3)

    return weighted_r2_score(y_true_5, y_pred_5, target_cols=TARGET_COLS_ALL)


def r2_score_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
) -> dict[str, float]:
    """Calculate R² score for each target individually.

    Args:
        y_true: Ground truth values, shape [N, num_targets]
        y_pred: Predicted values, shape [N, num_targets]
        target_cols: List of target column names

    Returns:
        Dictionary mapping target name to R² score
    """
    results = {}
    for i, col in enumerate(target_cols):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

        if ss_tot == 0:
            results[col] = 0.0
        else:
            results[col] = 1 - ss_res / ss_tot

    return results
