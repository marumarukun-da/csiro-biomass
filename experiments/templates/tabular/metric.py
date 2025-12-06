"""Custom metric functions for evaluation."""

import numpy as np
from sklearn.metrics import mean_squared_error


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate evaluation metric.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Metric score (lower is better for RMSE)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
