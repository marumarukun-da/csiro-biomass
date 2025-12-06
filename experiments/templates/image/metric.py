"""Custom metric functions for evaluation."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Calculate F1 score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')

    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average)


def auc(y_true: np.ndarray, y_prob: np.ndarray, multi_class: str = "ovr") -> float:
    """Calculate ROC AUC score.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        multi_class: Multi-class strategy ('ovr' or 'ovo')

    Returns:
        ROC AUC score
    """
    return roc_auc_score(y_true, y_prob, multi_class=multi_class)
