import numpy as np


def score(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)を計算する関数

    Parameters:
    -----------
    y_true : array-like
        実際の目的変数の値
    y_pred : array-like
        予測値

    Returns:
    --------
    float
        RMSE値
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
