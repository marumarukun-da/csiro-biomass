"""Head models for multi-target regression.

This module provides various regression heads that can be used as the final
layer after feature extraction (e.g., from DINOv3 backbone).
"""

import pickle
from pathlib import Path
from typing import Any

from .base import MultiTargetHead
from .catboost_head import MultiTargetCatBoostHead
from .extratrees import MultiTargetExtraTreesHead
from .gbdt import MultiTargetGBDTHead, MultiTargetHistGBDTHead
from .gpr import MultiTargetGPRHead
from .kernel_ridge import MultiTargetKernelRidgeHead
from .lightgbm_head import MultiTargetLightGBMHead
from .ridge import (
    MultiTargetBayesianRidgeHead,
    MultiTargetElasticNetHead,
    MultiTargetLassoHead,
    MultiTargetRidgeHead,
)
from .svr import MultiTargetSVRHead
from .xgboost_head import MultiTargetXGBoostHead

# Registry of available head types
HEAD_REGISTRY: dict[str, type[MultiTargetHead]] = {
    # Linear models
    "svr": MultiTargetSVRHead,
    "ridge": MultiTargetRidgeHead,
    "lasso": MultiTargetLassoHead,
    "elasticnet": MultiTargetElasticNetHead,
    "bayesian_ridge": MultiTargetBayesianRidgeHead,
    "kernel_ridge": MultiTargetKernelRidgeHead,
    "gpr": MultiTargetGPRHead,
    # GBDT models
    "gbdt": MultiTargetGBDTHead,
    "histgbdt": MultiTargetHistGBDTHead,
    "xgboost": MultiTargetXGBoostHead,
    "lightgbm": MultiTargetLightGBMHead,
    "catboost": MultiTargetCatBoostHead,
    # Tree ensemble models
    "extratrees": MultiTargetExtraTreesHead,
}

__all__ = [
    # Base class
    "MultiTargetHead",
    # Linear models
    "MultiTargetSVRHead",
    "MultiTargetRidgeHead",
    "MultiTargetLassoHead",
    "MultiTargetElasticNetHead",
    "MultiTargetBayesianRidgeHead",
    "MultiTargetKernelRidgeHead",
    "MultiTargetGPRHead",
    # GBDT models
    "MultiTargetGBDTHead",
    "MultiTargetHistGBDTHead",
    "MultiTargetXGBoostHead",
    "MultiTargetLightGBMHead",
    "MultiTargetCatBoostHead",
    # Tree ensemble models
    "MultiTargetExtraTreesHead",
    # Registry and factory functions
    "HEAD_REGISTRY",
    "create_head",
    "load_head",
    "get_available_heads",
]


def create_head(head_type: str, **kwargs: Any) -> MultiTargetHead:
    """Factory function to create a head model.

    Args:
        head_type: Type of head. Available types:
            - Linear: 'svr', 'ridge', 'lasso', 'elasticnet', 'bayesian_ridge',
                      'kernel_ridge', 'gpr'
            - GBDT: 'gbdt', 'histgbdt', 'xgboost', 'lightgbm', 'catboost'
            - Tree ensemble: 'extratrees'
        **kwargs: Additional arguments passed to the head constructor.

    Returns:
        Instantiated head model.

    Raises:
        ValueError: If head_type is not recognized.
    """
    if head_type not in HEAD_REGISTRY:
        available = ", ".join(HEAD_REGISTRY.keys())
        raise ValueError(f"Unknown head type: {head_type}. Available: {available}")

    head_class = HEAD_REGISTRY[head_type]
    return head_class(**kwargs)


def load_head(path: str | Path) -> MultiTargetHead:
    """Load a head model from file.

    Automatically determines the head type from the saved file.

    Args:
        path: Path to saved model.

    Returns:
        Loaded head model.

    Raises:
        ValueError: If head type in file is not recognized.
    """
    with open(path, "rb") as f:
        save_dict = pickle.load(f)

    head_type = save_dict.get("head_type")
    if head_type is None:
        # Backward compatibility: assume SVR if no head_type
        head_type = "svr"

    if head_type not in HEAD_REGISTRY:
        available = ", ".join(HEAD_REGISTRY.keys())
        raise ValueError(f"Unknown head type in file: {head_type}. Available: {available}")

    head_class = HEAD_REGISTRY[head_type]
    return head_class.load(path)


def get_available_heads() -> list[str]:
    """Get list of available head types.

    Returns:
        List of head type names.
    """
    return list(HEAD_REGISTRY.keys())
