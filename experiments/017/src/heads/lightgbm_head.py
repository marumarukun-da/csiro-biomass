"""LightGBM head implementation."""

from lightgbm import LGBMRegressor

from .base import MultiTargetHead


class MultiTargetLightGBMHead(MultiTargetHead):
    """Multi-target LightGBM regression head.

    LightGBM with leaf-wise tree growth and GPU support.
    """

    HEAD_TYPE: str = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = -1,
        target_names: list[str] | None = None,
        scale_features: bool = False,
    ):
        """Initialize MultiTargetLightGBMHead.

        Args:
            n_estimators: Number of boosting iterations.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum number of leaves in one tree.
            max_depth: Maximum tree depth (-1 for unlimited).
            min_child_samples: Minimum number of data needed in a leaf.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns for each tree.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            n_jobs: Number of parallel threads.
            random_state: Random seed.
            verbose: Verbosity level (-1=silent).
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _create_model(self) -> LGBMRegressor:
        """Create a new LGBMRegressor instance."""
        return LGBMRegressor(
            n_estimators=self.model_params["n_estimators"],
            learning_rate=self.model_params["learning_rate"],
            num_leaves=self.model_params["num_leaves"],
            max_depth=self.model_params["max_depth"],
            min_child_samples=self.model_params["min_child_samples"],
            subsample=self.model_params["subsample"],
            colsample_bytree=self.model_params["colsample_bytree"],
            reg_alpha=self.model_params["reg_alpha"],
            reg_lambda=self.model_params["reg_lambda"],
            n_jobs=self.model_params["n_jobs"],
            random_state=self.model_params["random_state"],
            verbose=self.model_params["verbose"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "n_estimators": [500, 1000, 1500],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "subsample": [0.7, 0.9],
        }
