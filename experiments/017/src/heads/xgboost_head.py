"""XGBoost head implementation."""

from xgboost import XGBRegressor

from .base import MultiTargetHead


class MultiTargetXGBoostHead(MultiTargetHead):
    """Multi-target XGBoost regression head.

    XGBoost with GPU support and efficient gradient boosting.
    """

    HEAD_TYPE: str = "xgboost"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        device: str = "cpu",
        n_jobs: int = -1,
        random_state: int = 42,
        verbosity: int = 0,
        target_names: list[str] | None = None,
        scale_features: bool = False,
    ):
        """Initialize MultiTargetXGBoostHead.

        Args:
            n_estimators: Number of boosting rounds.
            learning_rate: Boosting learning rate.
            max_depth: Maximum tree depth.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns for each tree.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            tree_method: Tree construction algorithm ('hist', 'approx', 'exact').
            device: Device to use ('cpu', 'cuda').
            n_jobs: Number of parallel threads.
            random_state: Random seed.
            verbosity: Verbosity level (0=silent).
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            device=device,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=verbosity,
        )

    def _create_model(self) -> XGBRegressor:
        """Create a new XGBRegressor instance."""
        return XGBRegressor(
            n_estimators=self.model_params["n_estimators"],
            learning_rate=self.model_params["learning_rate"],
            max_depth=self.model_params["max_depth"],
            subsample=self.model_params["subsample"],
            colsample_bytree=self.model_params["colsample_bytree"],
            reg_alpha=self.model_params["reg_alpha"],
            reg_lambda=self.model_params["reg_lambda"],
            tree_method=self.model_params["tree_method"],
            device=self.model_params["device"],
            n_jobs=self.model_params["n_jobs"],
            random_state=self.model_params["random_state"],
            verbosity=self.model_params["verbosity"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "n_estimators": [500, 1000, 1500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.9],
        }
