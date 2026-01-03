"""Gradient Boosting head implementations (GradientBoosting, HistGradientBoosting)."""

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from .base import MultiTargetHead


class MultiTargetGBDTHead(MultiTargetHead):
    """Multi-target Gradient Boosting regression head.

    sklearn's GradientBoostingRegressor with standard GBDT algorithm.
    """

    HEAD_TYPE: str = "gbdt"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
        target_names: list[str] | None = None,
        scale_features: bool = False,  # GBDTはスケーリング不要
    ):
        """Initialize MultiTargetGBDTHead.

        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Learning rate shrinks contribution of each tree.
            max_depth: Maximum depth of individual trees.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            subsample: Fraction of samples used for fitting individual trees.
            random_state: Random seed.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler (default False for GBDT).
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )

    def _create_model(self) -> GradientBoostingRegressor:
        """Create a new GradientBoostingRegressor instance."""
        return GradientBoostingRegressor(
            n_estimators=self.model_params["n_estimators"],
            learning_rate=self.model_params["learning_rate"],
            max_depth=self.model_params["max_depth"],
            min_samples_split=self.model_params["min_samples_split"],
            min_samples_leaf=self.model_params["min_samples_leaf"],
            subsample=self.model_params["subsample"],
            random_state=self.model_params["random_state"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "n_estimators": [100, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }


class MultiTargetHistGBDTHead(MultiTargetHead):
    """Multi-target Histogram-based Gradient Boosting regression head.

    sklearn's HistGradientBoostingRegressor - faster for large datasets.
    """

    HEAD_TYPE: str = "histgbdt"

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        target_names: list[str] | None = None,
        scale_features: bool = False,
    ):
        """Initialize MultiTargetHistGBDTHead.

        Args:
            max_iter: Maximum number of iterations (trees).
            learning_rate: Learning rate.
            max_depth: Maximum depth of trees (None for unlimited).
            max_leaf_nodes: Maximum number of leaf nodes per tree.
            min_samples_leaf: Minimum samples per leaf.
            l2_regularization: L2 regularization parameter.
            random_state: Random seed.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def _create_model(self) -> HistGradientBoostingRegressor:
        """Create a new HistGradientBoostingRegressor instance."""
        return HistGradientBoostingRegressor(
            max_iter=self.model_params["max_iter"],
            learning_rate=self.model_params["learning_rate"],
            max_depth=self.model_params["max_depth"],
            max_leaf_nodes=self.model_params["max_leaf_nodes"],
            min_samples_leaf=self.model_params["min_samples_leaf"],
            l2_regularization=self.model_params["l2_regularization"],
            random_state=self.model_params["random_state"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "max_iter": [100, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
        }
