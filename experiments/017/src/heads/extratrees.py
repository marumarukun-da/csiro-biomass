"""ExtraTrees head implementation."""

from sklearn.ensemble import ExtraTreesRegressor

from .base import MultiTargetHead


class MultiTargetExtraTreesHead(MultiTargetHead):
    """Multi-target Extra Trees regression head.

    Extremely Randomized Trees - similar to Random Forest but with
    random split thresholds for faster training.
    """

    HEAD_TYPE: str = "extratrees"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float | str = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        target_names: list[str] | None = None,
        scale_features: bool = False,
    ):
        """Initialize MultiTargetExtraTreesHead.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            max_features: Number of features to consider for best split.
                - float: fraction of features
                - 'sqrt': sqrt(n_features)
                - 'log2': log2(n_features)
            bootstrap: Whether to use bootstrap samples.
            n_jobs: Number of parallel jobs.
            random_state: Random seed.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _create_model(self) -> ExtraTreesRegressor:
        """Create a new ExtraTreesRegressor instance."""
        return ExtraTreesRegressor(
            n_estimators=self.model_params["n_estimators"],
            max_depth=self.model_params["max_depth"],
            min_samples_split=self.model_params["min_samples_split"],
            min_samples_leaf=self.model_params["min_samples_leaf"],
            max_features=self.model_params["max_features"],
            bootstrap=self.model_params["bootstrap"],
            n_jobs=self.model_params["n_jobs"],
            random_state=self.model_params["random_state"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
