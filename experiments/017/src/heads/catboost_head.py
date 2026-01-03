"""CatBoost head implementation."""

from catboost import CatBoostRegressor

from .base import MultiTargetHead


class MultiTargetCatBoostHead(MultiTargetHead):
    """Multi-target CatBoost regression head.

    CatBoost with ordered boosting and native categorical feature support.
    """

    HEAD_TYPE: str = "catboost"

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        border_count: int = 254,
        random_state: int = 42,
        verbose: int = 0,
        task_type: str = "CPU",
        target_names: list[str] | None = None,
        scale_features: bool = False,
    ):
        """Initialize MultiTargetCatBoostHead.

        Args:
            iterations: Maximum number of trees.
            learning_rate: Learning rate.
            depth: Depth of the trees.
            l2_leaf_reg: L2 regularization coefficient.
            random_strength: Amount of randomness for scoring splits.
            bagging_temperature: Controls intensity of Bayesian bagging.
            border_count: Number of splits for numerical features.
            random_state: Random seed.
            verbose: Verbosity level (0=silent).
            task_type: 'CPU' or 'GPU'.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            border_count=border_count,
            random_state=random_state,
            verbose=verbose,
            task_type=task_type,
        )

    def _create_model(self) -> CatBoostRegressor:
        """Create a new CatBoostRegressor instance."""
        return CatBoostRegressor(
            iterations=self.model_params["iterations"],
            learning_rate=self.model_params["learning_rate"],
            depth=self.model_params["depth"],
            l2_leaf_reg=self.model_params["l2_leaf_reg"],
            random_strength=self.model_params["random_strength"],
            bagging_temperature=self.model_params["bagging_temperature"],
            border_count=self.model_params["border_count"],
            random_state=self.model_params["random_state"],
            verbose=self.model_params["verbose"],
            task_type=self.model_params["task_type"],
            allow_writing_files=False,
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "iterations": [500, 1000, 1500],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8],
        }
