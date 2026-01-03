"""SVR head implementation."""

from sklearn.svm import SVR

from .base import MultiTargetHead


class MultiTargetSVRHead(MultiTargetHead):
    """Multi-target SVR head using scikit-learn.

    Trains separate SVR models for each target.
    """

    HEAD_TYPE: str = "svr"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
        cache_size: int = 2000,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetSVRHead.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
            C: Regularization parameter.
            gamma: Kernel coefficient ('scale', 'auto', or float).
            epsilon: Epsilon in epsilon-SVR model.
            cache_size: Cache size in MB for kernel matrix.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            cache_size=cache_size,
        )

    def _create_model(self) -> SVR:
        """Create a new SVR instance with current parameters."""
        return SVR(
            kernel=self.model_params["kernel"],
            C=self.model_params["C"],
            gamma=self.model_params["gamma"],
            epsilon=self.model_params["epsilon"],
            cache_size=self.model_params["cache_size"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01, 0.1],
            "epsilon": [0.01, 0.1, 0.5],
        }
