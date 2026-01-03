"""Kernel Ridge regression head implementation."""

from sklearn.kernel_ridge import KernelRidge

from .base import MultiTargetHead


class MultiTargetKernelRidgeHead(MultiTargetHead):
    """Multi-target Kernel Ridge regression head.

    Combines Ridge regression with kernel trick for non-linear regression.
    """

    HEAD_TYPE: str = "kernel_ridge"

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetKernelRidgeHead.

        Args:
            alpha: Regularization strength.
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'cosine').
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
                   If None, defaults to 1/n_features.
            degree: Degree for polynomial kernel.
            coef0: Independent term in 'poly' and 'sigmoid' kernels.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

    def _create_model(self) -> KernelRidge:
        """Create a new KernelRidge instance."""
        return KernelRidge(
            alpha=self.model_params["alpha"],
            kernel=self.model_params["kernel"],
            gamma=self.model_params["gamma"],
            degree=self.model_params["degree"],
            coef0=self.model_params["coef0"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": [0.001, 0.01, 0.1, None],
        }
