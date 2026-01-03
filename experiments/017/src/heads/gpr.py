"""Gaussian Process Regression head implementation."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel

from .base import MultiTargetHead


class MultiTargetGPRHead(MultiTargetHead):
    """Multi-target Gaussian Process Regression head.

    Note: GPR has O(n^3) complexity, so it may be slow for large datasets.
    Consider using a subset of data or approximate methods for large datasets.
    """

    HEAD_TYPE: str = "gpr"

    def __init__(
        self,
        kernel_type: str = "rbf",
        length_scale: float = 1.0,
        length_scale_bounds: tuple = (1e-3, 1e3),
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
        normalize_y: bool = True,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetGPRHead.

        Args:
            kernel_type: Type of kernel ('rbf', 'matern', 'rational_quadratic').
            length_scale: Initial length scale for kernel.
            length_scale_bounds: Bounds for length scale optimization.
            alpha: Noise level (regularization).
            n_restarts_optimizer: Number of optimizer restarts.
            normalize_y: Whether to normalize target values.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            kernel_type=kernel_type,
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
        )

    def _create_kernel(self):
        """Create kernel based on kernel_type."""
        kernel_type = self.model_params["kernel_type"]
        length_scale = self.model_params["length_scale"]
        length_scale_bounds = self.model_params["length_scale_bounds"]

        # Constant kernel for amplitude
        const_kernel = ConstantKernel(1.0, (1e-3, 1e3))

        if kernel_type == "rbf":
            base_kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        elif kernel_type == "matern":
            base_kernel = Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5)
        elif kernel_type == "rational_quadratic":
            base_kernel = RationalQuadratic(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Add white noise kernel
        white_kernel = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

        return const_kernel * base_kernel + white_kernel

    def _create_model(self) -> GaussianProcessRegressor:
        """Create a new GaussianProcessRegressor instance."""
        kernel = self._create_kernel()

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.model_params["alpha"],
            n_restarts_optimizer=self.model_params["n_restarts_optimizer"],
            normalize_y=self.model_params["normalize_y"],
            random_state=42,
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "kernel_type": ["rbf", "matern"],
            "length_scale": [0.1, 1.0, 10.0],
            "alpha": [1e-10, 1e-8, 1e-6],
        }

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation.

        Args:
            X: Features [N, D].

        Returns:
            Tuple of (predictions, std) each of shape [N, num_targets].
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        predictions = []
        stds = []
        for target_name in self.target_names:
            pred, std = self.models[target_name].predict(X_scaled, return_std=True)
            pred = np.maximum(pred, 0)
            predictions.append(pred)
            stds.append(std)

        return np.stack(predictions, axis=1), np.stack(stds, axis=1)
