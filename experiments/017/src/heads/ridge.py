"""Linear regression head implementations (Ridge, Lasso, ElasticNet, BayesianRidge)."""

from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge

from .base import MultiTargetHead


class MultiTargetRidgeHead(MultiTargetHead):
    """Multi-target Ridge regression head.

    L2 regularized linear regression.
    """

    HEAD_TYPE: str = "ridge"

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetRidgeHead.

        Args:
            alpha: Regularization strength.
            fit_intercept: Whether to fit intercept.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            alpha=alpha,
            fit_intercept=fit_intercept,
        )

    def _create_model(self) -> Ridge:
        """Create a new Ridge instance."""
        return Ridge(
            alpha=self.model_params["alpha"],
            fit_intercept=self.model_params["fit_intercept"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }


class MultiTargetLassoHead(MultiTargetHead):
    """Multi-target Lasso regression head.

    L1 regularized linear regression (sparse solutions).
    """

    HEAD_TYPE: str = "lasso"

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 10000,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetLassoHead.

        Args:
            alpha: Regularization strength.
            fit_intercept: Whether to fit intercept.
            max_iter: Maximum number of iterations.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )

    def _create_model(self) -> Lasso:
        """Create a new Lasso instance."""
        return Lasso(
            alpha=self.model_params["alpha"],
            fit_intercept=self.model_params["fit_intercept"],
            max_iter=self.model_params["max_iter"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        }


class MultiTargetElasticNetHead(MultiTargetHead):
    """Multi-target ElasticNet regression head.

    L1 + L2 regularized linear regression.
    """

    HEAD_TYPE: str = "elasticnet"

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 10000,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetElasticNetHead.

        Args:
            alpha: Regularization strength.
            l1_ratio: L1/L2 mixing ratio (0=Ridge, 1=Lasso).
            fit_intercept: Whether to fit intercept.
            max_iter: Maximum number of iterations.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )

    def _create_model(self) -> ElasticNet:
        """Create a new ElasticNet instance."""
        return ElasticNet(
            alpha=self.model_params["alpha"],
            l1_ratio=self.model_params["l1_ratio"],
            fit_intercept=self.model_params["fit_intercept"],
            max_iter=self.model_params["max_iter"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }


class MultiTargetBayesianRidgeHead(MultiTargetHead):
    """Multi-target Bayesian Ridge regression head.

    Bayesian linear regression with automatic relevance determination.
    """

    HEAD_TYPE: str = "bayesian_ridge"

    def __init__(
        self,
        n_iter: int = 300,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        fit_intercept: bool = True,
        target_names: list[str] | None = None,
        scale_features: bool = True,
    ):
        """Initialize MultiTargetBayesianRidgeHead.

        Args:
            n_iter: Maximum number of iterations.
            alpha_1: Shape parameter for Gamma prior over alpha.
            alpha_2: Rate parameter for Gamma prior over alpha.
            lambda_1: Shape parameter for Gamma prior over lambda.
            lambda_2: Rate parameter for Gamma prior over lambda.
            fit_intercept: Whether to fit intercept.
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler.
        """
        super().__init__(
            target_names=target_names,
            scale_features=scale_features,
            n_iter=n_iter,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            fit_intercept=fit_intercept,
        )

    def _create_model(self) -> BayesianRidge:
        """Create a new BayesianRidge instance."""
        return BayesianRidge(
            n_iter=self.model_params["n_iter"],
            alpha_1=self.model_params["alpha_1"],
            alpha_2=self.model_params["alpha_2"],
            lambda_1=self.model_params["lambda_1"],
            lambda_2=self.model_params["lambda_2"],
            fit_intercept=self.model_params["fit_intercept"],
        )

    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search."""
        return {
            "alpha_1": [1e-7, 1e-6, 1e-5],
            "alpha_2": [1e-7, 1e-6, 1e-5],
            "lambda_1": [1e-7, 1e-6, 1e-5],
            "lambda_2": [1e-7, 1e-6, 1e-5],
        }
