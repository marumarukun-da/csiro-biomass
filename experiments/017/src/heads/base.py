"""Base class for multi-target regression heads."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiTargetHead(ABC):
    """Abstract base class for multi-target regression heads.

    All head implementations should inherit from this class and implement
    the abstract methods. Each head trains separate models for each target.
    """

    # Class attribute to identify head type (override in subclasses)
    HEAD_TYPE: str = "base"

    def __init__(
        self,
        target_names: list[str] | None = None,
        scale_features: bool = True,
        **kwargs: Any,
    ):
        """Initialize MultiTargetHead.

        Args:
            target_names: Names of target columns.
            scale_features: Whether to apply StandardScaler to features.
            **kwargs: Additional parameters for the underlying model.
        """
        self.target_names = target_names or ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
        self.scale_features = scale_features
        self.model_params = kwargs

        self.scaler: StandardScaler | None = None
        self.models: dict[str, Any] = {}
        self.is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create a new model instance with current parameters.

        Returns:
            A new unfitted model instance.
        """
        pass

    @abstractmethod
    def get_param_grid(self) -> dict[str, list]:
        """Get default parameter grid for grid search.

        Returns:
            Dictionary mapping parameter names to lists of values.
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiTargetHead":
        """Fit models for each target.

        Args:
            X: Features [N, D].
            y: Targets [N, num_targets].

        Returns:
            self
        """
        # Fit scaler if needed
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit model for each target
        self.models = {}
        for i, target_name in enumerate(self.target_names):
            model = self._create_model()
            model.fit(X_scaled, y[:, i])
            self.models[target_name] = model

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets.

        Args:
            X: Features [N, D].

        Returns:
            Predictions [N, num_targets].
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        predictions = []
        for target_name in self.target_names:
            pred = self.models[target_name].predict(X_scaled)
            # Ensure non-negative (biomass values)
            pred = np.maximum(pred, 0)
            predictions.append(pred)

        return np.stack(predictions, axis=1)

    def save(self, path: str | Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "head_type": self.HEAD_TYPE,
            "target_names": self.target_names,
            "scale_features": self.scale_features,
            "model_params": self.model_params,
            "scaler": self.scaler,
            "models": self.models,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str | Path) -> "MultiTargetHead":
        """Load model from file.

        Args:
            path: Path to saved model.

        Returns:
            Loaded MultiTargetHead instance.
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        instance = cls(
            target_names=save_dict["target_names"],
            scale_features=save_dict["scale_features"],
            **save_dict["model_params"],
        )
        instance.scaler = save_dict["scaler"]
        instance.models = save_dict["models"]
        instance.is_fitted = save_dict["is_fitted"]

        return instance

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of parameters.
        """
        return {
            "head_type": self.HEAD_TYPE,
            **self.model_params,
        }

    def set_params(self, **params: Any) -> "MultiTargetHead":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self
        """
        for key, value in params.items():
            if key in self.model_params:
                self.model_params[key] = value
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.model_params.items())
        return f"{self.__class__.__name__}({params_str}, targets={self.target_names})"
