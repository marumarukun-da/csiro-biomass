"""SVR model wrapper for biomass regression using scikit-learn."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class MultiTargetSVR:
    """Multi-target SVR wrapper using scikit-learn.

    Trains separate SVR models for each target (Dry_Total_g, GDM_g, Dry_Green_g).
    Includes StandardScaler for feature normalization.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
        cache_size: int = 2000,
        target_names: list[str] | None = None,
    ):
        """Initialize MultiTargetSVR.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
            C: Regularization parameter.
            gamma: Kernel coefficient ('scale', 'auto', or float).
            epsilon: Epsilon in epsilon-SVR model.
            cache_size: Cache size in MB for kernel matrix.
            target_names: Names of target columns.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.target_names = target_names or ["Dry_Total_g", "GDM_g", "Dry_Green_g"]

        self.scaler: StandardScaler | None = None
        self.models: dict[str, SVR] = {}
        self.is_fitted = False

    def _create_svr(self) -> SVR:
        """Create a new SVR instance with current parameters."""
        return SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon,
            cache_size=self.cache_size,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiTargetSVR":
        """Fit SVR models for each target.

        Args:
            X: Features [N, D] (D=2560 for CLS+PatchMean).
            y: Targets [N, num_targets].

        Returns:
            self
        """
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit SVR for each target
        self.models = {}
        for i, target_name in enumerate(self.target_names):
            svr = self._create_svr()
            svr.fit(X_scaled, y[:, i])
            self.models[target_name] = svr

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

        X_scaled = self.scaler.transform(X)

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
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "cache_size": self.cache_size,
            "target_names": self.target_names,
            "scaler": self.scaler,
            "models": self.models,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str | Path) -> "MultiTargetSVR":
        """Load model from file.

        Args:
            path: Path to saved model.

        Returns:
            Loaded MultiTargetSVR instance.
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        instance = cls(
            kernel=save_dict["kernel"],
            C=save_dict["C"],
            gamma=save_dict["gamma"],
            epsilon=save_dict["epsilon"],
            cache_size=save_dict["cache_size"],
            target_names=save_dict["target_names"],
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
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

    def __repr__(self) -> str:
        return (
            f"MultiTargetSVR(kernel={self.kernel}, C={self.C}, "
            f"gamma={self.gamma}, epsilon={self.epsilon}, "
            f"targets={self.target_names})"
        )


def extract_features_from_npz(npz_path: str | Path, aug_idx: int = 0) -> np.ndarray:
    """Extract CLS + PatchMean features from .npz file.

    Args:
        npz_path: Path to .npz feature file.
        aug_idx: Augmentation index to use (default: 0 for original).

    Returns:
        Feature vector [2560] (CLS 1280 + PatchMean 1280).
    """
    data = np.load(npz_path)
    cls_token = data[f"cls_{aug_idx}"]  # [1280]
    patch_tokens = data[f"patches_{aug_idx}"]  # [3600, 1280]
    patch_mean = patch_tokens.mean(axis=0)  # [1280]

    return np.concatenate([cls_token, patch_mean])  # [2560]


def load_all_features(
    df,
    feature_dir: str | Path,
    aug_idx: int = 0,
) -> np.ndarray:
    """Load all features from precomputed .npz files.

    Args:
        df: DataFrame with image_id column.
        feature_dir: Directory containing .npz files.
        aug_idx: Augmentation index to use.

    Returns:
        Feature matrix [N, 2560].
    """
    feature_dir = Path(feature_dir)
    features = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        npz_path = feature_dir / f"{image_id}.npz"
        feat = extract_features_from_npz(npz_path, aug_idx=aug_idx)
        features.append(feat)

    return np.stack(features, axis=0)
