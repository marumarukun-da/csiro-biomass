"""Feature engineering engine with PCA and PLS."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEngine:
    """Feature preprocessing engine using PCA and PLS.

    This engine applies dimensionality reduction using both PCA (unsupervised)
    and PLS (supervised) and concatenates the resulting components.

    Args:
        n_pca: Number of PCA components. Can be:
            - int: exact number of components
            - float (0-1): cumulative variance ratio to preserve
            - 0 or None: disable PCA
        n_pls: Number of PLS components. Can be:
            - int > 0: number of components
            - 0 or None: disable PLS
        scale: Whether to apply StandardScaler before PCA/PLS.
    """

    def __init__(
        self,
        n_pca: int | float | None = 0.95,
        n_pls: int | None = 8,
        scale: bool = True,
    ):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.scale = scale

        # Components to be fitted
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.pls: PLSRegression | None = None

        # State
        self.is_fitted = False
        self.n_pca_components_: int | None = None
        self.n_pls_components_: int | None = None

    @property
    def use_pca(self) -> bool:
        """Check if PCA is enabled."""
        return self.n_pca is not None and self.n_pca != 0

    @property
    def use_pls(self) -> bool:
        """Check if PLS is enabled."""
        return self.n_pls is not None and self.n_pls != 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureEngine:
        """Fit the feature engine.

        Args:
            X: Features [N, D].
            y: Targets [N, num_targets]. Required for PLS.

        Returns:
            self
        """
        # Scale features
        if self.scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit PCA (unsupervised)
        if self.use_pca:
            self.pca = PCA(n_components=self.n_pca, random_state=42)
            self.pca.fit(X_scaled)
            self.n_pca_components_ = self.pca.n_components_

        # Fit PLS (supervised)
        if self.use_pls:
            # Ensure y is 2D
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # PLS n_components must be <= min(n_samples, n_features, n_targets)
            max_pls = min(X_scaled.shape[0], X_scaled.shape[1], y.shape[1])
            n_pls_actual = min(self.n_pls, max_pls)

            self.pls = PLSRegression(n_components=n_pls_actual, scale=False)
            self.pls.fit(X_scaled, y)
            self.n_pls_components_ = n_pls_actual

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted PCA and PLS.

        Args:
            X: Features [N, D].

        Returns:
            Transformed features [N, n_pca + n_pls].
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngine is not fitted. Call fit() first.")

        # Scale features
        if self.scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        components = []

        # PCA transform
        if self.use_pca and self.pca is not None:
            pca_features = self.pca.transform(X_scaled)
            components.append(pca_features)

        # PLS transform
        if self.use_pls and self.pls is not None:
            pls_features = self.pls.transform(X_scaled)
            components.append(pls_features)

        # If no components, return scaled features
        if not components:
            return X_scaled

        return np.hstack(components)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Features [N, D].
            y: Targets [N, num_targets].

        Returns:
            Transformed features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_output_dim(self) -> int:
        """Get the output dimension after transformation."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngine is not fitted.")

        dim = 0
        if self.n_pca_components_:
            dim += self.n_pca_components_
        if self.n_pls_components_:
            dim += self.n_pls_components_
        return dim

    def save(self, path: str | Path) -> None:
        """Save the fitted engine to a file.

        Args:
            path: Path to save the engine.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_pca": self.n_pca,
            "n_pls": self.n_pls,
            "scale": self.scale,
            "scaler": self.scaler,
            "pca": self.pca,
            "pls": self.pls,
            "is_fitted": self.is_fitted,
            "n_pca_components_": self.n_pca_components_,
            "n_pls_components_": self.n_pls_components_,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> FeatureEngine:
        """Load a fitted engine from a file.

        Args:
            path: Path to the saved engine.

        Returns:
            Loaded FeatureEngine instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        engine = cls(
            n_pca=state["n_pca"],
            n_pls=state["n_pls"],
            scale=state["scale"],
        )
        engine.scaler = state["scaler"]
        engine.pca = state["pca"]
        engine.pls = state["pls"]
        engine.is_fitted = state["is_fitted"]
        engine.n_pca_components_ = state["n_pca_components_"]
        engine.n_pls_components_ = state["n_pls_components_"]

        return engine

    def get_params(self) -> dict[str, Any]:
        """Get parameters for this engine."""
        return {
            "n_pca": self.n_pca,
            "n_pls": self.n_pls,
            "scale": self.scale,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        pca_info = f"n_pca={self.n_pca}"
        if self.is_fitted and self.n_pca_components_:
            pca_info += f" ({self.n_pca_components_} components)"
        pls_info = f"n_pls={self.n_pls}"
        if self.is_fitted and self.n_pls_components_:
            pls_info += f" ({self.n_pls_components_} components)"
        return f"FeatureEngine({pca_info}, {pls_info}, scale={self.scale}) [{status}]"
