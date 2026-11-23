"""Provided checks for real numeric NumPy arrays. No input conversion is needed."""
from __future__ import annotations
import numpy as np


def validate_array(X: np.ndarray, n_columns: int | None = None) -> np.ndarray:
    """Check shape and finite values; return the original array unchanged."""
    if X.ndim != 2:
        raise ValueError("Expected a two-dimensional array")
    if not np.isfinite(X).all():
        raise ValueError("Data must contain only finite values")
    if n_columns is not None and X.shape[1] != n_columns:
        raise ValueError(f"Expected {n_columns} columns")
    return X


def validate_fit_input(X: np.ndarray, n_components: int) -> np.ndarray:
    """Check fitting dimensions, the Python integer k, and nonzero variation."""
    validate_array(X)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("Fitting requires at least two rows and one feature")
    if type(n_components) is not int or not 1 <= n_components <= min(n, p):
        raise ValueError("n_components must be an integer in [1, min(n, p)]")
    # Check before centering, which can leave rounding residuals for equal rows.
    if np.all(X == X[0]):
        raise ValueError("Zero total centered variance")
    return X


def require_fitted(model: object) -> None:
    """Check that PCA has been fitted before projecting or reconstructing."""
    if not hasattr(model, "components_"):
        raise RuntimeError("Fit PCA before transforming or reconstructing")
