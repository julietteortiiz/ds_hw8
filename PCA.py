"""HW08 PCA with NumPy's reduced SVD. Inputs are real numeric NumPy arrays; validation is provided."""
import numpy as np
from validation import validate_fit_input, validate_array, require_fitted


class PCA:
    def __init__(self, n_components: int) -> None:
        """Store the requested component count (a Python int); validate when fitting."""
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit (n, p) data; store the k leading directions and variance statistics.

        Store mean_ (p,), components_ (k, p), singular_values_ (k,),
        explained_variance_ (k,), and explained_variance_ratio_ (k,).
        Ratios must use ALL singular values in the denominator. Return self.
        Do not modify X; refitting must replace the fitted state.
        """
        X = validate_fit_input(X, self.n_components)
        n = len(X)
        k = self.n_components
        # 1. Center Data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # 2. Compute the Reduced SVD
        U, self.singular_values_, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:k]
        # 3. Store Required Attributes
        self.explained_variance_ = (self.singular_values_ ** 2) / (n-1)
        self.explained_variance_ratio_ = (self.singular_values_ ** 2) / np.sum(self.singular_values_ ** 2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return (m, k) coordinates using the stored fitting mean and directions."""
        require_fitted(self)
        X = validate_array(X, n_columns=self.mean_.size)
        # Project X using the stored mean and principal directions.
        X_centered = X - self.mean_
        principle_coordinates = np.matmul(X_centered, self.components_.T)
        return principle_coordinates

    def reconstruct_data(self, Z: np.ndarray) -> np.ndarray:
        """Return (m, p) reconstructions, restoring the fitting mean."""
        require_fitted(self)
        Z = validate_array(Z, n_columns=self.components_.shape[0])
        #Map Z back to feature space and restore the fitting mean.
        approximate_og_data = np.matmul(Z, self.components_)
        approximate_og_data = approximate_og_data + self.mean_
        return approximate_og_data
        
