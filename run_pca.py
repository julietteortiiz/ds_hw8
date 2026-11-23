"""HW08 PCA experiments. Figure layouts and experiment orchestration are provided."""
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from PCA import PCA
from data_utils import load_digits_data
from plotting import plot_variance, plot_digits_2d, plot_digit_reconstructions


def numerical_checks() -> None:
    """Part 1: check orthonormality, full reconstruction, and error versus k."""
    X = np.array([[1., 2.], [2., 1.], [4., 5.], [5., 4.]])
    model = PCA(2)
    model.fit(X)
    #Check Orthogonal
    a = np.matmul(model.components_, model.components_.T)
    I = np.identity(2)
    if (np.allclose(a, I, rtol=1e-05, atol=1e-08, equal_nan=False) != True):
        print("Principle Components are not orthogonal")
        exit(1)

    #Check Full Reconstruction
    principle_components = model.transform(X)
    reconstructed_data = model.reconstruct_data(principle_components)
    if (np.allclose(reconstructed_data, X, rtol=1e-05, atol=1e-08, equal_nan=False) != True):
        print("Reconstruction Failed")
    
    #Check Error Versus K
    k1 = 1 
    model2 = PCA(k1)
    model2.fit(X)
    pc = model2.transform(X)
    rd = model2.reconstruct_data(pc)
    #checks of k=1 yields same result as k=2 then the error rate stayed the same 
    #when really it shoulde decrease. 
    if (np.allclose(rd, X, rtol=1e-05, atol=1e-08, equal_nan=False) == True):
        print("Error increases as K increases, issue w k")


def components_for_variance(ratios: np.ndarray, threshold: float) -> int:
    """Return the smallest k whose cumulative explained variance >= threshold.

    ratios contains all explained-variance ratios in component order.
    For this assignment threshold is 0.90 or 0.95. Counts start at 1.
    """
    # TODO: accumulate variance ratios and find the first count reaching the threshold.
    raise NotImplementedError("Implement components_for_variance")


def variance_experiment(X: np.ndarray) -> PCA:
    """Fit all components, report 90%/95% counts, and plot cumulative variance."""
    model = PCA(X.shape[1]).fit(X)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    for threshold in (.90, .95):
        k = components_for_variance(model.explained_variance_ratio_, threshold)
        print(f"Components for {threshold:.0%} variance: {k}")
    plot_variance(cumulative)
    return model


def projection_experiment(model: PCA, X: np.ndarray, y: np.ndarray) -> None:
    """Plot the first two coordinates, using labels only to color points."""
    Z = model.transform(X)[:, :2]
    plot_digits_2d(Z, y, model.explained_variance_ratio_[:2])


def reconstruction_experiment(X: np.ndarray, y: np.ndarray) -> PCA:
    """Report all-pixel MSE and plot the first occurrence of each digit.

    Return the fitted 10-component model for the sklearn comparison.
    """
    indices = np.array([np.flatnonzero(y == digit)[0] for digit in range(10)])
    reconstructions = {}
    models = {}
    for k in (2, 10, 30, 64):
        model = PCA(k).fit(X)
        restored = model.reconstruct_data(model.transform(X))
        print(f"Digits k={k}: MSE={np.mean((X - restored)**2):.12g}")
        reconstructions[k] = restored[indices]
        models[k] = model
    plot_digit_reconstructions(X[indices], reconstructions)
    return models[10]


def compare_with_sklearn(X: np.ndarray, model: PCA) -> None:
    """Compare the given 10-component model with sklearn's full-solver PCA.

    Report the maximum absolute differences in explained-variance ratios
    and reconstructed data. Sklearn calls reconstruction inverse_transform.
    """
    # TODO: fit SklearnPCA(n_components=10, svd_solver="full") on X.
    # Compare variance ratios and reconstructions; print both maximum differences.
    raise NotImplementedError("Complete compare_with_sklearn")


def main() -> None:
    numerical_checks()
    #X, y = load_digits_data()
    #full = variance_experiment(X)
    #projection_experiment(full, X, y)
    #ten = reconstruction_experiment(X, y)
    #compare_with_sklearn(X, ten)

if __name__ == "__main__":
    main()
