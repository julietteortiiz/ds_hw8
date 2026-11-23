"""HW08 uncentered image SVD. Plotting and the rank loop are provided."""
import numpy as np
from data_utils import load_svd_image
from plotting import plot_svd_image


def low_rank_approximation(U: np.ndarray, s: np.ndarray,
                           Vt: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct the rank-k approximation from an already-computed SVD.

    U has shape (m, r), s (r,), and Vt (r, p). Return shape (m, p).
    Use the first k terms without modifying the factors. Here 1 <= k <= r.
    """
    # TODO: combine the first k columns/values/rows of the SVD factors.
    raise NotImplementedError("Implement low_rank_approximation")


def main() -> None:
    image = load_svd_image()
    # Compute one uncentered decomposition, then truncate it at each rank.
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    approximations = {}
    for k in (1, 2, 5, 10):
        approximations[k] = low_rank_approximation(U, s, Vt, k)
    plot_svd_image(image, approximations)


if __name__ == "__main__":
    main()
