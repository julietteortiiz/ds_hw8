"""Provided figure layouts. All PCA and SVD calculations belong in your code."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure under figures/ in the current homework directory."""
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path("figures") / filename)
    plt.close(fig)


def plot_variance(cumulative: np.ndarray) -> None:
    """Plot already-computed cumulative ratios; component counts start at 1."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    ax.set(xlabel="Component count", ylabel="Cumulative explained variance ratio",
           title="Digits: cumulative explained variance",
           xlim=(1, len(cumulative)), ylim=(0, 1.01))
    ax.grid(alpha=.25)
    _save(fig, "variance.pdf")


def plot_digits_2d(Z: np.ndarray, y: np.ndarray, ratios: np.ndarray) -> None:
    """Plot provided (n, 2) coordinates, labels, and two variance ratios."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10",
                         norm=BoundaryNorm(np.arange(-.5, 10.5), 10), s=9, alpha=.6)
    ax.set(xlabel=f"PC1 ({ratios[0]:.1%})", ylabel=f"PC2 ({ratios[1]:.1%})",
           title="Digits in two PCA dimensions")
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit")
    _save(fig, "digits_2d.pdf")


def plot_digit_reconstructions(originals: np.ndarray,
                               reconstructions: dict[int, np.ndarray]) -> None:
    """Plot ten original rows (digits 0-9) and ten reconstructed rows per k."""
    titles = ["Original"] + [f"k={k}" for k in reconstructions]
    columns = [originals] + list(reconstructions.values())
    fig, axes = plt.subplots(10, len(columns), figsize=(8, 14))
    for digit in range(10):
        for column, values in enumerate(columns):
            ax = axes[digit, column]
            ax.imshow(values[digit].reshape(8, 8), cmap="gray", vmin=0, vmax=16)
            ax.set_xticks([])
            ax.set_yticks([])
            if digit == 0:
                ax.set_title(titles[column])
            if column == 0:
                ax.set_ylabel(str(digit), rotation=0, labelpad=10)
    fig.suptitle("First occurrence of each digit: PCA reconstructions")
    _save(fig, "digits_reconstructed.pdf")


def plot_svd_image(image: np.ndarray, approximations: dict[int, np.ndarray]) -> None:
    """Plot a single original image and the supplied rank-k approximations."""
    titles = ["Original"] + [f"k={k}" for k in approximations]
    images = [image] + list(approximations.values())
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    for ax, values, title in zip(axes, images, titles):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Uncentered image SVD approximations")
    _save(fig, "svd_image.pdf")
