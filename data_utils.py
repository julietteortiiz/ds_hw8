"""Provided data loaders; paths are relative to this module."""
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_digits_data() -> tuple[np.ndarray, np.ndarray]:
    """Return independent copies: X (1797, 64), y (1797,), original order."""
    with np.load(DATA_DIR / "digits.npz", allow_pickle=False) as data:
        return data["X"].copy(), data["y"].copy()


def load_svd_image() -> np.ndarray:
    """Return the provided uncentered 128 by 128 grayscale disk image."""
    return np.load(DATA_DIR / "svd_image.npy", allow_pickle=False)
