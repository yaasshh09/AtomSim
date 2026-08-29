"""Small analysis utilities I keep because they mean something physically."""

import numpy as np


def count_sign_changes(y: np.ndarray, rel_floor: float = 1e-6) -> int:
    """I count sign changes of y, ignoring values below rel_floor * max|y|, which I treat as noise."""
    y = np.asarray(y)
    mask = np.abs(y) > rel_floor * np.abs(y).max()
    signs = np.sign(y[mask])
    return int(np.sum(signs[1:] != signs[:-1]))
