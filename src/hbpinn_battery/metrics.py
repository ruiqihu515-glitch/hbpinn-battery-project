"""Generic regression metrics."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Return the root mean squared error between two equally shaped arrays."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Return the mean absolute error between two equally shaped arrays."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(true - pred)))
