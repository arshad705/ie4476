"""Fit-and-predict classifiers. Fitting uses training data only."""
import numpy as np


def nn_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray) -> np.ndarray:
    """1-nearest-neighbour (Euclidean). Ztr (N, d), ytr (N,), Zte (M, d) -> (M,)."""
    raise NotImplementedError


def min_euclid_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray) -> np.ndarray:
    """Closest class mean (Euclidean). Ztr (N, d), ytr (N,), Zte (M, d) -> (M,)."""
    raise NotImplementedError


def min_mahal_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
                      eps: float = 0.0) -> np.ndarray:
    """Closest class mean under pooled within-class covariance (+ eps * I).

    Ztr (N, d), ytr (N,), Zte (M, d) -> (M,). Requires d <= N - c when eps == 0.
    """
    raise NotImplementedError
