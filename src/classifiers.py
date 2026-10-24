"""Fit-and-predict classifiers. Fitting uses training data only."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular


def _class_means(Ztr: np.ndarray, ytr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return classes (c,) and class means M (c, d)."""
    classes = np.unique(ytr)
    return classes, np.stack([Ztr[ytr == k].mean(axis=0) for k in classes])


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (M, K) between rows of A (M, d) and B (K, d).

    Uses |a|^2 + |b|^2 - 2 a.b, so no (M, K, d) intermediate is formed.
    """
    D = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.maximum(D, 0)


def nn_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray) -> np.ndarray:
    """1-nearest-neighbour (Euclidean). Ztr (N, d), ytr (N,), Zte (M, d) -> (M,)."""
    return ytr[_sq_dists(Zte, Ztr).argmin(axis=1)]


def min_euclid_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray) -> np.ndarray:
    """Closest class mean (Euclidean). Ztr (N, d), ytr (N,), Zte (M, d) -> (M,)."""
    classes, M = _class_means(Ztr, ytr)
    return classes[_sq_dists(Zte, M).argmin(axis=1)]


def min_mahal_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
                      eps: float = 0.0) -> np.ndarray:
    """Closest class mean under pooled within-class covariance (+ eps * I).

    Sigma_w = S_w / (N - c) (slides 17-56; regularised form slide 93).
    Ztr (N, d), ytr (N,), Zte (M, d) -> (M,). Requires d <= N - c when eps == 0.
    """
    classes, M = _class_means(Ztr, ytr)
    N, d = Ztr.shape
    dof = N - len(classes)
    if eps == 0 and d > dof:
        raise ValueError(f"Sigma_w is singular for d = {d} > N - c = {dof}; use eps > 0")
    R = Ztr - M[np.searchsorted(classes, ytr)]
    Sigma = R.T @ R / dof + eps * np.eye(d)
    # Sigma = L L^T, so (z - mu)^T Sigma^-1 (z - mu) = |L^-1 (z - mu)|^2:
    # whiten with L^-1 and use Euclidean distance, without forming Sigma^-1
    try:
        L = cholesky(Sigma, lower=True)
    except np.linalg.LinAlgError as e:
        raise ValueError("Sigma_w is not positive definite; use eps > 0") from e
    Wte = solve_triangular(L, Zte.T, lower=True).T
    WM = solve_triangular(L, M.T, lower=True).T
    return classes[_sq_dists(Wte, WM).argmin(axis=1)]
