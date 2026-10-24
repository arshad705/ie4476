"""PCA and LDA, fitted on training data only."""
import warnings

import numpy as np
from scipy.linalg import eigh


def fit_pca(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA via SVD on Xtr (N, n).

    Returns mean (n,), eigvecs (n, r) as columns sorted largest first,
    eigvals (r,), where r = min(N - 1, n). Eigenvalues are the covariance
    eigenvalues with 1/N normalisation (slides 61-72).
    """
    N, n = Xtr.shape
    m = Xtr.mean(axis=0)
    # SVD of the centred data avoids forming the n x n covariance matrix
    _, s, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    # centring removes one degree of freedom, so the N-th component has eigenvalue ~0
    r = min(N - 1, n)
    return m, Vt[:r].T, s[:r] ** 2 / N


def pca_project(X: np.ndarray, m: np.ndarray, eigvecs: np.ndarray, d: int) -> np.ndarray:
    """Project X (M, n) onto the first d principal components -> (M, d)."""
    if not 1 <= d <= eigvecs.shape[1]:
        raise ValueError(f"d must be in [1, {eigvecs.shape[1]}], got {d}")
    return (X - m) @ eigvecs[:, :d]


def fit_lda(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit LDA on PCA features Z (N, m) with labels y (N,).

    Solves S_b w = lambda S_w w (slides 73-77). Z must have m <= N - c columns
    so that S_w is invertible. Returns W (m, c-1) sorted by decreasing
    eigenvalue, normalised so that W^T S_w W = I, and the eigenvalues (c-1,).
    """
    classes = np.unique(y)
    mean = Z.mean(axis=0)
    Sw = np.zeros((Z.shape[1], Z.shape[1]))
    Sb = np.zeros_like(Sw)
    for k in classes:
        Zk = Z[y == k]
        mu = Zk.mean(axis=0)
        Sw += (Zk - mu).T @ (Zk - mu)
        Sb += len(Zk) * np.outer(mu - mean, mu - mean)
    try:
        vals, vecs = eigh(Sb, Sw)
    except np.linalg.LinAlgError:
        # S_w numerically not positive definite (m at the N - c limit): add a tiny ridge
        warnings.warn("S_w not positive definite; adding a 1e-10 relative ridge")
        Sw += 1e-10 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
        vals, vecs = eigh(Sb, Sw)
    # eigh returns ascending order; S_b has rank c - 1, so keep the top c - 1
    order = np.argsort(vals)[::-1][:len(classes) - 1]
    return vecs[:, order], vals[order]
