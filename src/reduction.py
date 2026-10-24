"""PCA and LDA, fitted on training data only."""
import numpy as np


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

    Returns W (m, c-1) sorted by decreasing eigenvalue, and the eigenvalues (c-1,).
    """
    raise NotImplementedError
