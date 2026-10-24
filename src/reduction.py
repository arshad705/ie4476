"""PCA and LDA, fitted on training data only."""
import numpy as np


def fit_pca(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA via SVD on Xtr (N, n).

    Returns mean (n,), eigvecs (n, N-1) as columns sorted largest first,
    eigvals (N-1,).
    """
    raise NotImplementedError


def pca_project(X: np.ndarray, m: np.ndarray, eigvecs: np.ndarray, d: int) -> np.ndarray:
    """Project X (M, n) onto the first d principal components -> (M, d)."""
    raise NotImplementedError


def fit_lda(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit LDA on PCA features Z (N, m) with labels y (N,).

    Returns W (m, c-1) sorted by decreasing eigenvalue, and the eigenvalues (c-1,).
    """
    raise NotImplementedError
