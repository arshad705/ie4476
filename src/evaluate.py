"""Metrics, experiment runners and training-only validation."""
from typing import Callable, Sequence

import numpy as np

from .classifiers import min_euclid_predict, min_mahal_predict, nn_predict
from .config import STEP
from .data import make_split
from .reduction import fit_pca, pca_project

CLASSIFIERS = {"1-NN": nn_predict, "min-Euclid": min_euclid_predict, "min-Mahal": min_mahal_predict}


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return float(np.mean(y_pred == y_true))


def error_rate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - accuracy."""
    return 1.0 - accuracy(y_pred, y_true)


def run_baseline(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                 step: int = STEP) -> np.ndarray:
    """E0. Classify raw pixels, no reduction.

    Returns acc[offset, classifier] for 1-NN and min-Euclid; min-Mahal is nan
    because Sigma_w is singular in pixel space (rank <= N - c < n).
    """
    acc = np.full((step, len(CLASSIFIERS)), np.nan)
    for k in range(step):
        tr, te = make_split(pose, k, step)
        acc[k, 0] = accuracy(nn_predict(X[tr], y[tr], X[te]), y[te])
        acc[k, 1] = accuracy(min_euclid_predict(X[tr], y[tr], X[te]), y[te])
    return acc


def run_pca_experiment(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                       d_list: Sequence[int], step: int = STEP) -> np.ndarray:
    """E1. PCA fitted per split on training data, then each classifier.

    Returns acc[offset, classifier, d_index] with classifiers in CLASSIFIERS
    order; nan where undefined (d > N - 1, or d > N - c for min-Mahal).
    """
    acc = np.full((step, len(CLASSIFIERS), len(d_list)), np.nan)
    for k in range(step):
        tr, te = make_split(pose, k, step)
        m, V, _ = fit_pca(X[tr])
        # project once to the maximum dimension, then slice columns per d
        Ztr = pca_project(X[tr], m, V, V.shape[1])
        Zte = pca_project(X[te], m, V, V.shape[1])
        cap = len(tr) - len(np.unique(y[tr]))   # N - c
        for j, d in enumerate(d_list):
            if d > V.shape[1]:
                continue
            for i, (name, predict) in enumerate(CLASSIFIERS.items()):
                if name == "min-Mahal" and d > cap:
                    continue
                acc[k, i, j] = accuracy(predict(Ztr[:, :d], y[tr], Zte[:, :d]), y[te])
    return acc


def run_lda_experiment(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                       d_list: Sequence[int]) -> np.ndarray:
    """E2 (PCA(N-c) -> LDA). Returns acc[offset, classifier, d_index]."""
    raise NotImplementedError


def run_trainsize_experiment(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                             steps: Sequence[int]) -> dict:
    """E4. Returns per-step accuracy arrays keyed by step."""
    raise NotImplementedError


def select_by_validation(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                         train_idx: np.ndarray, candidates: Sequence,
                         score_fn: Callable) -> object:
    """Leave-one-pose-out within the training set.

    For each training pose p: validate on images at pose p, fit on the
    other training poses. Return the candidate with the best mean score.
    """
    raise NotImplementedError
