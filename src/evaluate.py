"""Metrics, experiment runners and training-only validation."""
from typing import Callable, Sequence

import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct predictions."""
    raise NotImplementedError


def error_rate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - accuracy."""
    raise NotImplementedError


def run_pca_experiment(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                       d_list: Sequence[int]) -> np.ndarray:
    """E1. Returns acc[offset, classifier, d_index]; nan where undefined."""
    raise NotImplementedError


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
