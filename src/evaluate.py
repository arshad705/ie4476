"""Metrics, experiment runners and training-only validation."""
from typing import Callable, Sequence

import numpy as np

from .classifiers import min_euclid_predict, min_mahal_predict, nn_predict
from .config import STEP
from .data import make_split
from .reduction import fit_lda, fit_pca, pca_project

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
                       d_list: Sequence[int], step: int = STEP,
                       m_pca: int | None = None) -> np.ndarray:
    """E2. PCA to m_pca dims (default N - c), then LDA, both fitted per split
    on training data, then each classifier.

    Returns acc[offset, classifier, d_index] with classifiers in CLASSIFIERS
    order; nan for d > c - 1.
    """
    acc = np.full((step, len(CLASSIFIERS), len(d_list)), np.nan)
    for k in range(step):
        tr, te = make_split(pose, k, step)
        m, V, _ = fit_pca(X[tr])
        mp = len(tr) - len(np.unique(y[tr])) if m_pca is None else m_pca
        W, _ = fit_lda(pca_project(X[tr], m, V, mp), y[tr])
        Ltr = pca_project(X[tr], m, V, mp) @ W
        Lte = pca_project(X[te], m, V, mp) @ W
        for j, d in enumerate(d_list):
            if d > W.shape[1]:
                continue
            for i, predict in enumerate(CLASSIFIERS.values()):
                acc[k, i, j] = accuracy(predict(Ltr[:, :d], y[tr], Lte[:, :d]), y[te])
    return acc


def run_trainsize_experiment(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                             steps: Sequence[int]) -> dict:
    """E4. Returns per-step accuracy arrays keyed by step."""
    raise NotImplementedError


def select_by_validation(X: np.ndarray, y: np.ndarray, pose: np.ndarray,
                         train_idx: np.ndarray, candidates: Sequence,
                         score_fn: Callable) -> tuple[object, np.ndarray]:
    """Leave-one-pose-out within the training set.

    For each training pose p: validate on images at pose p, fit on the
    other training poses. Return the candidate with the best mean score.

    score_fn(Xfit, yfit, Xval, yval) returns one score per candidate (nan if
    the candidate is undefined for that fold). Only train_idx is ever used.
    Ties go to the earliest candidate. Returns (best candidate, mean scores).
    """
    train_idx = np.asarray(train_idx)
    folds = []
    for p in np.unique(pose[train_idx]):
        val = train_idx[pose[train_idx] == p]
        fit = train_idx[pose[train_idx] != p]
        folds.append(score_fn(X[fit], y[fit], X[val], y[val]))
    scores = np.array(folds)
    # a candidate counts only if it is defined in every fold
    mean = np.where(np.isnan(scores).any(axis=0), np.nan, scores.mean(axis=0))
    return candidates[int(np.nanargmax(mean))], mean


def pca_scorer(predict: Callable, d_list: Sequence[int]) -> Callable:
    """score_fn for select_by_validation over PCA dimension d (one per d_list entry)."""
    def score(Xf, yf, Xv, yv):
        m, V, _ = fit_pca(Xf)
        Zf, Zv = pca_project(Xf, m, V, V.shape[1]), pca_project(Xv, m, V, V.shape[1])
        out = np.full(len(d_list), np.nan)
        for j, d in enumerate(d_list):
            if d > V.shape[1]:
                continue
            try:
                out[j] = accuracy(predict(Zf[:, :d], yf, Zv[:, :d]), yv)
            except ValueError:   # min-Mahal above the N - c cap of this fold
                pass
        return out
    return score


def lda_scorer(predict: Callable, m_list: Sequence[int],
               d_list: Sequence[int]) -> tuple[list[tuple[int, int]], Callable]:
    """Candidates [(m, d), ...] and score_fn for joint selection of the
    intermediate PCA dim m and the LDA dim d. m above N - c of a fold is nan.
    """
    candidates = [(mp, d) for mp in m_list for d in d_list]

    def score(Xf, yf, Xv, yv):
        m, V, _ = fit_pca(Xf)
        cap = len(yf) - len(np.unique(yf))
        out = np.full(len(candidates), np.nan)
        for a, mp in enumerate(m_list):
            if mp > cap:
                continue
            Zf, Zv = pca_project(Xf, m, V, mp), pca_project(Xv, m, V, mp)
            W, _ = fit_lda(Zf, yf)
            Lf, Lv = Zf @ W, Zv @ W
            for b, d in enumerate(d_list):
                if d <= W.shape[1]:
                    out[a * len(d_list) + b] = accuracy(predict(Lf[:, :d], yf, Lv[:, :d]), yv)
        return out
    return candidates, score
