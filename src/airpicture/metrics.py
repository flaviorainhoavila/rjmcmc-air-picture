"""Tracking metrics: OSPA, position NEES, cardinality error, duplicate tracks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _as_points(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(0, 2) if X.size == 0 else np.atleast_2d(X)


def ospa(X, Y, c: float, p: float = 2.0) -> float:
    """Optimal sub-pattern assignment distance between two finite sets of points.

    X and Y have shapes (m, d) and (n, d); either may be empty. With m <= n,

        OSPA = ( (1/n) * ( min_pi sum_i min(d(x_i, y_pi(i)), c)^p + c^p (n - m) ) )^(1/p),

    which is 0 for two empty sets and c when exactly one set is empty
    (Schuhmacher, Vo and Vo, IEEE Trans. Signal Processing, 2008).
    """
    X, Y = _as_points(X), _as_points(Y)
    m, n = len(X), len(Y)
    if m > n:
        X, Y, m, n = Y, X, n, m
    if n == 0:
        return 0.0
    if m == 0:
        return float(c)
    cost = np.minimum(cdist(X, Y), c) ** p
    rows, cols = linear_sum_assignment(cost)
    total = cost[rows, cols].sum() + c**p * (n - m)
    return float((total / n) ** (1.0 / p))


def position_nees(est_pos, cov_pos, true_pos) -> np.ndarray:
    """Normalised estimation error squared for positions.

    est_pos and true_pos have shape (..., 2); cov_pos has shape (..., 2, 2).
    Returns e^T P^{-1} e with shape (...). For a consistent estimator the
    expected value is 2.
    """
    e = np.asarray(est_pos, dtype=float) - np.asarray(true_pos, dtype=float)
    P = np.asarray(cov_pos, dtype=float)
    return np.einsum("...i,...i->...", e, np.linalg.solve(P, e[..., None])[..., 0])


def cardinality_error(n_est, n_true):
    """Signed cardinality error, estimated minus true number of targets."""
    return np.asarray(n_est) - np.asarray(n_true)


def duplicate_track_count(est_pos, true_pos, gate: float) -> int:
    """Number of surplus tracks on true targets.

    Each estimated track is assigned to its nearest true target if it lies
    within `gate`; tracks beyond the first on the same target are duplicates.
    Tracks outside every gate are false tracks and are not counted here.
    """
    E, T = _as_points(est_pos), _as_points(true_pos)
    if len(E) == 0 or len(T) == 0:
        return 0
    d = cdist(E, T)
    nearest = d.argmin(axis=1)
    within = d[np.arange(len(E)), nearest] <= gate
    counts = np.bincount(nearest[within], minlength=len(T))
    return int(np.maximum(counts - 1, 0).sum())
