import numpy as np
import pytest
from scipy.stats import chi2

from airpicture.metrics import cardinality_error, duplicate_track_count, ospa, position_nees
from airpicture.models import ncv_process_noise, ncv_transition

# OSPA: hand-computed cases


def test_ospa_empty_sets():
    assert ospa([], [], c=10.0) == 0.0
    assert ospa([[1.0, 2.0]], [], c=10.0) == 10.0
    assert ospa([], [[1.0, 2.0], [3.0, 4.0]], c=10.0) == 10.0


def test_ospa_identical_sets():
    X = [[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]]
    assert ospa(X, X[::-1], c=10.0) == 0.0


def test_ospa_single_pair():
    # 3-4-5 triangle, below the cut-off: OSPA equals the distance for any p.
    assert ospa([[0.0, 0.0]], [[3.0, 4.0]], c=10.0, p=1) == pytest.approx(5.0)
    assert ospa([[0.0, 0.0]], [[3.0, 4.0]], c=10.0, p=2) == pytest.approx(5.0)


def test_ospa_cutoff():
    assert ospa([[0.0, 0.0]], [[20.0, 0.0]], c=10.0) == pytest.approx(10.0)


def test_ospa_cardinality_mismatch():
    # One matched pair at distance 0, one unmatched point costing c.
    # p=1: (0 + 10) / 2 = 5.   p=2: sqrt((0 + 100) / 2) = sqrt(50).
    X = [[0.0, 0.0]]
    Y = [[0.0, 0.0], [100.0, 0.0]]
    assert ospa(X, Y, c=10.0, p=1) == pytest.approx(5.0)
    assert ospa(X, Y, c=10.0, p=2) == pytest.approx(np.sqrt(50.0))
    assert ospa(Y, X, c=10.0, p=2) == pytest.approx(np.sqrt(50.0))


def test_ospa_mixed_case():
    # Pairs at distances 3 and 4, one extra point; c=5, p=2:
    # sqrt((3^2 + 4^2 + 5^2) / 3) = sqrt(50 / 3).
    X = [[0.0, 0.0], [10.0, 0.0]]
    Y = [[0.0, 3.0], [10.0, 4.0], [50.0, 50.0]]
    assert ospa(X, Y, c=5.0, p=2) == pytest.approx(np.sqrt(50.0 / 3.0))


def test_ospa_uses_optimal_assignment():
    # Greedy in listed order would pair (0,0)-(9,0); the optimum pairs crosswise, cost 1 each.
    X = [[0.0, 0.0], [10.0, 0.0]]
    Y = [[9.0, 0.0], [1.0, 0.0]]
    assert ospa(X, Y, c=100.0, p=1) == pytest.approx(1.0)


def test_ospa_is_a_metric_on_random_sets():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A, B, C = (rng.normal(0, 5, size=(rng.integers(0, 5), 2)) for _ in range(3))
        dab, dba = ospa(A, B, c=8.0), ospa(B, A, c=8.0)
        assert dab == pytest.approx(dba)
        assert 0.0 <= dab <= 8.0
        assert dab <= ospa(A, C, c=8.0) + ospa(C, B, c=8.0) + 1e-9


# Cardinality and duplicates


def test_cardinality_error_is_signed():
    assert cardinality_error(5, 3) == 2
    np.testing.assert_array_equal(cardinality_error([2, 3, 4], [3, 3, 3]), [-1, 0, 1])


def test_duplicate_track_count():
    truth = [[0.0, 0.0], [10000.0, 0.0]]
    # Two tracks on target 0 (one duplicate), three on target 1 (two duplicates),
    # one false track far from both.
    est = [[50.0, 0.0], [400.0, 300.0], [10000.0, 10.0], [9800.0, 0.0], [10300.0, 0.0], [0, 9e3]]
    assert duplicate_track_count(est, truth, gate=1000.0) == 3
    assert duplicate_track_count(est[:1] + est[2:3], truth, gate=1000.0) == 0
    assert duplicate_track_count([], truth, gate=1000.0) == 0
    assert duplicate_track_count(est, [], gate=1000.0) == 0


# Position NEES


def test_position_nees_hand_case():
    P = np.diag([4.0, 9.0])
    assert position_nees([2.0, 3.0], P, [0.0, 0.0]) == pytest.approx(1.0 + 1.0)
    np.testing.assert_allclose(
        position_nees(np.zeros((3, 2)), np.broadcast_to(P, (3, 2, 2)), [[2, 0], [0, 3], [2, 3]]),
        [1.0, 1.0, 2.0],
    )


def _run_kalman_filter(n_runs, n_steps, r_assumed_scale=1.0, seed=0):
    """Linear toy problem: NCV motion, noisy Cartesian position measurements.

    Returns position NEES with shape (n_steps, n_runs). The filter's model matches
    the simulation exactly when r_assumed_scale is 1.
    """
    rng = np.random.default_rng(seed)
    dt, q, sigma = 5.0, 1.0, 50.0
    F, Q = ncv_transition(dt), ncv_process_noise(dt, q)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = sigma**2 * np.eye(2)
    R_filter = r_assumed_scale * R
    m0 = np.array([0.0, 50000.0, 200.0, 0.0])
    P0 = np.diag([100.0**2, 100.0**2, 20.0**2, 20.0**2])

    x = m0 + rng.multivariate_normal(np.zeros(4), P0, size=n_runs)
    m = np.tile(m0, (n_runs, 1))
    P = P0.copy()
    nees = np.empty((n_steps, n_runs))
    for k in range(n_steps):
        x = x @ F.T + rng.multivariate_normal(np.zeros(4), Q, size=n_runs)
        z = x[:, :2] + rng.normal(0.0, sigma, size=(n_runs, 2))
        m, P = m @ F.T, F @ P @ F.T + Q
        S = H @ P @ H.T + R_filter
        K = P @ H.T @ np.linalg.inv(S)
        m = m + (z - m @ H.T) @ K.T
        P = (np.eye(4) - K @ H) @ P
        nees[k] = position_nees(m[:, :2], np.broadcast_to(P[:2, :2], (n_runs, 2, 2)), x[:, :2])
    return nees


def test_nees_near_two_for_ideal_kalman_filter():
    n_runs, n_steps = 2000, 40
    nees = _run_kalman_filter(n_runs, n_steps)
    # At every step, n_runs * mean NEES ~ chi2(2 * n_runs) across independent runs.
    lo, hi = chi2.ppf([0.0005, 0.9995], df=2 * n_runs) / n_runs
    step_means = nees.mean(axis=1)
    # Allow one or two steps out of 40 at this level purely by chance.
    assert np.sum((step_means < lo) | (step_means > hi)) <= 2
    assert nees.mean() == pytest.approx(2.0, abs=0.05)


def test_nees_flags_overconfident_filter():
    # Filter assumes a quarter of the true measurement variance: covariance too small.
    nees = _run_kalman_filter(500, 40, r_assumed_scale=0.25)
    assert nees.mean() > 3.0
