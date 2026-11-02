import numpy as np
import pytest

from airpicture.models import Radar, ncv_process_noise, ncv_transition, wrap_angle


@pytest.fixture
def radar():
    return Radar(
        name="B",
        position=[1000.0, -2000.0],
        sigma_range=50.0,
        sigma_azimuth=1e-3,
        range_bias=300.0,
        azimuth_bias=5e-3,
    )


def test_wrap_angle():
    a = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi / 2, 2 * np.pi])
    np.testing.assert_allclose(wrap_angle(a), [0.0, np.pi, np.pi, np.pi, np.pi / 2, 0.0])


def test_ncv_transition_propagates_constant_velocity():
    x = np.array([10.0, -5.0, 200.0, 50.0])
    np.testing.assert_allclose(ncv_transition(5.0) @ x, [1010.0, 245.0, 200.0, 50.0])


def test_ncv_process_noise_values():
    dt, q = 2.0, 3.0
    Q = ncv_process_noise(dt, q)
    expected_axis = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    np.testing.assert_allclose(Q[np.ix_([0, 2], [0, 2])], expected_axis)
    np.testing.assert_allclose(Q[np.ix_([1, 3], [1, 3])], expected_axis)
    assert Q[0, 1] == Q[0, 3] == Q[2, 3] == 0.0
    np.testing.assert_allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_measure_known_geometry(radar):
    # Target 3 km east and 4 km north of the radar: range 5 km, azimuth atan2(4, 3).
    x = np.array([radar.position[0] + 3000.0, radar.position[1] + 4000.0, 0.0, 0.0])
    np.testing.assert_allclose(radar.measure(x, bias=(0.0, 0.0)), [5000.0, np.arctan2(4, 3)])
    np.testing.assert_allclose(radar.measure(x), [5300.0, np.arctan2(4, 3) + 5e-3])


def test_measure_is_vectorised(radar):
    xs = np.random.default_rng(0).normal(0.0, 1e4, size=(7, 4))
    z = radar.measure(xs)
    assert z.shape == (7, 2)
    np.testing.assert_allclose(z[3], radar.measure(xs[3]))


def test_to_cartesian_inverts_measure(radar):
    xs = np.random.default_rng(1).normal(0.0, 1e4, size=(20, 4))
    np.testing.assert_allclose(radar.to_cartesian(radar.measure(xs)), xs[:, :2], atol=1e-6)


def test_ignoring_bias_displaces_position(radar):
    # A 300 m / 5 mrad bias at 50 km range: the inferred point is at range 50.3 km,
    # rotated by 5 mrad, so the shift is |50300 * exp(i * 5e-3) - 50000| (about 391 m).
    x = np.array([radar.position[0], radar.position[1] + 50000.0, 0.0, 0.0])
    p = radar.to_cartesian(radar.measure(x), bias=(0.0, 0.0))
    shift = np.linalg.norm(p - x[:2])
    assert shift == pytest.approx(abs(50300.0 * np.exp(5e-3j) - 50000.0), rel=1e-9)


def test_jacobian_matches_finite_differences(radar):
    x = np.array([15000.0, 30000.0, 120.0, -80.0])
    H = radar.jacobian(x)
    eps = 1e-3
    H_fd = np.zeros_like(H)
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        H_fd[:, i] = (radar.measure(x + dx) - radar.measure(x - dx)) / (2 * eps)
    np.testing.assert_allclose(H, H_fd, rtol=1e-6, atol=1e-12)


def test_fov(radar):
    r = Radar("A", [0.0, 0.0], 1.0, 1.0, fov_range=(10.0, 20.0), fov_azimuth=(0.0, 1.0))
    z = np.array([[15.0, 0.5], [5.0, 0.5], [15.0, 1.5]])
    np.testing.assert_array_equal(r.in_fov(z), [True, False, False])
    assert r.fov_volume == pytest.approx(10.0)
