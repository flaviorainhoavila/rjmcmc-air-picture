"""Motion and measurement models.

State convention: x = [px, py, vx, vy] in metres and metres per second.
Measurement convention: z = [range, azimuth], with azimuth in radians measured
counter-clockwise from the +x axis, wrapped to (-pi, pi].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 4
MEAS_DIM = 2
POS = slice(0, 2)


def wrap_angle(a):
    """Wrap angles to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    wrapped = np.mod(a + np.pi, 2.0 * np.pi) - np.pi
    return np.where(wrapped == -np.pi, np.pi, wrapped)


# Nearly-constant-velocity (NCV) motion model


def ncv_transition(dt: float) -> np.ndarray:
    """State transition matrix F for the NCV model over a step of `dt` seconds."""
    F = np.eye(STATE_DIM)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def ncv_process_noise(dt: float, psd: float) -> np.ndarray:
    """Process noise covariance Q for continuous white-noise acceleration.

    `psd` is the power spectral density of the acceleration noise per axis
    (m^2/s^3). Per axis, Q = psd * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
    """
    q11 = dt**3 / 3.0
    q12 = dt**2 / 2.0
    q22 = dt
    Q = np.zeros((STATE_DIM, STATE_DIM))
    for p, v in ((0, 2), (1, 3)):
        Q[p, p] = q11
        Q[p, v] = Q[v, p] = q12
        Q[v, v] = q22
    return psd * Q


# Polar measurement model with sensor bias


@dataclass(frozen=True)
class Radar:
    """A 2-D radar measuring range and azimuth.

    The measurement model is

        z = [ ||p - s|| + range_bias,
              atan2(py - sy, px - sx) + azimuth_bias ] + w,   w ~ N(0, R),

    with R = diag(sigma_range^2, sigma_azimuth^2). The bias fields hold the
    true bias used by the simulator; a tracker that does not know the bias
    should evaluate the model with its own bias estimate (see `measure`).
    """

    name: str
    position: np.ndarray
    sigma_range: float
    sigma_azimuth: float
    range_bias: float = 0.0
    azimuth_bias: float = 0.0
    detection_prob: float = 1.0
    clutter_rate: float = 0.0
    fov_range: tuple[float, float] = (0.0, np.inf)
    fov_azimuth: tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self):
        object.__setattr__(self, "position", np.asarray(self.position, dtype=float))

    @property
    def bias(self) -> np.ndarray:
        return np.array([self.range_bias, self.azimuth_bias])

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.sigma_range**2, self.sigma_azimuth**2])

    @property
    def fov_volume(self) -> float:
        """Volume of the field of view in measurement space (m * rad)."""
        (r0, r1), (a0, a1) = self.fov_range, self.fov_azimuth
        return (r1 - r0) * (a1 - a0)

    def in_fov(self, z) -> np.ndarray:
        """True for measurements (..., 2) that lie in the field of view."""
        z = np.asarray(z, dtype=float)
        r, a = z[..., 0], z[..., 1]
        (r0, r1), (a0, a1) = self.fov_range, self.fov_azimuth
        return (r >= r0) & (r <= r1) & (a >= a0) & (a <= a1)

    def measure(self, x, bias=None) -> np.ndarray:
        """Noise-free measurement h(x; bias) for states of shape (..., 4).

        `bias` defaults to the radar's true bias. Pass `bias=(0, 0)` for the
        nominal model that a bias-ignoring tracker uses.
        """
        b = self.bias if bias is None else np.asarray(bias, dtype=float)
        d = np.asarray(x, dtype=float)[..., POS] - self.position
        r = np.hypot(d[..., 0], d[..., 1])
        a = np.arctan2(d[..., 1], d[..., 0])
        return np.stack([r + b[0], wrap_angle(a + b[1])], axis=-1)

    def jacobian(self, x) -> np.ndarray:
        """Jacobian dh/dx of shape (2, 4) at a single state. Independent of the bias."""
        d = np.asarray(x, dtype=float)[POS] - self.position
        r2 = d @ d
        r = np.sqrt(r2)
        H = np.zeros((MEAS_DIM, STATE_DIM))
        H[0, 0:2] = d / r
        H[1, 0:2] = np.array([-d[1], d[0]]) / r2
        return H

    def to_cartesian(self, z, bias=None) -> np.ndarray:
        """Invert the noise-free model: measurements (..., 2) to positions (..., 2).

        Uses the radar's true bias by default; `bias=(0, 0)` gives the position a
        bias-ignoring tracker would infer.
        """
        b = self.bias if bias is None else np.asarray(bias, dtype=float)
        z = np.asarray(z, dtype=float)
        r = z[..., 0] - b[0]
        a = z[..., 1] - b[1]
        return self.position + np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)
