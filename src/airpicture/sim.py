"""Seeded scenario generator: births, deaths, missed detections, clutter, sensor bias.

Each scan k at time t_k = k * scan_period has a set of true targets and, for
every radar, a set of detections. Both radars report at every scan time.
Detection rows are shuffled so their order says nothing about their origin;
the `origin` array (target id, or -1 for clutter) is ground truth for
evaluation only and must not be read by trackers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from airpicture.models import Radar, ncv_process_noise, ncv_transition, wrap_angle

CLUTTER = -1


@dataclass(frozen=True)
class ScenarioConfig:
    n_scans: int
    scan_period: float
    region_x: tuple[float, float]
    region_y: tuple[float, float]
    process_noise_psd: float
    initial_targets: int
    birth_rate: float
    survival_prob: float
    speed_range: tuple[float, float]
    radars: tuple[Radar, ...]

    def __post_init__(self):
        for radar in self.radars:
            _check_region_in_fov(radar, self.region_x, self.region_y)

    @classmethod
    def from_dict(cls, d: Mapping) -> ScenarioConfig:
        radars = tuple(
            Radar(
                name=name,
                position=r["position_m"],
                sigma_range=r["sigma_range_m"],
                sigma_azimuth=r["sigma_azimuth_rad"],
                range_bias=r.get("range_bias_m", 0.0),
                azimuth_bias=r.get("azimuth_bias_rad", 0.0),
                detection_prob=r["detection_prob"],
                clutter_rate=r["clutter_rate"],
                fov_range=tuple(r["fov_range_m"]),
                fov_azimuth=tuple(r["fov_azimuth_rad"]),
            )
            for name, r in d["radars"].items()
        )
        return cls(
            n_scans=int(d["n_scans"]),
            scan_period=float(d["scan_period_s"]),
            region_x=tuple(d["region_x_m"]),
            region_y=tuple(d["region_y_m"]),
            process_noise_psd=float(d["process_noise_psd"]),
            initial_targets=int(d["initial_targets"]),
            birth_rate=float(d["birth_rate"]),
            survival_prob=float(d["survival_prob"]),
            speed_range=tuple(d["speed_range_mps"]),
            radars=radars,
        )

    def radar(self, name: str) -> Radar:
        return next(r for r in self.radars if r.name == name)


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _check_region_in_fov(radar: Radar, region_x, region_y) -> None:
    """Raise if any point of the region falls outside the radar's field of view.

    For a box that does not contain the radar, the azimuth extremes and the
    maximum range are attained at corners; the minimum range is the distance
    from the radar to the box.
    """
    (x0, x1), (y0, y1) = region_x, region_y
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
    d = corners - radar.position
    az = np.arctan2(d[:, 1], d[:, 0])
    closest = np.clip(radar.position, [x0, y0], [x1, y1])
    r_min = np.linalg.norm(closest - radar.position)
    r_max = np.linalg.norm(d, axis=1).max()
    (fr0, fr1), (fa0, fa1) = radar.fov_range, radar.fov_azimuth
    if r_min == 0.0:
        raise ValueError(f"radar {radar.name} lies inside the surveillance region")
    if np.ptp(az) > np.pi:
        raise ValueError(f"region straddles the azimuth wrap for radar {radar.name}")
    if r_min < fr0 or r_max > fr1 or az.min() < fa0 or az.max() > fa1:
        raise ValueError(f"surveillance region is not inside the field of view of {radar.name}")


@dataclass
class Detections:
    """Detections of one radar at one scan."""

    radar: str
    z: np.ndarray  # (m, 2) range and azimuth
    origin: np.ndarray  # (m,) target id, or CLUTTER; evaluation only


@dataclass
class Scan:
    k: int
    time: float
    truth_ids: np.ndarray  # (n,)
    truth_states: np.ndarray  # (n, 4)
    detections: dict[str, Detections] = field(default_factory=dict)


@dataclass
class Scenario:
    config: ScenarioConfig
    seed: int
    scans: list[Scan]

    def truth_tracks(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Per target id: (scan indices, states) over the target's lifetime."""
        tracks: dict[int, tuple[list, list]] = {}
        for scan in self.scans:
            for tid, x in zip(scan.truth_ids, scan.truth_states, strict=True):
                ks, xs = tracks.setdefault(int(tid), ([], []))
                ks.append(scan.k)
                xs.append(x)
        return {tid: (np.array(ks), np.array(xs)) for tid, (ks, xs) in tracks.items()}


def _in_region(states: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    (x0, x1), (y0, y1) = cfg.region_x, cfg.region_y
    px, py = states[:, 0], states[:, 1]
    return (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)


def _sample_births(n: int, cfg: ScenarioConfig, rng: np.random.Generator) -> np.ndarray:
    # ASSUMPTION: new targets appear uniformly inside the region with uniform heading
    # and a speed drawn uniformly from speed_range. See docs/assumptions.md.
    px = rng.uniform(*cfg.region_x, size=n)
    py = rng.uniform(*cfg.region_y, size=n)
    speed = rng.uniform(*cfg.speed_range, size=n)
    heading = rng.uniform(-np.pi, np.pi, size=n)
    return np.column_stack([px, py, speed * np.cos(heading), speed * np.sin(heading)])


def _detect(
    radar: Radar,
    ids: np.ndarray,
    states: np.ndarray,
    rng: np.random.Generator,
) -> Detections:
    detected = rng.random(len(ids)) < radar.detection_prob
    ids, states = ids[detected], states[detected]
    noise = rng.normal(size=(len(ids), 2)) * [radar.sigma_range, radar.sigma_azimuth]
    z_t = radar.measure(states).reshape(-1, 2) + noise
    z_t[:, 1] = wrap_angle(z_t[:, 1])
    keep = radar.in_fov(z_t)
    z_t, ids = z_t[keep], ids[keep]

    # ASSUMPTION: clutter is uniform over the radar's field of view in (range, azimuth),
    # so its intensity in measurement space is clutter_rate / fov_volume.
    n_c = rng.poisson(radar.clutter_rate)
    z_c = np.column_stack(
        [rng.uniform(*radar.fov_range, size=n_c), rng.uniform(*radar.fov_azimuth, size=n_c)]
    )

    z = np.vstack([z_t, z_c])
    origin = np.concatenate([ids, np.full(n_c, CLUTTER)]).astype(int)
    perm = rng.permutation(len(z))
    return Detections(radar=radar.name, z=z[perm], origin=origin[perm])


def simulate(cfg: ScenarioConfig, seed: int) -> Scenario:
    """Generate one scenario.

    Truth and each radar's detections use independent random streams derived from
    `seed`, so changing a radar's settings leaves the trajectories unchanged.
    """
    truth_rng, *radar_rngs = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(1 + len(cfg.radars))
    )
    F = ncv_transition(cfg.scan_period)
    L = np.linalg.cholesky(ncv_process_noise(cfg.scan_period, cfg.process_noise_psd))

    states = _sample_births(cfg.initial_targets, cfg, truth_rng)
    ids = np.arange(cfg.initial_targets)
    next_id = cfg.initial_targets

    scans = []
    for k in range(cfg.n_scans):
        if k > 0:
            # Survive, move, then die on leaving the region; then add births.
            alive = truth_rng.random(len(ids)) < cfg.survival_prob
            ids, states = ids[alive], states[alive]
            states = states @ F.T + truth_rng.normal(size=states.shape) @ L.T
            inside = _in_region(states, cfg)
            ids, states = ids[inside], states[inside]

            n_b = truth_rng.poisson(cfg.birth_rate)
            ids = np.concatenate([ids, np.arange(next_id, next_id + n_b)])
            states = np.vstack([states, _sample_births(n_b, cfg, truth_rng)])
            next_id += n_b

        scan = Scan(k=k, time=k * cfg.scan_period, truth_ids=ids.copy(), truth_states=states.copy())
        for radar, rng in zip(cfg.radars, radar_rngs, strict=True):
            scan.detections[radar.name] = _detect(radar, ids, states, rng)
        scans.append(scan)

    return Scenario(config=cfg, seed=seed, scans=scans)


# Plotting

_RADAR_STYLE = {
    # Fixed categorical order (blue, orange); marker shape is the secondary encoding.
    0: {"color": "#2a78d6", "marker": "o"},
    1: {"color": "#eb6834", "marker": "^"},
}
_INK = "#1a1a19"
_MUTED = "#8c8b84"


def plot_scenario(scenario: Scenario, ax=None, show_clutter: bool = True):
    """Plot true trajectories and detections in Cartesian coordinates (km).

    Detections are mapped to the plane with the nominal (zero-bias) model, which is
    what a tracker that ignores the bias sees: Radar B's detections sit visibly off
    the true trajectories.
    """
    import matplotlib.pyplot as plt

    cfg = scenario.config
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0), layout="constrained")

    (x0, x1), (y0, y1) = cfg.region_x, cfg.region_y
    ax.add_patch(
        plt.Rectangle(
            (x0 / 1e3, y0 / 1e3),
            (x1 - x0) / 1e3,
            (y1 - y0) / 1e3,
            fill=False,
            edgecolor=_MUTED,
            linewidth=1.0,
            linestyle="--",
        )
    )

    for i, radar in enumerate(cfg.radars):
        style = _RADAR_STYLE[i % len(_RADAR_STYLE)]
        z = [scan.detections[radar.name] for scan in scenario.scans]
        zs = np.vstack([d.z for d in z])
        is_target = np.concatenate([d.origin for d in z]) != CLUTTER
        p = radar.to_cartesian(zs, bias=(0.0, 0.0)) / 1e3
        if show_clutter:
            ax.scatter(
                *p[~is_target].T,
                s=6,
                marker=style["marker"],
                facecolors="none",
                edgecolors=style["color"],
                linewidths=0.5,
                alpha=0.35,
            )
        ax.scatter(
            *p[is_target].T,
            s=10,
            marker=style["marker"],
            color=style["color"],
            linewidths=0,
            alpha=0.8,
            label=f"Radar {radar.name} detections ({_bias_label(radar)})",
        )
        ax.plot(
            *radar.position / 1e3,
            marker=style["marker"],
            markersize=11,
            color=style["color"],
            markeredgecolor=_INK,
            linestyle="none",
        )
        ax.annotate(
            f"Radar {radar.name}",
            radar.position / 1e3,
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color=_INK,
        )

    for n, (_, (_, xs)) in enumerate(scenario.truth_tracks().items()):
        ax.plot(
            xs[:, 0] / 1e3,
            xs[:, 1] / 1e3,
            color=_INK,
            linewidth=1.5,
            label="True trajectory" if n == 0 else None,
        )
        ax.plot(*xs[0, :2] / 1e3, marker="o", markersize=4, color=_INK)

    if show_clutter:
        ax.scatter([], [], s=6, marker="o", facecolors="none", edgecolors=_MUTED, label="Clutter")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    ax.set_xlim(x0 / 1e3 - 10, x1 / 1e3 + 10)
    ax.set_ylim(min(r.position[1] for r in cfg.radars) / 1e3 - 8, y1 / 1e3 + 5)
    ax.grid(color="#e4e3dc", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    n_targets = len(scenario.truth_tracks())
    ax.set_title(
        f"Scenario, seed {scenario.seed}: {n_targets} targets over {cfg.n_scans} scans",
        loc="left",
        color=_INK,
    )
    return ax


def _bias_label(radar: Radar) -> str:
    if radar.range_bias == 0.0 and radar.azimuth_bias == 0.0:
        return "no bias"
    return f"bias {radar.range_bias:.0f} m, {radar.azimuth_bias * 1e3:.1f} mrad, ignored"
