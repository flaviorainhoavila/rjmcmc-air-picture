from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from airpicture.models import wrap_angle
from airpicture.sim import (
    CLUTTER,
    ScenarioConfig,
    detection_errors,
    load_config,
    plot_detection_errors,
    plot_scenario,
    simulate,
)

CONFIG = Path(__file__).parents[1] / "configs" / "benchmark.yaml"


@pytest.fixture(scope="module")
def cfg():
    return ScenarioConfig.from_dict(load_config(CONFIG)["scenario"])


@pytest.fixture(scope="module")
def long_cfg(cfg):
    # Many targets and scans, so the statistical checks below have enough samples.
    return replace(cfg, n_scans=200, initial_targets=10, birth_rate=1.0, survival_prob=0.995)


def test_same_seed_same_scenario(cfg):
    a, b = simulate(cfg, seed=3), simulate(cfg, seed=3)
    for sa, sb in zip(a.scans, b.scans, strict=True):
        np.testing.assert_array_equal(sa.truth_ids, sb.truth_ids)
        np.testing.assert_array_equal(sa.truth_states, sb.truth_states)
        for name in sa.detections:
            np.testing.assert_array_equal(sa.detections[name].z, sb.detections[name].z)


def test_different_seeds_differ(cfg):
    a, b = simulate(cfg, seed=0), simulate(cfg, seed=1)
    assert not np.allclose(a.scans[0].truth_states, b.scans[0].truth_states)


def test_truth_independent_of_radar_settings(cfg):
    noisier = replace(cfg, radars=tuple(replace(r, clutter_rate=50.0) for r in cfg.radars))
    a, b = simulate(cfg, seed=5), simulate(noisier, seed=5)
    for sa, sb in zip(a.scans, b.scans, strict=True):
        np.testing.assert_array_equal(sa.truth_states, sb.truth_states)


def test_truth_stays_in_region_and_ids_are_unique(cfg):
    sc = simulate(cfg, seed=0)
    (x0, x1), (y0, y1) = cfg.region_x, cfg.region_y
    for scan in sc.scans:
        p = scan.truth_states[:, :2]
        assert np.all((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))
        assert len(set(scan.truth_ids)) == len(scan.truth_ids)
    # A target id that disappears never comes back.
    for ks, _ in sc.truth_tracks().values():
        np.testing.assert_array_equal(np.diff(ks), 1)


def test_births_and_deaths_happen(long_cfg):
    sc = simulate(long_cfg, seed=0)
    n = np.array([len(s.truth_ids) for s in sc.scans])
    assert len(sc.truth_tracks()) > long_cfg.initial_targets
    assert np.any(np.diff(n) < 0)


def test_detection_probability_and_clutter_rate(long_cfg):
    sc = simulate(long_cfg, seed=1)
    for radar in long_cfg.radars:
        n_truth = sum(len(s.truth_ids) for s in sc.scans)
        n_det = sum(np.sum(s.detections[radar.name].origin != CLUTTER) for s in sc.scans)
        n_clut = [np.sum(s.detections[radar.name].origin == CLUTTER) for s in sc.scans]
        # Binomial and Poisson 4-sigma bounds.
        pd = radar.detection_prob
        assert abs(n_det / n_truth - pd) < 4 * np.sqrt(pd * (1 - pd) / n_truth)
        lam = radar.clutter_rate
        assert abs(np.mean(n_clut) - lam) < 4 * np.sqrt(lam / len(sc.scans))


def test_clutter_lies_in_fov(cfg):
    sc = simulate(cfg, seed=2)
    for scan in sc.scans:
        for radar in cfg.radars:
            assert np.all(radar.in_fov(scan.detections[radar.name].z))


def test_residuals_reveal_bias(long_cfg):
    sc = simulate(long_cfg, seed=4)
    for radar in long_cfg.radars:
        res = []
        for scan in sc.scans:
            det = scan.detections[radar.name]
            states = dict(zip(scan.truth_ids, scan.truth_states, strict=True))
            for z, o in zip(det.z, det.origin, strict=True):
                if o != CLUTTER:
                    res.append(z - radar.measure(states[o], bias=(0.0, 0.0)))
        res = np.array(res)
        res[:, 1] = wrap_angle(res[:, 1])
        sem = np.array([radar.sigma_range, radar.sigma_azimuth]) / np.sqrt(len(res))
        np.testing.assert_array_less(np.abs(res.mean(axis=0) - radar.bias), 4 * sem)
        np.testing.assert_allclose(
            res.std(axis=0), [radar.sigma_range, radar.sigma_azimuth], rtol=0.05
        )


def test_config_has_biased_radar_b(cfg):
    assert np.all(cfg.radar("A").bias == 0.0)
    assert np.all(cfg.radar("B").bias != 0.0)


def test_region_outside_fov_is_rejected(cfg):
    narrow = replace(cfg.radars[0], fov_range=(15000.0, 50000.0))
    with pytest.raises(ValueError, match="field of view"):
        replace(cfg, radars=(narrow,))


def test_plot_scenario_runs(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = simulate(cfg, seed=0)
    ax = plot_scenario(sc)
    assert ax.get_xlabel() == "x (km)"
    plt.close(ax.figure)
    ax = plot_detection_errors(sc)
    plt.close(ax.figure)


def test_detection_errors_with_and_without_bias(long_cfg):
    sc = simulate(long_cfg, seed=6)
    a, b = long_cfg.radar("A"), long_cfg.radar("B")
    # Unbiased radar, or biased radar with its true bias: errors centred on zero.
    assert np.linalg.norm(detection_errors(sc, a).mean(axis=0)) < 10.0
    assert np.linalg.norm(detection_errors(sc, b, bias=b.bias).mean(axis=0)) < 10.0
    # Biased radar with the bias ignored: displaced by hundreds of metres.
    assert np.linalg.norm(detection_errors(sc, b).mean(axis=0)) > 200.0
