# Modelling assumptions

Choices made where the project description does not fix the model. Each one is
marked in the code with an `# ASSUMPTION:` comment where it applies. All numeric
defaults live in `configs/benchmark.yaml`.

## Coordinates and models

- 2-D Cartesian plane in metres. State `[px, py, vx, vy]`.
- Azimuth is measured counter-clockwise from the +x axis (mathematical
  convention), wrapped to (-pi, pi]. A navigation convention (clockwise from
  north) would only change `Radar.measure` and `Radar.to_cartesian`.
- Motion: nearly-constant-velocity model with continuous white-noise
  acceleration, power spectral density 1 m^2/s^3 per axis. Over a 5 s scan this
  gives about 6.5 m of position noise and 2.2 m/s of velocity noise; no
  coordinated turns.
- Measurement: `z = [range + range_bias, azimuth + azimuth_bias] + w`, with
  `w ~ N(0, diag(sigma_r^2, sigma_az^2))`. The bias is additive and constant
  over the scenario. Radar A has zero bias; Radar B has 300 m and 5 mrad.
- Noise: 50 m in range and 1 mrad in azimuth for both radars. At 60 km, 1 mrad
  is 60 m of cross-range error, so Radar B's bias (roughly 300 m radial and
  300 m cross-range) is several standard deviations. This is large enough for
  a tracker that ignores the bias to form duplicate tracks.

## Scenario

- Both radars report at the same scan times, every 5 s, for 60 scans.
- Surveillance region: x in [-40, 40] km, y in [20, 100] km. Radars at
  (-20, 0) km and (20, 0) km, south of the region, so azimuth never wraps.
- Each radar's field of view is a box in (range, azimuth) that contains the
  region; the config loader rejects a region outside the field of view.
- Births: Poisson, mean 0.1 per scan, plus 4 targets at scan 0. A new target
  appears uniformly inside the region with a uniform heading and a speed
  uniform in [150, 250] m/s. Births anywhere in the region (not only at the
  edges) give a standard uniform birth intensity, which is easy to match in
  the trackers.
- Deaths: each target survives a scan with probability 0.99 and dies when it
  leaves the region.
- Detection: each target is detected by each radar independently with
  probability 0.9. Detections that noise pushes outside the field of view are
  dropped.
- Clutter: Poisson, mean 10 per scan per radar, uniform over the field of view
  in (range, azimuth). The clutter intensity in measurement space is therefore
  constant, `clutter_rate / fov_volume`.
- Random streams: truth and each radar use independent streams spawned from
  the scenario seed, so changing a radar setting (for example, removing the
  bias for the oracle) leaves the trajectories unchanged.

## Metrics

- OSPA with cut-off c = 1000 m and order p = 2 on 2-D positions. c is
  larger than Radar B's bias displacement, so a duplicate track is costed as a
  cardinality error, not hidden as a localisation error.
- Position NEES uses the 2x2 position block of the track covariance; the ideal
  mean is 2.
- Duplicate tracks: each estimated track is assigned to its nearest true
  target if it is within 1000 m; the duplicate count is the sum over targets
  of (number of assigned tracks - 1) for targets with at least one.
- Cardinality error is signed (estimated minus true), so over- and
  under-counting can be told apart; report the mean absolute value as well.
