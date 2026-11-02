"""Save a figure of one simulated scenario and of its detection errors.

uv run python -m airpicture.experiments.plot_scenario --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from airpicture.sim import (
    ScenarioConfig,
    load_config,
    plot_detection_errors,
    plot_scenario,
    simulate,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="default: results/figures/scenario_seed<N>.png")
    args = parser.parse_args(argv)

    cfg = ScenarioConfig.from_dict(load_config(args.config)["scenario"])
    scenario = simulate(cfg, seed=args.seed)
    out = Path(args.out or f"results/figures/scenario_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_scene, ax_err) = plt.subplots(
        1, 2, figsize=(12, 6.8), width_ratios=[1.45, 1], layout="constrained"
    )
    plot_scenario(scenario, ax=ax_scene)
    plot_detection_errors(scenario, ax=ax_err)

    ax_scene.set_title(
        f"Seed {scenario.seed}: {len(scenario.truth_tracks())} targets over {cfg.n_scans} scans",
        loc="left",
    )
    ax_err.set_title("Detection position error, bias ignored", loc="left")
    handles, labels = ax_scene.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
