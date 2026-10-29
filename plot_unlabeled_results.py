from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "unlabeled_results" / "all_unlabeled_results.csv"
OUTPUT_DIR = ROOT / "unlabeled_results" / "graphs"
DISPLAY_NAMES = {
    "A": "GX010164",
    "B": "GX010202",
    "C_GX010101": "GX010101",
    "D_GX010105": "GX010105",
    "E_GX010165": "GX010165",
    "F_GX010213": "GX010213",
}
COLORS = {"high": "#2878b5", "medium": "#e07a1f", "low": "#c73e1d"}


def draw(axis, group: pd.DataFrame, title: str):
    ordered = group.sort_values("time_s")
    measured = ordered[ordered["status"] == "measured"]
    axis.plot(
        measured["time_s"], measured["velocity_cm_s"],
        color="#9a9a9a", linewidth=0.8, alpha=0.65, zorder=1,
    )
    for label in ("high", "medium", "low"):
        points = measured[measured["confidence_label"] == label]
        axis.scatter(
            points["time_s"], points["velocity_cm_s"],
            s=24, color=COLORS[label], label=label.capitalize(), zorder=2,
        )
    rejected = ordered[ordered["status"] != "measured"]
    if not rejected.empty:
        axis.scatter(
            rejected["time_s"], rejected["raw_velocity_cm_s"],
            marker="x", s=35, color="#555555", label="Unmeasurable", zorder=3,
        )
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Velocity (cm/s)")
    axis.grid(alpha=0.2)


def main():
    data = pd.read_csv(INPUT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DISPLAY_NAMES):
        group = data[data["dataset"] == dataset]
        measured = group[group["status"] == "measured"]
        median = float(measured["velocity_cm_s"].median())
        draw(
            axis, group,
            f"{DISPLAY_NAMES[dataset]}  |  measured {len(measured)}/{len(group)}, median {median:.1f} cm/s",
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=4, frameon=False)
    overview = OUTPUT_DIR / "all_unlabeled_velocity_timeseries.png"
    figure.savefig(overview, dpi=180, bbox_inches="tight")
    plt.close(figure)

    for dataset, display in DISPLAY_NAMES.items():
        group = data[data["dataset"] == dataset]
        figure, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        draw(axis, group, f"{display}: automatic velocity at previously unmeasured times")
        axis.legend(frameon=False, ncol=4)
        figure.savefig(OUTPUT_DIR / f"{display}_unlabeled_timeseries.png", dpi=180)
        plt.close(figure)
    print(overview)


if __name__ == "__main__":
    main()
