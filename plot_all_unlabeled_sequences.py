from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_name(dataset: str) -> str:
    return dataset.replace("_timestamp_JST", "")


def plot_individual(group: pd.DataFrame, output: Path):
    dataset = str(group["dataset"].iloc[0])
    measured = group[group["status"] == "measured"].copy()
    low = group[group["status"] != "measured"].copy()
    rate = len(measured) / len(group)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(
        measured["time_s"], measured["velocity_cm_s"],
        color="#e07a1f", linewidth=1.0, label="Measured (confidence >= 0.85)",
    )
    axes[0].scatter(
        low["time_s"], low["raw_velocity_cm_s"],
        marker="x", color="#c73e1d", s=16, alpha=0.65, label="Low confidence / unmeasurable",
    )
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("Velocity (cm/s)")
    axes[0].set_title(
        f"{display_name(dataset)} | measured {len(measured)}/{len(group)} ({100 * rate:.1f}%)"
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(group["time_s"], group["confidence"], color="#2878b5", linewidth=0.9)
    axes[1].axhline(0.85, color="#c73e1d", linestyle="--", linewidth=0.9, label="Threshold 0.85")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Confidence")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="lower right")

    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_overview(data: pd.DataFrame, output: Path):
    datasets = sorted(data["dataset"].unique())
    cols = 5
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, dataset in zip(axes, datasets):
        group = data[data["dataset"] == dataset]
        measured = group[group["status"] == "measured"]
        low = group[group["status"] != "measured"]
        rate = len(measured) / len(group)
        axis.plot(measured["time_s"], measured["velocity_cm_s"], color="#e07a1f", linewidth=0.75)
        axis.scatter(
            low["time_s"], low["raw_velocity_cm_s"], marker="x",
            color="#c73e1d", s=7, alpha=0.55,
        )
        axis.axhline(0, color="#777777", linewidth=0.5)
        axis.set_title(f"{display_name(dataset)} | {100 * rate:.1f}%", fontsize=9)
        axis.set_xlabel("Time (s)", fontsize=8)
        axis.set_ylabel("cm/s", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.18)
    for axis in axes[len(datasets):]:
        axis.axis("off")
    fig.suptitle(
        "Automatic velocity estimates for sequences without particle-manual CSV\n"
        "Orange: confidence >= 0.85, red x: low confidence",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_rates(summary: pd.DataFrame, output: Path):
    summary = summary.sort_values("measurement_rate", ascending=True)
    colors = ["#c73e1d" if rate < 0.90 else "#e07a1f" if rate < 0.95 else "#2878b5"
              for rate in summary["measurement_rate"]]
    fig, axis = plt.subplots(figsize=(10, 12), constrained_layout=True)
    y = np.arange(len(summary))
    axis.barh(y, summary["measurement_rate"] * 100, color=colors)
    axis.set_yticks(y, [display_name(name) for name in summary["dataset"]], fontsize=8)
    axis.set_xlim(75, 101)
    axis.axvline(95, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Measurement rate (%)")
    axis.set_title("Measurement coverage by image sequence")
    axis.grid(axis="x", alpha=0.2)
    for index, value in enumerate(summary["measurement_rate"] * 100):
        axis.text(value + 0.2, index, f"{value:.1f}%", va="center", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.results_dir / "all_unlabeled_velocity.csv")
    summary = pd.read_csv(args.results_dir / "sequence_summary.csv")
    graph_dir = args.results_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    for dataset, group in data.groupby("dataset", sort=True):
        plot_individual(group, graph_dir / f"{display_name(dataset)}_auto_velocity.png")
    plot_overview(data, graph_dir / "all_43_sequences_overview.png")
    plot_measurement_rates(summary, graph_dir / "measurement_rate_by_sequence.png")
    print(f"Created {len(summary) + 2} graphs in {graph_dir.resolve()}")


if __name__ == "__main__":
    main()
