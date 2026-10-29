from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "loso_results" / "loso_predictions.csv"
OUTPUT_DIR = ROOT / "loso_results" / "per_sequence_graphs"
DISPLAY_NAMES = {
    "A": "GX010164",
    "B": "GX010202",
    "C_GX010101": "GX010101",
    "D_GX010105": "GX010105",
    "E_GX010165": "GX010165",
    "F_GX010213": "GX010213",
}


def metrics(group: pd.DataFrame) -> tuple[float, float, float]:
    error = group["predicted_velocity"] - group["manual_velocity"]
    mae = float(np.mean(np.abs(error)))
    median_ae = float(np.median(np.abs(error)))
    correlation = float(np.corrcoef(group["manual_velocity"], group["predicted_velocity"])[0, 1])
    return mae, median_ae, correlation


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = max(upper - lower, 20.0)
    return lower - span * 0.08, upper + span * 0.08


def draw_time_series(axis, group: pd.DataFrame, compact: bool = False):
    ordered = group.sort_values("before_frame")
    axis.plot(
        ordered["before_frame"], ordered["manual_velocity"],
        marker="o", markersize=3.5 if compact else 4.5, linewidth=1.1,
        color="#2878b5", label="Manual measurement",
    )
    axis.plot(
        ordered["before_frame"], ordered["predicted_velocity"],
        marker="s", markersize=3.0 if compact else 4.0, linewidth=1.0,
        linestyle="--", color="#e07a1f", label="Automatic estimate",
    )
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set_xlabel("Before-frame number")
    axis.set_ylabel("Velocity (cm/s)")
    axis.grid(alpha=0.22)


def save_individual(dataset: str, group: pd.DataFrame, output_dir: Path):
    display = DISPLAY_NAMES[dataset]
    mae, median_ae, correlation = metrics(group)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    draw_time_series(axes[0], group)
    axes[0].set_title(f"{display}: frame-by-frame comparison")
    axes[0].legend(frameon=False)

    colors = group["direction"].map({"right": "#d95f02", "left": "#1b9e77", "slow": "#7570b3"})
    axes[1].scatter(
        group["manual_velocity"], group["predicted_velocity"],
        c=colors, s=36, alpha=0.82,
    )
    combined = np.concatenate([group["manual_velocity"], group["predicted_velocity"]])
    lower, upper = padded_limits(combined)
    axes[1].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Manual velocity (cm/s)")
    axes[1].set_ylabel("Automatic velocity (cm/s)")
    axes[1].set_title("Point-by-point agreement")
    axes[1].grid(alpha=0.22)
    axes[1].text(
        0.03, 0.97,
        f"MAE = {mae:.2f} cm/s\nMedian AE = {median_ae:.2f} cm/s\nr = {correlation:.3f}",
        transform=axes[1].transAxes, va="top", ha="left",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
    )
    path = output_dir / f"{display}_manual_vs_auto.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_overview(data: pd.DataFrame, output_dir: Path):
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DISPLAY_NAMES):
        group = data[data["dataset"] == dataset]
        draw_time_series(axis, group, compact=True)
        mae, _, correlation = metrics(group)
        axis.set_title(f"{DISPLAY_NAMES[dataset]}  |  MAE {mae:.2f} cm/s, r {correlation:.3f}")
    axes.flat[0].legend(frameon=False, loc="best")
    path = output_dir / "all_sequences_manual_vs_auto.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset in DISPLAY_NAMES:
        paths.append(save_individual(dataset, data[data["dataset"] == dataset], args.output_dir))
    paths.append(save_overview(data, args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
