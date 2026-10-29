from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_tracking import DATASETS, image_resolver, list_images, read_rows


ROOT = Path(__file__).resolve().parent
FINE_PATH = ROOT / "fine_results" / "all_fine_results.csv"
OUTPUT_DIR = ROOT / "fine_results" / "manual_comparison"
DISPLAY_NAMES = {
    "A": "GX010164",
    "B": "GX010202",
    "C_GX010101": "GX010101",
    "D_GX010105": "GX010105",
    "E_GX010165": "GX010165",
    "F_GX010213": "GX010213",
}


def nearest_measured(fine: pd.DataFrame, actual_index: int):
    measured = fine[fine["status"] == "measured"]
    if measured.empty:
        return None
    distance = np.abs(measured["before_frame"].to_numpy(dtype=int) - 1 - actual_index)
    position = int(np.argmin(distance))
    if int(distance[position]) > 5:
        return None
    return measured.iloc[position], int(distance[position])


def build_comparison(fine_data: pd.DataFrame):
    records = []
    for dataset in DATASETS:
        images = list_images(dataset.frames)
        resolve = image_resolver(images)
        index_by_path = {path.resolve(): index for index, path in enumerate(images)}
        fine = fine_data[fine_data["dataset"] == dataset.name]
        for row in read_rows(dataset.labels):
            before_path = resolve(row["before_file"], row["before_frame"])
            actual_index = index_by_path[before_path.resolve()]
            match = nearest_measured(fine, actual_index)
            if match is None:
                continue
            automatic, distance = match
            manual = float(row["velocity"])
            predicted = float(automatic["velocity_cm_s"])
            records.append({
                "dataset": dataset.name,
                "manual_before_frame": int(row["before_frame"]),
                "actual_image_index": actual_index,
                "auto_before_frame": int(automatic["before_frame"]),
                "frame_offset": distance,
                "time_s": actual_index / 240.0,
                "manual_velocity": manual,
                "automatic_velocity": predicted,
                "abs_error": abs(predicted - manual),
                "confidence": float(automatic["confidence"]),
                "confidence_label": automatic["confidence_label"],
                "frame_gap": int(automatic["frame_gap"]),
            })
    return pd.DataFrame(records)


def binned_comparison(all_rows: pd.DataFrame):
    grouped = all_rows.groupby(["dataset", "auto_before_frame"], as_index=False).agg(
        time_s=("time_s", "mean"),
        manual_velocity=("manual_velocity", "median"),
        manual_velocity_mean=("manual_velocity", "mean"),
        automatic_velocity=("automatic_velocity", "first"),
        manual_count=("manual_velocity", "size"),
        confidence=("confidence", "first"),
        confidence_label=("confidence_label", "first"),
        frame_gap=("frame_gap", "first"),
        max_frame_offset=("frame_offset", "max"),
    )
    grouped["abs_error"] = np.abs(grouped["automatic_velocity"] - grouped["manual_velocity"])
    return grouped


def metric_row(dataset: str, group: pd.DataFrame):
    correlation = (
        float(np.corrcoef(group["manual_velocity"], group["automatic_velocity"])[0, 1])
        if len(group) > 1 else float("nan")
    )
    sign_rows = group[np.abs(group["manual_velocity"]) > 10]
    sign_ok = int((np.sign(sign_rows["manual_velocity"]) == np.sign(sign_rows["automatic_velocity"])).sum())
    return {
        "dataset": dataset,
        "n_time_bins": len(group),
        "mae": float(group["abs_error"].mean()),
        "median_ae": float(group["abs_error"].median()),
        "p90_ae": float(group["abs_error"].quantile(0.9)),
        "correlation": correlation,
        "sign_ok": sign_ok,
        "sign_n": len(sign_rows),
    }


def padded_limits(group: pd.DataFrame):
    values = np.concatenate([group["manual_velocity"], group["automatic_velocity"]])
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(upper - lower, 20.0)
    return lower - 0.08 * span, upper + 0.08 * span


def make_graphs(
    fine_data: pd.DataFrame,
    all_rows: pd.DataFrame,
    binned: pd.DataFrame,
    summary: pd.DataFrame,
    reliable_summary: pd.DataFrame,
):
    graph_dir = OUTPUT_DIR / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    for dataset, display in DISPLAY_NAMES.items():
        fine = fine_data[(fine_data["dataset"] == dataset) & (fine_data["status"] == "measured")]
        manual = all_rows[all_rows["dataset"] == dataset].sort_values("time_s")
        compared = binned[binned["dataset"] == dataset]
        stats = summary[summary["dataset"] == dataset].iloc[0]
        reliable_stats = reliable_summary[reliable_summary["dataset"] == dataset].iloc[0]
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

        reliable_fine = fine.copy()
        reliable_fine.loc[reliable_fine["confidence"] < 0.85, "velocity_cm_s"] = np.nan
        axes[0].plot(
            reliable_fine["time_s"], reliable_fine["velocity_cm_s"], color="#e07a1f",
            linewidth=1.0, alpha=0.85, label="Automatic (confidence >= 0.85)",
        )
        low = fine[fine["confidence"] < 0.85]
        axes[0].scatter(
            low["time_s"], low["velocity_cm_s"], marker="x", color="#c73e1d",
            s=22, alpha=0.65, label="Low-confidence automatic", zorder=2,
        )
        axes[0].scatter(
            manual["time_s"], manual["manual_velocity"], color="#2878b5",
            s=12, alpha=0.55, label="Manual measurement", zorder=3,
        )
        axes[0].axhline(0, color="#777777", linewidth=0.7)
        axes[0].set_title(f"{display}: time-series comparison")
        axes[0].set_xlabel("Time in current image sequence (s)")
        axes[0].set_ylabel("Velocity (cm/s)")
        axes[0].grid(alpha=0.2)
        axes[0].legend(frameon=False)

        axes[1].scatter(
            compared["manual_velocity"], compared["automatic_velocity"],
            c=compared["confidence"], cmap="viridis", vmin=0.75, vmax=1.0,
            s=32, alpha=0.82,
        )
        lower, upper = padded_limits(compared)
        axes[1].plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axes[1].set_xlim(lower, upper)
        axes[1].set_ylim(lower, upper)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title("Matched 5-frame time bins")
        axes[1].set_xlabel("Manual median velocity (cm/s)")
        axes[1].set_ylabel("Automatic velocity (cm/s)")
        axes[1].grid(alpha=0.2)
        axes[1].text(
            0.03, 0.97,
            f"All n = {int(stats['n_time_bins'])}, MAE = {stats['mae']:.2f}\n"
            f"Reliable n = {int(reliable_stats['n_time_bins'])}, MAE = {reliable_stats['mae']:.2f}\n"
            f"Reliable median AE = {reliable_stats['median_ae']:.2f} cm/s\n"
            f"All r = {stats['correlation']:.3f}",
            transform=axes[1].transAxes, va="top",
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        figure.savefig(graph_dir / f"{display}_fine_vs_manual.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, (dataset, display) in zip(axes.flat, DISPLAY_NAMES.items()):
        fine = fine_data[(fine_data["dataset"] == dataset) & (fine_data["status"] == "measured")]
        manual = all_rows[all_rows["dataset"] == dataset]
        stats = summary[summary["dataset"] == dataset].iloc[0]
        reliable_fine = fine.copy()
        reliable_fine.loc[reliable_fine["confidence"] < 0.85, "velocity_cm_s"] = np.nan
        axis.plot(reliable_fine["time_s"], reliable_fine["velocity_cm_s"], color="#e07a1f", linewidth=0.9, label="Automatic, confidence >= 0.85")
        low = fine[fine["confidence"] < 0.85]
        axis.scatter(low["time_s"], low["velocity_cm_s"], marker="x", color="#c73e1d", s=9, alpha=0.55, label="Low confidence")
        axis.scatter(manual["time_s"], manual["manual_velocity"], color="#2878b5", s=8, alpha=0.5, label="Manual")
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set_title(f"{display} | MAE {stats['mae']:.1f} cm/s, r {stats['correlation']:.3f}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Velocity (cm/s)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    figure.savefig(graph_dir / "all_sequences_fine_vs_manual.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fine_data = pd.read_csv(FINE_PATH)
    all_rows = build_comparison(fine_data)
    binned = binned_comparison(all_rows)
    summary_rows = []
    for dataset in DISPLAY_NAMES:
        summary_rows.append(metric_row(dataset, binned[binned["dataset"] == dataset]))
    summary_rows.append(metric_row("ALL", binned))
    summary = pd.DataFrame(summary_rows)
    reliable = binned[binned["confidence"] >= 0.85]
    reliable_summary_rows = []
    for dataset in DISPLAY_NAMES:
        reliable_summary_rows.append(metric_row(dataset, reliable[reliable["dataset"] == dataset]))
    reliable_summary_rows.append(metric_row("ALL", reliable))
    reliable_summary = pd.DataFrame(reliable_summary_rows)
    all_rows.to_csv(OUTPUT_DIR / "fine_vs_manual_all_rows.csv", index=False, encoding="utf-8-sig")
    binned.to_csv(OUTPUT_DIR / "fine_vs_manual_binned.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "fine_vs_manual_summary.csv", index=False, encoding="utf-8-sig")
    reliable_summary.to_csv(
        OUTPUT_DIR / "fine_vs_manual_summary_confidence_085.csv", index=False, encoding="utf-8-sig"
    )
    make_graphs(fine_data, all_rows, binned, summary, reliable_summary)
    print(summary.to_string(index=False))
    print("\nConfidence >= 0.85")
    print(reliable_summary.to_string(index=False))
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
