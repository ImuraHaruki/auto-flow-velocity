from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_temporal_filter import (
    AUTO_PATH, MANUAL_PATH, evaluate, manual_targets, temporal_filter_cluster,
)


PARAMS = {
    "window": 6,          # +/- 6 sampling points = +/- 0.125 s at step 5, 240 fps
    "zero_abs": 3.0,     # candidate static-lock zone (cm/s)
    "ref_motion": 20.0,  # surrounding flow must be clearly moving (cm/s)
    "rescue_conf": 0.75,
    "rescue_abs": 10.0,  # allowed difference from interpolated trend (cm/s)
    "min_count": 5,
    "dominance": 0.75,
}
OUTPUT_CSV = HERE / "manual_series_auto_velocity_temporal_v2.csv"
SUMMARY_CSV = HERE / "temporal_filter_summary_v2.csv"
GRAPH_PATH = HERE / "GX010164_temporal_filter_comparison_v2.png"
PEAK_GRAPH_PATH = HERE / "GX010164_peak_zero_zoom_v2.png"


def make_graph(filtered: pd.DataFrame, manual: pd.DataFrame) -> None:
    datasets = [
        ("GX010164_timestamp_JST", "GX010164 full"),
        ("GX010164_timestamp_JST_12s", "GX010164 12s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="row", sharey="row", constrained_layout=True)
    for row_index, (dataset, label) in enumerate(datasets):
        group = filtered[filtered["dataset"] == dataset].sort_values("time_s")
        m = manual[manual["dataset"] == dataset]
        base = group[group["confidence"] >= 0.85]
        low = group[group["confidence"] < 0.85]

        ax = axes[row_index, 0]
        ax.scatter(low["time_s"], low["raw_velocity_cm_s"], marker="x", s=16,
                   color="#9CA3AF", alpha=0.55, label=f"Rejected low confidence: {len(low)}")
        ax.scatter(base["time_s"], base["raw_velocity_cm_s"], s=11,
                   color="#2563EB", alpha=0.72, edgecolors="none", label=f"Accepted: {len(base)}")
        ax.scatter(m["time_s"], m["velocity"], s=25, color="#DC2626", alpha=0.85,
                   edgecolors="white", linewidths=0.3, label=f"Manual: {len(m)}", zorder=5)
        ax.set_title(f"{label}: confidence-only")

        ax = axes[row_index, 1]
        kept = group[group["temporal_status"] == "measured_base"]
        rescued = group[group["temporal_status"] == "rescued_temporal"]
        static = group[group["temporal_status"] == "rejected_static_lock"]
        remaining = group[group["temporal_status"] == "unmeasurable"]
        ax.scatter(remaining["time_s"], remaining["raw_velocity_cm_s"], marker="x", s=13,
                   color="#D1D5DB", alpha=0.45, label=f"Still rejected: {len(remaining)}")
        ax.scatter(kept["time_s"], kept["raw_velocity_cm_s"], s=11,
                   color="#2563EB", alpha=0.70, edgecolors="none", label=f"Base accepted: {len(kept)}")
        ax.scatter(rescued["time_s"], rescued["raw_velocity_cm_s"], s=23,
                   color="#16A34A", alpha=0.9, edgecolors="white", linewidths=0.25,
                   label=f"Trend-rescued: {len(rescued)}", zorder=4)
        ax.scatter(static["time_s"], static["raw_velocity_cm_s"], marker="x", s=48,
                   color="#111827", linewidths=1.3, label=f"Static-lock removed: {len(static)}", zorder=6)
        ax.scatter(m["time_s"], m["velocity"], s=25, color="#DC2626", alpha=0.85,
                   edgecolors="white", linewidths=0.3, label=f"Manual: {len(m)}", zorder=5)
        ax.set_title(f"{label}: temporal consistency added")

        for ax in axes[row_index]:
            ax.axhline(0, color="#6B7280", linewidth=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity (cm/s)")
            ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.65)
            ax.legend(loc="best", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "GX010164: confidence-only vs temporal-consistency filtering\n"
        "Green = rescued because it follows both neighboring measurements; black x = near-zero static lock removed",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(GRAPH_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_peak_graph(filtered: pd.DataFrame, manual: pd.DataFrame) -> None:
    dataset = "GX010164_timestamp_JST_12s"
    group = filtered[(filtered["dataset"] == dataset) & filtered["time_s"].between(5.4, 7.5)].copy()
    m = manual[(manual["dataset"] == dataset) & manual["time_s"].between(5.4, 7.5)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharex=True, sharey=True, constrained_layout=True)

    base = group[group["confidence"] >= 0.85]
    low = group[group["confidence"] < 0.85]
    axes[0].scatter(low["time_s"], low["raw_velocity_cm_s"], marker="x", s=24,
                    color="#9CA3AF", alpha=0.65, label="Low confidence")
    axes[0].scatter(base["time_s"], base["raw_velocity_cm_s"], s=20,
                    color="#2563EB", alpha=0.78, edgecolors="none", label="Accepted")
    axes[0].scatter(m["time_s"], m["velocity"], s=32, color="#DC2626", alpha=0.85,
                    edgecolors="white", linewidths=0.3, label="Manual", zorder=5)
    axes[0].set_title("Before: confidence-only")

    kept = group[group["temporal_status"] == "measured_base"]
    rescued = group[group["temporal_status"] == "rescued_temporal"]
    static = group[group["temporal_status"] == "rejected_static_lock"]
    remaining = group[group["temporal_status"] == "unmeasurable"]
    axes[1].scatter(remaining["time_s"], remaining["raw_velocity_cm_s"], marker="x", s=18,
                    color="#D1D5DB", alpha=0.45, label="Still rejected")
    axes[1].scatter(kept["time_s"], kept["raw_velocity_cm_s"], s=20,
                    color="#2563EB", alpha=0.75, edgecolors="none", label="Base accepted")
    axes[1].scatter(rescued["time_s"], rescued["raw_velocity_cm_s"], s=34,
                    color="#16A34A", alpha=0.9, edgecolors="white", linewidths=0.25,
                    label="Trend-rescued", zorder=4)
    axes[1].scatter(static["time_s"], static["raw_velocity_cm_s"], marker="x", s=70,
                    color="#111827", linewidths=1.6, label=f"Static-lock removed: {len(static)}", zorder=6)
    axes[1].scatter(m["time_s"], m["velocity"], s=32, color="#DC2626", alpha=0.85,
                    edgecolors="white", linewidths=0.3, label="Manual", zorder=5)
    axes[1].set_title("After: local moving-population consistency")

    for ax in axes:
        ax.axhline(0, color="#6B7280", linewidth=0.8)
        ax.set_xlim(5.4, 7.5)
        ax.set_ylim(-25, 365)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (cm/s)")
        ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.65)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.suptitle("GX010164 12s: peak-region zero-lock check", fontsize=15, weight="bold")
    fig.savefig(PEAK_GRAPH_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    auto = pd.read_csv(AUTO_PATH)
    manual = pd.read_csv(MANUAL_PATH)
    targets = manual_targets(auto, manual)
    filtered, validation = evaluate(auto, targets, PARAMS, filter_func=temporal_filter_cluster)
    filtered["velocity_final_cm_s"] = filtered["velocity_temporal_cm_s"]
    filtered["final_status"] = np.where(filtered["velocity_final_cm_s"].notna(), "measured", "unmeasurable")
    filtered["temporal_residual_cm_s"] = np.abs(
        filtered["raw_velocity_cm_s"] - filtered["temporal_reference_cm_s"]
    )
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    status = filtered.groupby(["dataset", "temporal_status"]).size().unstack(fill_value=0)
    summary = []
    for dataset, group in filtered.groupby("dataset"):
        counts = status.loc[dataset]
        summary.append({
            "dataset": dataset,
            "total_candidates": len(group),
            "base_measured": int((group["confidence"] >= 0.85).sum()),
            "rescued_temporal": int(counts.get("rescued_temporal", 0)),
            "rejected_static_lock": int(counts.get("rejected_static_lock", 0)),
            "final_measured": int(group["velocity_final_cm_s"].notna().sum()),
        })
    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    make_graph(filtered, manual)
    make_peak_graph(filtered, manual)
    print(validation.to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(OUTPUT_CSV)
    print(SUMMARY_CSV)
    print(GRAPH_PATH)
    print(PEAK_GRAPH_PATH)


if __name__ == "__main__":
    main()
