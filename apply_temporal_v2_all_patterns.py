from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MANUAL_WORK = ROOT / "manual_series_combined_20260831"
sys.path.insert(0, str(MANUAL_WORK))
from evaluate_temporal_filter import temporal_filter_cluster


UNLABELED_INPUT = ROOT / "all_unlabeled_results_20260829" / "all_unlabeled_velocity.csv"
MANUAL_INPUT = MANUAL_WORK / "manual_series_auto_velocity_combined.csv"
MANUAL_MEASUREMENTS = MANUAL_WORK / "manual_measurements_combined.csv"
OUTPUT_DIR = ROOT / "all_patterns_temporal_v2_20260831"
PER_SEQUENCE_DIR = OUTPUT_DIR / "per_sequence"
GRAPH_DIR = OUTPUT_DIR / "graphs"
OVERVIEW_DIR = OUTPUT_DIR / "overview"

PARAMS = {
    "window": 6,
    "zero_abs": 3.0,
    "ref_motion": 20.0,
    "rescue_conf": 0.75,
    "rescue_abs": 10.0,
    "min_count": 5,
    "dominance": 0.75,
}


def load_inputs() -> pd.DataFrame:
    unlabeled = pd.read_csv(UNLABELED_INPUT)
    labeled = pd.read_csv(MANUAL_INPUT)
    unlabeled["source_group"] = "unlabeled_existing_raw"
    labeled["source_group"] = "manual_series_existing_raw"
    unlabeled["manual_data_available"] = False
    labeled["manual_data_available"] = True
    for frame in (unlabeled, labeled):
        if "fps" not in frame:
            frame["fps"] = 240.0
        if "confidence_threshold" not in frame:
            frame["confidence_threshold"] = 0.85
        frame["original_status"] = frame["status"]
        frame["original_velocity_cm_s"] = frame["velocity_cm_s"]
    combined = pd.concat([unlabeled, labeled], ignore_index=True, sort=False)
    combined = combined.sort_values(["dataset", "before_frame"]).reset_index(drop=True)
    return combined


def apply_filter(data: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for dataset, group in data.groupby("dataset", sort=True):
        filtered = temporal_filter_cluster(group, **PARAMS)
        filtered["dataset"] = dataset
        groups.append(filtered)
    result = pd.concat(groups, ignore_index=True, sort=False)
    result["velocity_final_cm_s"] = result["velocity_temporal_cm_s"]
    result["final_status"] = np.where(result["velocity_final_cm_s"].notna(), "measured", "unmeasurable")
    result["temporal_residual_cm_s"] = np.abs(
        result["raw_velocity_cm_s"] - result["temporal_reference_cm_s"]
    )
    return result.sort_values(["dataset", "before_frame"]).reset_index(drop=True)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in data.groupby("dataset", sort=True):
        final = group["velocity_final_cm_s"].dropna()
        rows.append({
            "dataset": dataset,
            "duration_s": float(group["time_s"].max()),
            "total_candidates": len(group),
            "base_measured": int((group["confidence"] >= 0.85).sum()),
            "rescued_temporal": int((group["temporal_status"] == "rescued_temporal").sum()),
            "rejected_static_lock": int((group["temporal_status"] == "rejected_static_lock").sum()),
            "remaining_unmeasurable": int((group["temporal_status"] == "unmeasurable").sum()),
            "final_measured": int(final.notna().sum()),
            "final_measurement_rate": float(final.notna().sum() / len(group)),
            "median_velocity_cm_s": float(final.median()) if len(final) else np.nan,
            "mean_velocity_cm_s": float(final.mean()) if len(final) else np.nan,
            "min_velocity_cm_s": float(final.min()) if len(final) else np.nan,
            "max_velocity_cm_s": float(final.max()) if len(final) else np.nan,
            "leftward_count_below_minus5": int((final < -5).sum()),
            "near_zero_count_abs_le_5": int((final.abs() <= 5).sum()),
            "rightward_count_above_5": int((final > 5).sum()),
            "rightward_count_above_100": int((final > 100).sum()),
            "manual_data_available": bool(group["manual_data_available"].fillna(False).any()),
        })
    return pd.DataFrame(rows)


def robust_limits(group: pd.DataFrame, manual: pd.DataFrame) -> tuple[float, float]:
    values = [group["velocity_final_cm_s"].dropna().to_numpy(dtype=float)]
    if not manual.empty:
        values.append(manual["velocity"].dropna().to_numpy(dtype=float))
    joined = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([-10, 10])
    low, high = np.nanpercentile(joined, [0.5, 99.5])
    span = max(high - low, 30.0)
    return min(low - 0.12 * span, -10.0), max(high + 0.12 * span, 10.0)


def plot_sequence(group: pd.DataFrame, manual: pd.DataFrame, output: Path) -> None:
    group = group.sort_values("time_s")
    kept = group[group["temporal_status"] == "measured_base"]
    rescued = group[group["temporal_status"] == "rescued_temporal"]
    static = group[group["temporal_status"] == "rejected_static_lock"]
    remaining = group[group["temporal_status"] == "unmeasurable"]
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.scatter(remaining["time_s"], remaining["raw_velocity_cm_s"], marker="x", s=13,
               color="#D1D5DB", alpha=0.42, label=f"Still rejected: {len(remaining)}")
    ax.scatter(kept["time_s"], kept["raw_velocity_cm_s"], s=10,
               color="#2563EB", alpha=0.70, edgecolors="none", label=f"Base accepted: {len(kept)}")
    ax.scatter(rescued["time_s"], rescued["raw_velocity_cm_s"], s=22,
               color="#16A34A", alpha=0.88, edgecolors="white", linewidths=0.25,
               label=f"Trend-rescued: {len(rescued)}", zorder=4)
    ax.scatter(static["time_s"], static["raw_velocity_cm_s"], marker="x", s=45,
               color="#111827", linewidths=1.2, label=f"Static-lock removed: {len(static)}", zorder=5)
    if not manual.empty:
        ax.scatter(manual["time_s"], manual["velocity"], s=25, color="#DC2626", alpha=0.85,
                   edgecolors="white", linewidths=0.3, label=f"Manual: {len(manual)}", zorder=6)
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_ylim(*robust_limits(group, manual))
    ax.set_title(f"{group.iloc[0]['dataset']} | temporal-consistency v2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (cm/s)")
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.65)
    ax.legend(loc="best", fontsize=8, ncols=2, framealpha=0.92)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overviews(data: pd.DataFrame, manual: pd.DataFrame) -> None:
    datasets = sorted(data["dataset"].unique())
    for page, start in enumerate(range(0, len(datasets), 10), start=1):
        names = datasets[start:start + 10]
        fig, axes = plt.subplots(5, 2, figsize=(16, 17), constrained_layout=True)
        for ax, dataset in zip(axes.flat, names):
            group = data[data["dataset"] == dataset].sort_values("time_s")
            accepted = group[group["velocity_final_cm_s"].notna()]
            static = group[group["temporal_status"] == "rejected_static_lock"]
            m = manual[manual["dataset"] == dataset]
            ax.scatter(accepted["time_s"], accepted["velocity_final_cm_s"], s=4,
                       color="#2563EB", alpha=0.68, edgecolors="none")
            ax.scatter(static["time_s"], static["raw_velocity_cm_s"], marker="x", s=18,
                       color="#111827", linewidths=0.8)
            if not m.empty:
                ax.scatter(m["time_s"], m["velocity"], s=8, color="#DC2626", alpha=0.65)
            ax.axhline(0, color="#6B7280", linewidth=0.5)
            ax.set_ylim(*robust_limits(group, m))
            ax.set_title(dataset.replace("_timestamp_JST", ""), fontsize=10)
            ax.set_xlabel("s", fontsize=8)
            ax.set_ylabel("cm/s", fontsize=8)
            ax.grid(True, color="#E5E7EB", linewidth=0.45, alpha=0.7)
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        fig.suptitle(f"All-pattern velocity overview v2 | page {page}", fontsize=15, weight="bold")
        fig.savefig(OVERVIEW_DIR / f"overview_{page:02d}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    for folder in (OUTPUT_DIR, PER_SEQUENCE_DIR, GRAPH_DIR, OVERVIEW_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    raw = load_inputs()
    print(f"Loaded {len(raw)} candidates across {raw['dataset'].nunique()} datasets", flush=True)
    filtered = apply_filter(raw)
    print("Temporal filter applied", flush=True)

    output_csv = OUTPUT_DIR / "all_patterns_velocity_temporal_v2.csv"
    summary_csv = OUTPUT_DIR / "sequence_summary_temporal_v2.csv"
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = build_summary(filtered)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    manual = pd.read_csv(MANUAL_MEASUREMENTS)
    for index, (dataset, group) in enumerate(filtered.groupby("dataset", sort=True), start=1):
        group.to_csv(PER_SEQUENCE_DIR / f"{dataset}_velocity_temporal_v2.csv", index=False, encoding="utf-8-sig")
        plot_sequence(group, manual[manual["dataset"] == dataset], GRAPH_DIR / f"{dataset}_temporal_v2.png")
        if index == 1 or index % 10 == 0 or index == filtered["dataset"].nunique():
            print(f"Graphs {index}/{filtered['dataset'].nunique()}", flush=True)
    plot_overviews(filtered, manual)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "local moving-population temporal consistency v2",
        "parameters": PARAMS,
        "fps": 240.0,
        "sample_step_frames": 5,
        "source_csvs": [str(UNLABELED_INPUT), str(MANUAL_INPUT)],
        "datasets": int(filtered["dataset"].nunique()),
        "rows": int(len(filtered)),
        "excluded_non_sequence_folders": ["phase_analysis_20260831", "spacetime_output"],
        "note": "Existing raw image-pair estimates were reused; only the temporal v2 decision stage was rerun.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(output_csv.resolve(), flush=True)


if __name__ == "__main__":
    main()
