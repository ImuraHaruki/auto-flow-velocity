from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "all_patterns_step1_temporal_v2_20260831"
INPUT_CSV = RESULT_DIR / "all_patterns_velocity_step1_temporal_v2.csv"
OUTPUT_DIR = RESULT_DIR / "accepted_rejected_visualization"
PER_SEQUENCE_DIR = OUTPUT_DIR / "per_sequence"

ACCEPTED_COLOR = "#2563EB"
REJECTED_COLOR = "#DC2626"
GRID_COLOR = "#D1D5DB"


def short_name(name: str) -> str:
    return name.replace("_timestamp_JST", "")


def robust_limits(group: pd.DataFrame) -> tuple[float, float]:
    accepted = group.loc[group["final_status"].eq("measured"), "velocity_final_cm_s"]
    rejected = group.loc[group["final_status"].ne("measured"), "raw_velocity_cm_s"]
    values = pd.concat([accepted, rejected]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return -10.0, 10.0
    low, high = np.nanpercentile(values.to_numpy(float), [0.5, 99.5])
    span = max(float(high - low), 30.0)
    return min(float(low - 0.10 * span), -10.0), max(float(high + 0.10 * span), 10.0)


def plot_series_on_axis(ax: plt.Axes, group: pd.DataFrame, title: str) -> None:
    group = group.sort_values("time_s")
    accepted = group[group["final_status"].eq("measured")]
    rejected = group[group["final_status"].ne("measured")]
    rejected_with_value = rejected[rejected["raw_velocity_cm_s"].notna()]
    y_min, y_max = robust_limits(group)

    ax.scatter(
        accepted["time_s"], accepted["velocity_final_cm_s"],
        s=5, color=ACCEPTED_COLOR, alpha=0.70, edgecolors="none",
        label=f"Accepted: {len(accepted):,}", zorder=2,
    )
    ax.scatter(
        rejected_with_value["time_s"], rejected_with_value["raw_velocity_cm_s"],
        s=15, marker="x", color=REJECTED_COLOR, alpha=0.55, linewidths=0.7,
        label=f"Rejected: {len(rejected):,}", zorder=3,
    )
    if len(rejected):
        rug_y = y_min + 0.025 * (y_max - y_min)
        ax.scatter(
            rejected["time_s"], np.full(len(rejected), rug_y),
            marker="|", s=18, color=REJECTED_COLOR, alpha=0.55, linewidths=0.7,
            zorder=4,
        )
    ax.axhline(0, color="#6B7280", linewidth=0.7)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Velocity (cm/s)", fontsize=8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=7, framealpha=0.92)


def make_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.assign(accepted=data["final_status"].eq("measured"))
        .groupby("dataset", sort=True)
        .agg(total=("accepted", "size"), accepted=("accepted", "sum"))
        .reset_index()
    )
    summary["rejected"] = summary["total"] - summary["accepted"]
    summary["accepted_rate"] = summary["accepted"] / summary["total"]
    summary["rejected_rate"] = summary["rejected"] / summary["total"]
    return summary


def plot_summary(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    ordered = summary.sort_values(["accepted_rate", "dataset"], ascending=[True, True]).reset_index(drop=True)
    labels = [short_name(x) for x in ordered["dataset"]]
    y = np.arange(len(ordered))
    accepted_pct = ordered["accepted_rate"].to_numpy() * 100
    rejected_pct = ordered["rejected_rate"].to_numpy() * 100
    temporal_counts = data["temporal_status"].value_counts()
    status_order = ["measured_base", "rescued_temporal", "rejected_static_lock", "unmeasurable"]
    status_labels = ["Base accepted", "Trend-rescued", "Static-lock rejected", "Unmeasurable"]
    status_colors = [ACCEPTED_COLOR, "#16A34A", "#991B1B", REJECTED_COLOR]
    status_values = [int(temporal_counts.get(s, 0)) for s in status_order]

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    ax.barh(y, accepted_pct, color=ACCEPTED_COLOR, label="Accepted")
    ax.barh(y, rejected_pct, left=accepted_pct, color=REJECTED_COLOR, label="Rejected")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of evaluated frame starts (%)")
    ax.set_title("Accepted and rejected estimates by sequence", fontsize=14, weight="bold")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right")
    for idx, (a, r) in enumerate(zip(accepted_pct, rejected_pct)):
        if r >= 1.2:
            ax.text(a + r / 2, idx, f"{r:.1f}%", ha="center", va="center", fontsize=6.5, color="white")

    ax2 = fig.add_subplot(grid[0, 1])
    ypos = np.arange(len(status_labels))
    ax2.barh(ypos, status_values, color=status_colors)
    ax2.set_yticks(ypos, status_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of estimates")
    ax2.set_title("Decision-stage breakdown", fontsize=14, weight="bold")
    ax2.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    for idx, value in enumerate(status_values):
        ax2.text(value, idx, f" {value:,}", va="center", fontsize=9)

    accepted_total = int(data["final_status"].eq("measured").sum())
    rejected_total = int(len(data) - accepted_total)
    fig.suptitle(
        f"Step-1 velocity estimation: accepted {accepted_total:,} ({accepted_total/len(data):.1%}) / "
        f"rejected {rejected_total:,} ({rejected_total/len(data):.1%})",
        fontsize=17, weight="bold",
    )
    fig.savefig(OUTPUT_DIR / "accepted_rejected_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overview_pages(data: pd.DataFrame) -> None:
    datasets = sorted(data["dataset"].unique())
    for page, start in enumerate(range(0, len(datasets), 10), start=1):
        names = datasets[start:start + 10]
        fig, axes = plt.subplots(5, 2, figsize=(16, 18), constrained_layout=True)
        for ax, dataset in zip(axes.flat, names):
            group = data[data["dataset"].eq(dataset)]
            plot_series_on_axis(ax, group, short_name(dataset))
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        fig.suptitle(
            f"Accepted (blue) and rejected (red) velocity estimates | page {page}",
            fontsize=16, weight="bold",
        )
        fig.savefig(OUTPUT_DIR / f"accepted_rejected_overview_{page:02d}.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_individual_sequences(data: pd.DataFrame) -> None:
    for dataset, group in data.groupby("dataset", sort=True):
        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        plot_series_on_axis(ax, group, f"{dataset} | accepted vs rejected")
        fig.savefig(PER_SEQUENCE_DIR / f"{dataset}_accepted_rejected.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    summary = make_summary(data)
    summary.to_csv(OUTPUT_DIR / "accepted_rejected_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(data, summary)
    plot_overview_pages(data)
    plot_individual_sequences(data)
    print(f"rows={len(data)} datasets={data['dataset'].nunique()}")
    print(f"accepted={data['final_status'].eq('measured').sum()}")
    print(f"rejected={data['final_status'].ne('measured').sum()}")
    print(f"output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
