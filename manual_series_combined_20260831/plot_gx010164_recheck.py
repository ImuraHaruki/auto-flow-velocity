from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"C:\Users\haruh\Documents\Codex\flow_velocity_gui\auto_velocity_prototype\manual_series_combined_20260831")
AUTO_CSV = ROOT / "manual_series_auto_velocity_combined.csv"
MANUAL_CSV = ROOT / "manual_measurements_combined.csv"
OUTPUT = ROOT / "GX010164_auto_manual_recheck.png"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


auto_all = read_csv(AUTO_CSV)
manual_all = read_csv(MANUAL_CSV)

series = [
    ("GX010164_timestamp_JST", "GX010164 (full sequence)"),
    ("GX010164_timestamp_JST_12s", "GX010164 (12s sequence)"),
]


def subset(rows, dataset):
    return [r for r in rows if r["dataset"] == dataset]


def numbers(rows, x_key, y_key):
    x, y = [], []
    for row in rows:
        if row.get(x_key, "") == "" or row.get(y_key, "") == "":
            continue
        x.append(float(row[x_key]))
        y.append(float(row[y_key]))
    return np.asarray(x), np.asarray(y)


def plot_series(ax, dataset, title, xlim=None):
    auto = subset(auto_all, dataset)
    manual = subset(manual_all, dataset)
    reliable = [r for r in auto if r["status"] == "measured"]
    rejected = [r for r in auto if r["status"] != "measured"]

    xr, yr = numbers(reliable, "time_s", "velocity_cm_s")
    xb, yb = numbers(rejected, "time_s", "raw_velocity_cm_s")
    xm, ym = numbers(manual, "time_s", "velocity")

    ax.scatter(xb, yb, s=13, marker="x", linewidths=0.7, color="#9CA3AF", alpha=0.55,
               label=f"Rejected auto: {len(rejected)}")
    ax.scatter(xr, yr, s=12, color="#2563EB", alpha=0.72, edgecolors="none",
               label=f"Reliable auto: {len(reliable)}")
    ax.scatter(xm, ym, s=28, color="#DC2626", alpha=0.9, edgecolors="white", linewidths=0.35,
               label=f"Manual: {len(manual)}", zorder=3)
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (cm/s)")
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.65)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)


plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

plot_series(fig.add_subplot(gs[0, 0]), series[0][0], "A. Full 27.5 s coverage")
plot_series(fig.add_subplot(gs[0, 1]), series[0][0], "B. Full-sequence manual interval (zoom)", (0.12, 0.32))
plot_series(fig.add_subplot(gs[1, 0]), series[1][0], "C. 12s-sequence coverage")

ax = fig.add_subplot(gs[1, 1])
labels = ["164 full", "164 12s"]
candidate_counts = []
reliable_counts = []
rejected_counts = []
manual_counts = []
for dataset, _ in series:
    auto = subset(auto_all, dataset)
    manual = subset(manual_all, dataset)
    reliable = sum(r["status"] == "measured" for r in auto)
    candidate_counts.append(len(auto))
    reliable_counts.append(reliable)
    rejected_counts.append(len(auto) - reliable)
    manual_counts.append(len(manual))

x = np.arange(len(labels))
w = 0.23
bars1 = ax.bar(x - 1.5*w, candidate_counts, w, label="Auto candidates", color="#93C5FD")
bars2 = ax.bar(x - 0.5*w, reliable_counts, w, label="Reliable auto", color="#2563EB")
bars3 = ax.bar(x + 0.5*w, rejected_counts, w, label="Rejected auto", color="#9CA3AF")
bars4 = ax.bar(x + 1.5*w, manual_counts, w, label="Manual", color="#DC2626")
for bars in (bars1, bars2, bars3, bars4):
    ax.bar_label(bars, padding=2, fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("Number of measurements")
ax.set_title("D. Measurement counts", fontsize=12, weight="bold")
ax.legend(fontsize=8, ncols=2)
ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
ax.grid(axis="x", visible=False)

fig.suptitle(
    "GX010164 automatic-measurement coverage check\n"
    "Auto sampling: every 5 source frames (240 fps = 0.0208 s); reliable threshold: confidence >= 0.85",
    fontsize=15,
    weight="bold",
)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
print(OUTPUT)
