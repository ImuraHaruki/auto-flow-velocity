from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "auto_pair_results_10.csv"
OUTPUT = ROOT / "auto_pair_preliminary_comparison.png"


def main():
    data = pd.read_csv(INPUT)
    colors = {"right": "#d95f02", "left": "#1b9e77", "slow": "#7570b3"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for direction, group in data.groupby("direction"):
        axes[0].scatter(
            group["manual_velocity"], group["predicted_velocity"],
            s=28, alpha=0.75, color=colors[direction], label=direction,
        )
    lower = float(min(data["manual_velocity"].min(), data["predicted_velocity"].min()))
    upper = float(max(data["manual_velocity"].max(), data["predicted_velocity"].max()))
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].axhline(0, color="#999999", linewidth=0.6)
    axes[0].axvline(0, color="#999999", linewidth=0.6)
    axes[0].set_xlabel("Manual velocity (cm/s)")
    axes[0].set_ylabel("Automatic velocity (cm/s)")
    axes[0].set_title("Manual vs automatic (preliminary)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    order = [dataset.name for dataset in __import__("benchmark_tracking").DATASETS]
    summary = data.groupby("dataset")["abs_error"].agg(["mean", "median"]).reindex(order)
    positions = np.arange(len(summary))
    width = 0.38
    axes[1].bar(positions - width / 2, summary["mean"], width, label="MAE", color="#4c78a8")
    axes[1].bar(positions + width / 2, summary["median"], width, label="Median AE", color="#f2cf5b")
    axes[1].set_xticks(positions, summary.index, rotation=30, ha="right")
    axes[1].set_ylabel("Absolute error (cm/s)")
    axes[1].set_title("Error by sequence")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
