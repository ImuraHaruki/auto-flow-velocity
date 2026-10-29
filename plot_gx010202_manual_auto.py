from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = "GX010202_timestamp_JST"
AUTO_CSV = ROOT / "all_patterns_step1_temporal_v2_20260831" / "all_patterns_velocity_step1_temporal_v2.csv"
MANUAL_CSV = ROOT / "manual_series_combined_20260831" / "manual_measurements_combined.csv"
OUTPUT_DIR = ROOT / "all_patterns_step1_temporal_v2_20260831" / "manual_auto_comparison"
OUTPUT_PNG = OUTPUT_DIR / "GX010202_manual_vs_auto.png"


def set_japanese_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_japanese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manual = pd.read_csv(MANUAL_CSV)
    manual = manual[manual["dataset"].eq(DATASET)].sort_values("time_s").copy()
    auto = pd.read_csv(AUTO_CSV)
    auto = auto[auto["dataset"].eq(DATASET)].sort_values("time_s").copy()
    if manual.empty or auto.empty:
        raise RuntimeError("GX010202 data are missing")

    manual["frame_key"] = np.rint(manual["time_s"] * 240).astype(int)
    auto["frame_key"] = np.rint(auto["time_s"] * 240).astype(int)
    comparison = manual[["frame_key", "time_s", "velocity"]].merge(
        auto[["frame_key", "velocity_final_cm_s", "final_status", "confidence"]],
        on="frame_key", how="left", validate="one_to_one",
    )
    valid = comparison.dropna(subset=["velocity", "velocity_final_cm_s"]).copy()
    error = valid["velocity_final_cm_s"] - valid["velocity"]
    mae = float(error.abs().mean())
    medae = float(error.abs().median())
    rmse = float(np.sqrt(np.mean(error.to_numpy() ** 2)))
    correlation = float(valid[["velocity", "velocity_final_cm_s"]].corr().iloc[0, 1])

    display_auto = auto[auto["time_s"].between(manual["time_s"].min(), manual["time_s"].max())].copy()
    display_auto["rolling_median"] = display_auto["velocity_final_cm_s"].rolling(
        13, center=True, min_periods=3
    ).median()
    accepted = display_auto[display_auto["final_status"].eq("measured")]
    rejected = display_auto[display_auto["final_status"].ne("measured")]

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(
        accepted["time_s"], accepted["velocity_final_cm_s"],
        s=9, color="#2563EB", alpha=0.48, edgecolors="none",
        label=f"自動推定（採用） / Automatic accepted (n={len(accepted):,})",
    )
    ax.plot(
        display_auto["time_s"], display_auto["rolling_median"],
        color="#0F3D91", linewidth=2.0,
        label="自動推定13フレーム中央値 / Automatic 13-frame median",
    )
    ax.plot(
        manual["time_s"], manual["velocity"],
        color="#DC2626", linewidth=1.2, alpha=0.78,
        label=f"手動計測 / Manual measurement (n={len(manual):,})",
    )
    ax.scatter(
        manual["time_s"], manual["velocity"],
        s=10, color="#DC2626", alpha=0.62, edgecolors="none",
    )
    if len(rejected):
        all_values = pd.concat([manual["velocity"], accepted["velocity_final_cm_s"]]).dropna()
        low, high = np.nanpercentile(all_values, [0.5, 99.5])
        rug_y = float(low - 0.06 * max(high - low, 20.0))
        ax.scatter(
            rejected["time_s"], np.full(len(rejected), rug_y),
            marker="|", s=26, color="#6B7280", alpha=0.65,
            label=f"自動棄却時刻 / Automatic rejected (n={len(rejected):,})",
        )
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("時刻 / Time (s)")
    ax.set_ylabel("流速 / Velocity (cm/s)")
    ax.set_title("GX010202 手動計測と自動推定の時系列比較\nManual and automatic velocity time series", fontsize=15, weight="bold")
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=9, framealpha=0.94)

    ax = fig.add_subplot(grid[1, 0])
    ax.scatter(
        valid["velocity"], valid["velocity_final_cm_s"],
        s=13, color="#2563EB", alpha=0.48, edgecolors="none",
    )
    joined = pd.concat([valid["velocity"], valid["velocity_final_cm_s"]]).dropna().to_numpy(float)
    low, high = np.nanpercentile(joined, [0.5, 99.5])
    span = max(float(high - low), 20.0)
    lo, hi = float(low - 0.08 * span), float(high + 0.08 * span)
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.2, linestyle="--", label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("手動計測流速 / Manual velocity (cm/s)")
    ax.set_ylabel("自動推定流速 / Automatic velocity (cm/s)")
    ax.set_title("同一時刻の対応点 / Matched observations at the same frame start", fontsize=13, weight="bold")
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right")
    ax.text(
        0.02, 0.98,
        f"n = {len(valid):,}\nMAE = {mae:.2f} cm/s\nMedian AE = {medae:.2f} cm/s\n"
        f"RMSE = {rmse:.2f} cm/s\nPearson r = {correlation:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#9CA3AF", alpha=0.94),
    )

    fig.savefig(OUTPUT_PNG, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"manual={len(manual)} auto_interval={len(display_auto)} matched_valid={len(valid)}")
    print(f"mae={mae:.6f} median_ae={medae:.6f} rmse={rmse:.6f} r={correlation:.6f}")
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
