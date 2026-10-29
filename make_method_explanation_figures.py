from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from auto_pair_benchmark import candidate_centers, estimate_pair
from benchmark_tracking import grayscale, local_match


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "all_patterns_step1_temporal_v2_20260831"
RESULT_CSV = RESULT_DIR / "all_patterns_velocity_step1_temporal_v2.csv"
OUTPUT_DIR = RESULT_DIR / "method_explanation"
FRAME_DIR = Path(
    r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010164_timestamp_JST_12s"
)
BEFORE_PATH = FRAME_DIR / "GX010164_timestamp_JST_12s0879.jpg"
AFTER_PATH = FRAME_DIR / "GX010164_timestamp_JST_12s0880.jpg"
DATASET = "GX010164_timestamp_JST_12s"
BEFORE_FRAME = 880
RADIUS = 32
SEARCH_X = 150
SEARCH_Y = 55


def set_japanese_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def collect_matches(before: np.ndarray, after: np.ndarray):
    centers = candidate_centers(before, after, radius=RADIUS, count=80)
    records = []
    for x, y, activity in centers:
        try:
            dx, dy, corr, margin = local_match(
                before, after, (x, y), RADIUS, search_x=SEARCH_X, search_y=SEARCH_Y
            )
            rdx, rdy, _, _ = local_match(
                after, before, (x + dx, y + dy), RADIUS,
                search_x=SEARCH_X, search_y=SEARCH_Y,
            )
        except ValueError:
            continue
        round_trip = math.hypot(dx + rdx, dy + rdy)
        trusted = round_trip <= 3.0 and margin >= 0.015
        quality = max(margin, 0.001) * max(corr + 1.0, 0.05) * max(activity, 1e-5)
        records.append({
            "x": x, "y": y, "dx": dx, "dy": dy, "corr": corr,
            "margin": margin, "round_trip": round_trip,
            "quality": quality, "trusted": trusted,
        })
    return centers, pd.DataFrame(records)


def image_coordinates(ax, image: np.ndarray) -> None:
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])


def make_crop_figure(before_rgb, after_rgb, before_gray, matches, centers, estimate, row):
    trusted = matches[matches["trusted"]].copy()
    trusted["distance"] = np.abs(trusted["dx"] - estimate["dx"])
    chosen = trusted.sort_values(["distance", "quality"], ascending=[True, False]).iloc[0]
    x, y = float(chosen.x), float(chosen.y)
    dx, dy = float(chosen.dx), float(chosen.dy)
    mx, my = x + dx, y + dy

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.1], width_ratios=[1.55, 1.0, 1.35])
    ax_full = fig.add_subplot(gs[0, :])
    image_coordinates(ax_full, before_rgb)
    ax_full.set_title(
        "① 画像全体から、模様がありフレーム間で変化した小領域を自動選択\n"
        "Automatically select textured regions that change between frames",
        fontsize=15, weight="bold",
    )
    for cx, cy, _ in centers:
        ax_full.add_patch(Rectangle((cx - 10, cy - 10), 20, 20, fill=False,
                                    edgecolor="#00B8D9", linewidth=0.65, alpha=0.65))
    ax_full.add_patch(Rectangle((x - RADIUS, y - RADIUS), 2 * RADIUS + 1, 2 * RADIUS + 1,
                                fill=False, edgecolor="#FFD400", linewidth=3.0))
    ax_full.text(x + RADIUS + 12, y - RADIUS, "拡大例 / Example crop: 65×65 px", color="#111111",
                 fontsize=12, weight="bold", bbox=dict(facecolor="#FFD400", edgecolor="none", alpha=0.88, pad=3))

    ax_patch = fig.add_subplot(gs[1, 0])
    patch = before_rgb[int(y) - RADIUS:int(y) + RADIUS + 1, int(x) - RADIUS:int(x) + RADIUS + 1]
    image_coordinates(ax_patch, patch)
    ax_patch.scatter([RADIUS], [RADIUS], marker="+", s=170, linewidths=2.0, color="#FFD400")
    ax_patch.set_title(
        "② before側の切り出し画像（テンプレート）\n"
        "Crop from the before frame (template)", fontsize=12, weight="bold",
    )

    ax_search = fig.add_subplot(gs[1, 1:])
    left = max(0, int(x) - RADIUS - SEARCH_X)
    right = min(after_rgb.shape[1], int(x) + RADIUS + SEARCH_X + 1)
    top = max(0, int(y) - RADIUS - SEARCH_Y)
    bottom = min(after_rgb.shape[0], int(y) + RADIUS + SEARCH_Y + 1)
    search_rgb = after_rgb[top:bottom, left:right]
    image_coordinates(ax_search, search_rgb)
    sx, sy = x - left, y - top
    smx, smy = mx - left, my - top
    ax_search.add_patch(Rectangle((smx - RADIUS, smy - RADIUS), 2 * RADIUS + 1, 2 * RADIUS + 1,
                                  fill=False, edgecolor="#FF3B30", linewidth=2.5))
    ax_search.annotate("", xy=(smx, smy), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle="->", color="#FF3B30", lw=2.5))
    ax_search.scatter([sx], [sy], marker="+", s=120, linewidths=1.8, color="#FFD400")
    ax_search.set_title(
        f"③ after側の探索範囲で最も似た位置を探す（Δx={dx:.2f} px, Δy={dy:.2f} px）\n"
        "Find the most similar location within the after-frame search area",
        fontsize=12, weight="bold",
    )
    fig.add_artist(ConnectionPatch(
        xyA=(x + RADIUS, y + RADIUS), coordsA=ax_full.transData,
        xyB=(0, 0), coordsB=ax_patch.transData,
        color="#FFD400", linewidth=1.4, alpha=0.8,
    ))

    scale = float(row["cm_per_pixel"])
    fps = float(row["fps"])
    gap = int(row["frame_gap"])
    velocity = float(row["velocity_final_cm_s"])
    fig.text(
        0.5, 0.012,
        f"上下移動 Δy は使わず、校正軸方向の Δx だけを使用 / "
        f"Use Δx along the calibration axis; ignore vertical Δy\n"
        f"v = Δx × {scale:.6f} cm/px × {fps:.0f} fps ÷ {gap} = {velocity:.2f} cm/s",
        ha="center", va="bottom", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F3F4F6", edgecolor="#9CA3AF"),
    )
    fig.savefig(OUTPUT_DIR / "01_full_image_to_local_match.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chosen


def make_consensus_figure(data, matches, estimate, row):
    trusted = matches[matches["trusted"]].copy()
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.0, 1.35]})

    ax = axes[0]
    positions = np.arange(len(trusted))
    moving = trusted["dx"].abs() >= 2.0
    colors = np.where(moving, "#2563EB", "#9CA3AF")
    ax.scatter(positions, trusted["dx"], c=colors, s=32, alpha=0.78, edgecolors="none")
    ax.axhline(float(estimate["dx"]), color="#DC2626", linewidth=2.2,
               label=f"代表変位 / Representative Δx = {estimate['dx']:.2f} px")
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("信頼できた局所対応の番号 / Trusted local-match index")
    ax.set_ylabel("水平方向の変位 / Horizontal displacement Δx (px)")
    ax.set_title(
        "④ 多数の小領域の変位から、同じ方向へ動く粒子群を選ぶ\n"
        "Select the particle population moving coherently in one direction",
        fontsize=14, weight="bold",
    )
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best")

    ax = axes[1]
    center_time = float(row["time_s"])
    group = data[(data["dataset"] == DATASET) & data["time_s"].between(center_time - 0.55, center_time + 0.55)].copy()
    accepted = group[group["final_status"] == "measured"]
    rejected = group[group["final_status"] != "measured"]
    ax.scatter(accepted["time_s"], accepted["velocity_final_cm_s"], s=18, color="#2563EB",
               alpha=0.78, label="最終採用 / Accepted")
    ax.scatter(rejected["time_s"], rejected["raw_velocity_cm_s"], s=45, marker="x", color="#DC2626",
               linewidths=1.2, label="最終棄却 / Rejected")
    ax.plot(group["time_s"], group["temporal_reference_cm_s"], color="#16A34A", linewidth=2.0,
            alpha=0.9, label="周辺時刻の代表傾向 / Local temporal trend")
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("時刻 / Time (s)")
    ax.set_ylabel("流速 / Velocity (cm/s)")
    ax.set_title(
        "⑤ 前後約0.125秒の傾向で、誤った0付近を除外し妥当な低信頼度点を救済\n"
        "Use the ±0.125 s trend to reject false zeros and rescue plausible low-confidence estimates",
        fontsize=14, weight="bold",
    )
    ax.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best")
    fig.suptitle(
        "局所テンプレート照合＋方向別ロバスト合意＋時間整合性フィルタ\n"
        "Local template matching + directional robust consensus + temporal consistency filter",
        fontsize=17, weight="bold",
    )
    fig.savefig(OUTPUT_DIR / "02_consensus_and_temporal_filter.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    set_japanese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BEFORE_PATH.exists() or not AFTER_PATH.exists():
        raise FileNotFoundError("Example frames are not available on the E drive")
    before_rgb = np.asarray(Image.open(BEFORE_PATH).convert("RGB"))
    after_rgb = np.asarray(Image.open(AFTER_PATH).convert("RGB"))
    before_gray = grayscale(BEFORE_PATH)
    after_gray = grayscale(AFTER_PATH)
    centers, matches = collect_matches(before_gray, after_gray)
    estimate = estimate_pair(before_gray, after_gray, radius=RADIUS, candidate_count=80)
    if estimate is None:
        raise RuntimeError("No representative match found for the example pair")
    data = pd.read_csv(RESULT_CSV)
    row = data[(data["dataset"] == DATASET) & (data["before_frame"] == BEFORE_FRAME)].iloc[0]
    make_crop_figure(before_rgb, after_rgb, before_gray, matches, centers, estimate, row)
    make_consensus_figure(data, matches, estimate, row)
    print(f"trusted_matches={int(matches['trusted'].sum())}")
    print(f"aggregate_dx={estimate['dx']:.6f}")
    print(f"confidence={estimate['confidence']:.6f}")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
