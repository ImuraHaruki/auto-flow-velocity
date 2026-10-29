from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image

from auto_pair_benchmark import candidate_centers, estimate_from_matches
from benchmark_tracking import grayscale, local_match


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "all_patterns_step1_temporal_v2_20260831"
OUTPUT_DIR = RESULT_DIR / "method_explanation"
FRAME_DIR = Path(
    r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames"
    r"\GX010202_timestamp_JST"
)
BEFORE_PATH = FRAME_DIR / "GX010202_timestamp_JST_11082.jpg"
AFTER_PATH = FRAME_DIR / "GX010202_timestamp_JST_11083.jpg"
OUTPUT_PATH = OUTPUT_DIR / "00_tracking_candidates_GX010202_zoom_location.png"
PATCH_RADIUS = 32
PATCH_SIZE = 2 * PATCH_RADIUS + 1
COLOR = "#00C8E8"
REJECTED_COLOR = "#FF9F1C"
CANDIDATE_COUNT = 80


def set_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before_rgb = np.asarray(Image.open(BEFORE_PATH).convert("RGB"))
    before_gray = grayscale(BEFORE_PATH)
    after_gray = grayscale(AFTER_PATH)
    # Reproduce the exact 80 candidate centers used by the saved measurement.
    candidates = candidate_centers(
        before_gray, after_gray, radius=PATCH_RADIUS, count=CANDIDATE_COUNT
    )
    accepted: set[tuple[int, int]] = set()
    destinations = []
    matches: list[tuple[float, float]] = []
    for x, y, activity in candidates:
        try:
            dx, dy, correlation, margin = local_match(
                before_gray, after_gray, (x, y), PATCH_RADIUS,
                search_x=150, search_y=55,
            )
            reverse_dx, reverse_dy, _, _ = local_match(
                after_gray, before_gray, (x + dx, y + dy), PATCH_RADIUS,
                search_x=150, search_y=55,
            )
        except ValueError:
            continue
        round_trip = math.hypot(dx + reverse_dx, dy + reverse_dy)
        if round_trip > 3.0 or margin < 0.015:
            continue
        quality = max(margin, 0.001) * max(correlation + 1.0, 0.05) * max(activity, 1e-5)
        matches.append((dx, quality))
        accepted.add((x, y))
        destinations.append((x, y, x + dx, y + dy))
    estimate = estimate_from_matches(matches)
    if estimate is None:
        raise RuntimeError("The reproduced pair unexpectedly became unmeasurable")

    fig, ax = plt.subplots(figsize=(16, 9.5), constrained_layout=True)
    ax.imshow(before_rgb)
    ax.set_xlim(0, before_rgb.shape[1])
    ax.set_ylim(before_rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for x, y, _ in candidates:
        is_accepted = (x, y) in accepted
        edge_color = COLOR if is_accepted else REJECTED_COLOR
        ax.add_patch(
            Rectangle(
                (x - PATCH_RADIUS, y - PATCH_RADIUS),
                PATCH_SIZE,
                PATCH_SIZE,
                facecolor=(0.0, 0.78, 0.91, 0.08) if is_accepted else (1.0, 0.62, 0.11, 0.035),
                edgecolor=edge_color,
                linewidth=1.6 if is_accepted else 1.1,
                linestyle="-" if is_accepted else "--",
            )
        )

    ax.add_patch(Rectangle(
        (1118 - PATCH_RADIUS, 812 - PATCH_RADIUS), PATCH_SIZE, PATCH_SIZE,
        facecolor='none', edgecolor='white', linewidth=6, zorder=9,
    ))
    ax.add_patch(Rectangle(
        (1118 - PATCH_RADIUS, 812 - PATCH_RADIUS), PATCH_SIZE, PATCH_SIZE,
        facecolor='none', edgecolor='#FF3030', linewidth=3.8, zorder=10,
    ))
    ax.set_title(
        "実際の自動流速推定で抽出された追跡候補領域\n"
        "Tracking candidates reproduced from the actual velocity estimation",
        fontsize=19,
        weight="bold",
        pad=12,
    )
    legend_patch = Patch(
        facecolor=(0.0, 0.78, 0.91, 0.12),
        edgecolor=COLOR,
        label=f"照合通過 / Accepted match: {PATCH_SIZE}×{PATCH_SIZE} px (n={len(accepted)})",
    )
    rejected_patch = Patch(
        facecolor=(1.0, 0.62, 0.11, 0.05),
        edgecolor=REJECTED_COLOR,
        linestyle="--",
        label=f"照合棄却 / Rejected match (n={len(candidates) - len(accepted)})",
    )
    ax.legend(
        handles=[legend_patch, rejected_patch],
        loc="lower left",
        fontsize=12,
        framealpha=0.92,
        borderpad=0.65,
    )
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    after_rgb = np.asarray(Image.open(AFTER_PATH).convert('RGB'))
    fig, ax = plt.subplots(figsize=(16, 9.5), constrained_layout=True)
    ax.imshow(after_rgb)
    ax.set_xlim(0, after_rgb.shape[1])
    ax.set_ylim(after_rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for x, y, target_x, target_y in destinations:
        ax.add_patch(Rectangle(
            (target_x - PATCH_RADIUS, target_y - PATCH_RADIUS), PATCH_SIZE, PATCH_SIZE,
            facecolor=(0.0, 0.78, 0.91, 0.08), edgecolor=COLOR, linewidth=1.6,
        ))
        if (x, y) == (1118, 812):
            for edge_color, width in [('white', 6), ('#FF3030', 3.8)]:
                ax.add_patch(Rectangle(
                    (target_x - PATCH_RADIUS, target_y - PATCH_RADIUS), PATCH_SIZE, PATCH_SIZE,
                    facecolor='none', edgecolor=edge_color, linewidth=width, zorder=10,
                ))
            print(f'sample destination=({target_x:.6f}, {target_y:.6f})')
    ax.set_title(
        '次画像で対応すると判定された追跡領域\n'
        'Matched tracking regions in the next frame',
        fontsize=19, weight='bold', pad=12,
    )
    ax.legend(handles=[Patch(
        facecolor=(0.0, 0.78, 0.91, 0.12), edgecolor=COLOR,
        label=f'対応位置 / Matched region: {PATCH_SIZE}×{PATCH_SIZE} px (n={len(destinations)})',
    )], loc='lower left', fontsize=12, framealpha=0.92, borderpad=0.65)
    after_output = OUTPUT_DIR / '01_tracking_matches_GX010202_after.png'
    fig.savefig(after_output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(after_output)
    print(
        f"candidates={len(candidates)} accepted={len(accepted)} "
        f"dx={estimate['dx']:.14f} confidence={estimate['confidence']:.16f}"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
