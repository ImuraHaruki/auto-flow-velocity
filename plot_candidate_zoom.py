import math

import matplotlib.pyplot as plt
from PIL import Image

from auto_pair_benchmark import candidate_centers
from benchmark_tracking import grayscale, local_match
from plot_tracking_candidates_full_frame import BEFORE_PATH, AFTER_PATH, OUTPUT_DIR, set_font


def main():
    a, b = grayscale(BEFORE_PATH), grayscale(AFTER_PATH)
    accepted = []
    for x, y, activity in candidate_centers(a, b, 32, 80):
        if not (1000 <= x <= 2200 and 700 <= y <= 850):
            continue
        try:
            dx, dy, correlation, margin = local_match(a, b, (x, y), 32, search_x=150, search_y=55)
            rx, ry, _, _ = local_match(b, a, (x + dx, y + dy), 32, search_x=150, search_y=55)
        except ValueError:
            continue
        error = math.hypot(dx + rx, dy + ry)
        if error <= 3 and margin >= 0.015:
            accepted.append((abs(dx - 22.36162441323586), x, y, dx, error))
    _, x, y, dx, error = min(accepted)
    patch = Image.open(BEFORE_PATH).convert('RGB').crop((x-32, y-32, x+33, y+33))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch.save(OUTPUT_DIR / 'GX010202_candidate_original_65px.png')
    set_font()
    fig, ax = plt.subplots(figsize=(7, 7.5), constrained_layout=True)
    ax.imshow(patch, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#00C8E8')
        spine.set_linewidth(3)
    ax.set_title('照合を通過した追跡候補の拡大\nZoomed tracking candidate (accepted match)', fontsize=15, pad=16)
    fig.supxlabel(f'GX010202_timestamp_JST_11082.jpg\n中心 / Center: (x={x}, y={y}) px | 65 × 65 px\n元画素を拡大表示 / Original pixels enlarged', fontsize=11)
    output = OUTPUT_DIR / 'GX010202_candidate_zoom.png'
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'center={x},{y} dx={dx} round_trip_error={error}')
    print(output)


if __name__ == '__main__':
    main()
