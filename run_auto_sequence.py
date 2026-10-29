from __future__ import annotations

import argparse
import csv
from pathlib import Path

from auto_pair_benchmark import estimate_pair
from benchmark_tracking import grayscale, list_images


def choose_adaptive_result(
    images,
    index: int,
    max_gap: int,
    radius: int,
    low_speed_pixels: float,
    minimum_confidence: float,
):
    available_gap = min(max_gap, len(images) - index - 1)
    first = estimate_pair(grayscale(images[index]), grayscale(images[index + 1]), radius=radius)
    candidates = [(1, first)]
    needs_longer_gap = (
        first is None
        or abs(first["dx"]) < low_speed_pixels
        or first["confidence"] < 0.65
    )
    if needs_longer_gap:
        for gap in range(2, available_gap + 1):
            result = estimate_pair(
                grayscale(images[index]), grayscale(images[index + gap]), radius=radius
            )
            candidates.append((gap, result))

    direction_reference = None
    if first is not None and abs(first["dx"]) >= 0.5:
        direction_reference = 1 if first["dx"] > 0 else -1

    direction_consistent = [
        (gap, result) for gap, result in candidates
        if result is not None
        and (
            direction_reference is None
            or abs(result["dx"]) < 0.5
            or (1 if result["dx"] > 0 else -1) == direction_reference
        )
    ]
    valid = [
        (gap, result) for gap, result in direction_consistent
        if result is not None and result["confidence"] >= minimum_confidence
    ]
    pool = valid or direction_consistent or [(gap, result) for gap, result in candidates if result is not None]
    if not pool:
        return 1, None, "unmeasurable"

    def selection_score(item):
        gap, result = item
        displacement_score = min(abs(result["dx"]) / max(low_speed_pixels, 1e-6), 1.0)
        gap_bonus = 0.02 * gap if abs(result["dx"]) < low_speed_pixels else 0.0
        return result["confidence"] + 0.15 * displacement_score + gap_bonus

    gap, result = max(pool, key=selection_score)
    status = "measured" if result["confidence"] >= minimum_confidence else "unmeasurable"
    return gap, result, status


def main():
    parser = argparse.ArgumentParser(
        description="Estimate representative one-dimensional velocity from consecutive frames."
    )
    parser.add_argument("frames", type=Path, help="Folder containing sequential images")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument(
        "--cm-per-pixel",
        type=float,
        required=True,
        help="Calibration scale along the positive image-x/calibration-axis direction",
    )
    parser.add_argument("--start", type=int, default=1, help="First 1-based frame number")
    parser.add_argument("--end", type=int, default=None, help="Last 1-based frame number")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--low-speed-pixels", type=float, default=4.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.85)
    args = parser.parse_args()

    images = list_images(args.frames)
    if len(images) < 2:
        raise SystemExit("At least two images are required.")
    start_index = max(args.start - 1, 0)
    end_index = min(args.end if args.end is not None else len(images) - 1, len(images) - 1)
    if start_index >= end_index:
        raise SystemExit("The requested range contains no frame pair.")

    records = []
    pair_indices = list(range(start_index, end_index, args.step))
    for completed, index in enumerate(pair_indices, start=1):
        gap, result, status = choose_adaptive_result(
            images,
            index,
            max_gap=args.max_gap,
            radius=args.radius,
            low_speed_pixels=args.low_speed_pixels,
            minimum_confidence=args.minimum_confidence,
        )
        raw_velocity = (
            "" if result is None
            else result["dx"] * args.cm_per_pixel * args.fps / gap
        )
        records.append({
            "before_frame": index + 1,
            "after_frame": index + gap + 1,
            "before_file": images[index].name,
            "after_file": images[index + gap].name,
            "time_s": index / args.fps,
            "delta_time_s": gap / args.fps,
            "frame_gap": gap,
            "displacement_px": "" if result is None else result["dx"],
            "velocity_cm_s": raw_velocity if status == "measured" else "",
            "raw_velocity_cm_s": raw_velocity,
            "status": status,
            "support": 0 if result is None else int(result["support"]),
            "trusted_matches": 0 if result is None else int(result["trusted_matches"]),
            "support_fraction": 0 if result is None else result["support_fraction"],
            "direction_consistency": 0 if result is None else result["direction_consistency"],
            "spread_px": "" if result is None else result["spread_px"],
            "confidence": 0 if result is None else result["confidence"],
            "confidence_label": "unmeasurable" if result is None else result["confidence_label"],
        })
        if completed == 1 or completed % 25 == 0 or completed == len(pair_indices):
            print(f"Processed {completed}/{len(pair_indices)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
