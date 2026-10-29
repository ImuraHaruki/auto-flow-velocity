from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from auto_pair_benchmark import selected_rows
from benchmark_tracking import (
    DATASETS,
    image_resolver,
    list_images,
    read_rows,
    velocity_coefficients,
)
from run_auto_sequence import choose_adaptive_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("adaptive_gap_validation.csv"))
    args = parser.parse_args()
    output = []

    for dataset in DATASETS:
        print(f"Evaluating {dataset.name} ...", flush=True)
        rows = read_rows(dataset.labels)
        images = list_images(dataset.frames)
        resolve = image_resolver(images)
        index_by_path = {path.resolve(): index for index, path in enumerate(images)}
        velocity_per_pixel_at_gap1 = float(velocity_coefficients(rows)[0])
        for direction, row in selected_rows(rows, args.count):
            before_path = resolve(row["before_file"], row["before_frame"])
            index = index_by_path[before_path.resolve()]
            gap, result, status = choose_adaptive_result(
                images, index, max_gap=args.max_gap, radius=32,
                low_speed_pixels=4.0, minimum_confidence=0.75,
            )
            predicted = None if result is None else result["dx"] * velocity_per_pixel_at_gap1 / gap
            manual = float(row["velocity"])
            output.append({
                "dataset": dataset.name,
                "direction": direction,
                "before_frame": int(row["before_frame"]),
                "manual_velocity": manual,
                "predicted_velocity": predicted,
                "abs_error": None if predicted is None else abs(predicted - manual),
                "frame_gap": gap,
                "status": status,
                "confidence": 0 if result is None else result["confidence"],
                "confidence_label": "unmeasurable" if result is None else result["confidence_label"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    for direction in ("right", "left", "slow", "all"):
        group = output if direction == "all" else [row for row in output if row["direction"] == direction]
        measured = [row for row in group if row["status"] == "measured" and row["predicted_velocity"] is not None]
        errors = np.asarray([row["abs_error"] for row in measured])
        sign_rows = [row for row in measured if abs(row["manual_velocity"]) > 10]
        sign_ok = sum(
            math.copysign(1, row["manual_velocity"]) == math.copysign(1, row["predicted_velocity"])
            for row in sign_rows
        )
        print(
            f"{direction}: coverage={len(measured)}/{len(group)}, MAE={errors.mean():.2f}, "
            f"median={np.median(errors):.2f}, p90={np.quantile(errors, .9):.2f}, "
            f"sign={sign_ok}/{len(sign_rows)}"
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
