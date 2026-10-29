from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark_tracking import DATASETS, list_images, read_rows, velocity_coefficients
from run_auto_sequence import choose_adaptive_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=60, help="Sampling interval in source frames")
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("unlabeled_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []

    for dataset in DATASETS:
        print(f"Processing {dataset.name} ...", flush=True)
        rows = read_rows(dataset.labels)
        labeled_frames = {int(row["before_frame"]) for row in rows}
        images = list_images(dataset.frames)
        cm_per_pixel = float(velocity_coefficients(rows)[0]) / 240.0
        records = []
        used_indices = set()
        for nominal_index in range(0, len(images) - 1, args.step):
            index = nominal_index
            while index + args.max_gap < len(images) and index + 1 in labeled_frames:
                index += 1
            if index + 1 >= len(images) or index in used_indices or index + 1 in labeled_frames:
                continue
            used_indices.add(index)
            gap, result, status = choose_adaptive_result(
                images, index, max_gap=args.max_gap, radius=32,
                low_speed_pixels=4.0, minimum_confidence=0.75,
            )
            raw_velocity = (
                "" if result is None else result["dx"] * cm_per_pixel * 240.0 / gap
            )
            record = {
                "dataset": dataset.name,
                "before_frame": index + 1,
                "after_frame": index + gap + 1,
                "time_s": index / 240.0,
                "frame_gap": gap,
                "velocity_cm_s": raw_velocity if status == "measured" else "",
                "raw_velocity_cm_s": raw_velocity,
                "status": status,
                "confidence": 0 if result is None else result["confidence"],
                "confidence_label": "unmeasurable" if result is None else result["confidence_label"],
                "support": 0 if result is None else int(result["support"]),
                "trusted_matches": 0 if result is None else int(result["trusted_matches"]),
                "support_fraction": 0 if result is None else result["support_fraction"],
                "direction_consistency": 0 if result is None else result["direction_consistency"],
                "spread_px": "" if result is None else result["spread_px"],
            }
            records.append(record)
            combined.append(record)

        path = args.output_dir / f"{dataset.name}_unlabeled.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        measured = [row for row in records if row["status"] == "measured"]
        velocities = np.asarray([row["velocity_cm_s"] for row in measured], dtype=float)
        print(
            f"  measured={len(measured)}/{len(records)}, "
            f"median={np.median(velocities):.2f} cm/s, "
            f"range={velocities.min():.2f}..{velocities.max():.2f}",
            flush=True,
        )

    combined_path = args.output_dir / "all_unlabeled_results.csv"
    with combined_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved: {combined_path.resolve()}")


if __name__ == "__main__":
    main()
