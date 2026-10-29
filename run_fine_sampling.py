from __future__ import annotations

import argparse
import csv
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark_tracking import DATASETS, list_images, read_rows, velocity_coefficients
from run_auto_sequence import choose_adaptive_result


@functools.lru_cache(maxsize=12)
def cached_images(folder: str):
    return tuple(list_images(Path(folder)))


def process_task(task):
    dataset_name, folder, index, cm_per_pixel, max_gap = task
    images = cached_images(folder)
    gap, result, status = choose_adaptive_result(
        images, index, max_gap=max_gap, radius=32,
        low_speed_pixels=4.0, minimum_confidence=0.75,
    )
    raw_velocity = "" if result is None else result["dx"] * cm_per_pixel * 240.0 / gap
    return {
        "dataset": dataset_name,
        "before_frame": index + 1,
        "after_frame": index + gap + 1,
        "before_file": images[index].name,
        "after_file": images[index + gap].name,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=Path("fine_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset in DATASETS:
        images = list_images(dataset.frames)
        rows = read_rows(dataset.labels)
        cm_per_pixel = float(velocity_coefficients(rows)[0]) / 240.0
        for index in range(0, len(images) - 1, args.step):
            tasks.append((dataset.name, str(dataset.frames), index, cm_per_pixel, args.max_gap))

    records = []
    print(f"Processing {len(tasks)} frame pairs with {args.workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, record in enumerate(executor.map(process_task, tasks, chunksize=4), start=1):
            records.append(record)
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(f"Processed {completed}/{len(tasks)}", flush=True)

    fields = list(records[0])
    combined_path = args.output_dir / "all_fine_results.csv"
    with combined_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    for dataset in DATASETS:
        subset = [row for row in records if row["dataset"] == dataset.name]
        path = args.output_dir / f"{dataset.name}_fine.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
    measured = sum(row["status"] == "measured" for row in records)
    print(f"Measured {measured}/{len(records)}")
    print(f"Saved: {combined_path.resolve()}")


if __name__ == "__main__":
    main()
