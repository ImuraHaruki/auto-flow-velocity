from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from benchmark_tracking import grayscale, list_images
from run_auto_sequence import choose_adaptive_result


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANUAL_PATTERN = re.compile(r"^flow_velocity.*\.csv$", re.IGNORECASE)

# Median x-axis calibration among GX010101, GX010105, GX010164 (two sets),
# and GX010165.  The later group is the mean of GX010202 and GX010213.
EARLY_CM_PER_PIXEL = 0.010639189587055729
LATE_CM_PER_PIXEL = 0.010799203596492758
CALIBRATION_SPLIT_SEQUENCE = 202


def sequence_number(name: str) -> int:
    match = re.match(r"GX0*(\d+)_timestamp_JST", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract sequence number from {name!r}")
    return int(match.group(1))


def calibration_for(name: str) -> tuple[float, str]:
    number = sequence_number(name)
    if number < CALIBRATION_SPLIT_SEQUENCE:
        return EARLY_CM_PER_PIXEL, "early_GX010101_to_GX010193"
    return LATE_CM_PER_PIXEL, "late_GX010202_to_GX010213"


def has_particle_manual_csv(folder: Path) -> bool:
    return any(
        path.is_file() and MANUAL_PATTERN.match(path.name)
        for path in folder.iterdir()
    )


def discover_unlabeled(
    root: Path,
    include_manual: bool = False,
    dataset_names: set[str] | None = None,
) -> list[tuple[Path, list[Path]]]:
    result = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not folder.is_dir() or not re.match(r"^GX\d+_timestamp_JST", folder.name):
            continue
        if dataset_names is not None and folder.name not in dataset_names:
            continue
        if not include_manual and has_particle_manual_csv(folder):
            continue
        images = list_images(folder)
        if len(images) >= 2:
            result.append((folder, images))
    return result


@functools.lru_cache(maxsize=6)
def cached_images(folder: str):
    return tuple(list_images(Path(folder)))


def process_task(task):
    dataset, folder, index, step, cm_per_pixel, scale_group, max_gap, minimum_confidence = task
    images = cached_images(folder)
    read_error = None
    for attempt in range(3):
        try:
            gap, result, status = choose_adaptive_result(
                images,
                index,
                max_gap=max_gap,
                radius=32,
                low_speed_pixels=4.0,
                minimum_confidence=minimum_confidence,
            )
            break
        except OSError as error:
            read_error = error
            grayscale.cache_clear()
            time.sleep(0.5 * (attempt + 1))
    else:
        gap, result, status = 1, None, "unmeasurable"
    raw_velocity = "" if result is None else result["dx"] * cm_per_pixel * 240.0 / gap
    return {
        "dataset": dataset,
        "before_frame": index + 1,
        "after_frame": index + gap + 1,
        "before_file": images[index].name,
        "after_file": images[index + gap].name,
        "time_s": index / 240.0,
        "delta_time_s": gap / 240.0,
        "sample_step_frames": step,
        "frame_gap": gap,
        "cm_per_pixel": cm_per_pixel,
        "calibration_group": scale_group,
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
        "confidence_label": (
            "read_error" if result is None and read_error is not None
            else "unmeasurable" if result is None
            else result["confidence_label"]
        ),
    }


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_final_outputs(records: list[dict[str, object]], output_dir: Path):
    if not records:
        return
    records.sort(key=lambda row: (str(row["dataset"]), int(row["before_frame"])))
    fields = list(records[0])
    combined = output_dir / "all_unlabeled_velocity.csv"
    with combined.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    for dataset in sorted({str(row["dataset"]) for row in records}):
        subset = [row for row in records if row["dataset"] == dataset]
        path = output_dir / f"{dataset}_auto_velocity.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)

    summary_fields = [
        "dataset", "evaluated", "measured", "unmeasurable", "measurement_rate",
        "median_confidence", "mean_confidence", "median_velocity_cm_s",
        "minimum_velocity_cm_s", "maximum_velocity_cm_s", "cm_per_pixel",
        "calibration_group",
    ]
    summary_rows = []
    for dataset in sorted({str(row["dataset"]) for row in records}):
        subset = [row for row in records if row["dataset"] == dataset]
        measured = [row for row in subset if row["status"] == "measured" and row["velocity_cm_s"] != ""]
        confidences = sorted(float(row["confidence"]) for row in subset)
        velocities = sorted(float(row["velocity_cm_s"]) for row in measured)

        def median(values):
            if not values:
                return ""
            middle = len(values) // 2
            return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

        summary_rows.append({
            "dataset": dataset,
            "evaluated": len(subset),
            "measured": len(measured),
            "unmeasurable": len(subset) - len(measured),
            "measurement_rate": len(measured) / len(subset),
            "median_confidence": median(confidences),
            "mean_confidence": sum(confidences) / len(confidences),
            "median_velocity_cm_s": median(velocities),
            "minimum_velocity_cm_s": min(velocities) if velocities else "",
            "maximum_velocity_cm_s": max(velocities) if velocities else "",
            "cm_per_pixel": subset[0]["cm_per_pixel"],
            "calibration_group": subset[0]["calibration_group"],
        })
    with (output_dir / "sequence_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate velocity for all frame sequences without flow_velocity*.csv labels."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--minimum-confidence", type=float, default=0.85)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Optional task limit for timing tests")
    parser.add_argument(
        "--include-manual", action="store_true",
        help="Include sequences containing flow_velocity*.csv (excluded by default)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Optional exact sequence folder names to process",
    )
    parser.add_argument(
        "--cm-per-pixel", type=float, default=None,
        help="Optional calibration override applied to every selected sequence",
    )
    parser.add_argument(
        "--calibration-map", type=Path, default=None,
        help="Optional JSON mapping dataset names to cm_per_pixel or calibration objects",
    )
    args = parser.parse_args()

    calibration_map = {}
    if args.calibration_map is not None:
        calibration_map = json.loads(args.calibration_map.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.csv"
    datasets = discover_unlabeled(
        args.root,
        include_manual=args.include_manual,
        dataset_names=set(args.datasets) if args.datasets else None,
    )
    existing = read_existing(checkpoint)
    completed_keys = {
        (row["dataset"], int(row["before_frame"]) - 1)
        for row in existing
    }

    tasks = []
    inventory = []
    for folder, images in datasets:
        if args.cm_per_pixel is not None:
            scale, scale_group = args.cm_per_pixel, "user_override"
        elif folder.name in calibration_map:
            mapped = calibration_map[folder.name]
            if isinstance(mapped, dict):
                scale = float(mapped["cm_per_pixel"])
                scale_group = str(mapped.get("calibration_group", "calibration_map"))
            else:
                scale = float(mapped)
                scale_group = "calibration_map"
        else:
            scale, scale_group = calibration_for(folder.name)
        indices = list(range(0, len(images) - 1, args.step))
        inventory.append({
            "dataset": folder.name,
            "frames": len(images),
            "evaluation_times": len(indices),
            "cm_per_pixel": scale,
            "calibration_group": scale_group,
            "folder": str(folder),
        })
        for index in indices:
            if (folder.name, index) not in completed_keys:
                tasks.append((
                    folder.name, str(folder), index, args.step, scale, scale_group,
                    args.max_gap, args.minimum_confidence,
                ))
    if args.limit is not None:
        tasks = tasks[:args.limit]

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(args.root),
        "fps": 240.0,
        "step": args.step,
        "max_gap": args.max_gap,
        "minimum_confidence": args.minimum_confidence,
        "workers": args.workers,
        "manual_exclusion_pattern": MANUAL_PATTERN.pattern,
        "include_manual": args.include_manual,
        "dataset_filter": args.datasets,
        "cm_per_pixel_override": args.cm_per_pixel,
        "calibration_map": str(args.calibration_map) if args.calibration_map else None,
        "calibration": {
            "early_cm_per_pixel": EARLY_CM_PER_PIXEL,
            "late_cm_per_pixel": LATE_CM_PER_PIXEL,
            "split_sequence": CALIBRATION_SPLIT_SEQUENCE,
            "assumption": "Use median/mean calibration from labeled sequences in the corresponding acquisition group.",
        },
        "datasets": inventory,
        "existing_checkpoint_rows": len(existing),
        "tasks_this_run": len(tasks),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Discovered {len(datasets)} unlabeled sequences, "
        f"{sum(item['frames'] for item in inventory)} frames, "
        f"{sum(item['evaluation_times'] for item in inventory)} total evaluation times.",
        flush=True,
    )
    print(f"Checkpoint already contains {len(existing)} rows; processing {len(tasks)} tasks.", flush=True)
    if not tasks:
        write_final_outputs(existing, args.output_dir)
        print("Nothing to process. Final outputs refreshed.", flush=True)
        return

    fields = list(process_task.__annotations__)  # overwritten after first result
    mode = "a" if checkpoint.exists() and checkpoint.stat().st_size > 0 else "w"
    started = time.perf_counter()
    new_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(process_task, tasks, chunksize=2)
        checkpoint_file = None
        writer = None
        try:
            for completed, record in enumerate(iterator, start=1):
                if checkpoint_file is None:
                    fields = list(record)
                    checkpoint_file = checkpoint.open(mode, encoding="utf-8-sig", newline="")
                    writer = csv.DictWriter(checkpoint_file, fieldnames=fields)
                    if mode == "w":
                        writer.writeheader()
                writer.writerow(record)
                new_records.append(record)
                if completed % 5 == 0:
                    checkpoint_file.flush()
                if completed == 1 or completed % args.progress_every == 0 or completed == len(tasks):
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed
                    remaining = len(tasks) - completed
                    eta_seconds = remaining / rate if rate else math.inf
                    measured = sum(row["status"] == "measured" for row in new_records)
                    print(
                        f"PROGRESS {completed}/{len(tasks)} "
                        f"({100 * completed / len(tasks):.1f}%) | "
                        f"elapsed={elapsed / 60:.1f} min | "
                        f"rate={rate:.3f} pairs/s | "
                        f"ETA={eta_seconds / 3600:.2f} h | "
                        f"measured={measured}/{completed}",
                        flush=True,
                    )
        finally:
            if checkpoint_file is not None:
                checkpoint_file.flush()
                checkpoint_file.close()

    all_records = read_existing(checkpoint)
    write_final_outputs(all_records, args.output_dir)
    elapsed = time.perf_counter() - started
    print(f"COMPLETE rows={len(all_records)} elapsed_hours={elapsed / 3600:.3f}", flush=True)
    print(f"Saved: {(args.output_dir / 'all_unlabeled_velocity.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
