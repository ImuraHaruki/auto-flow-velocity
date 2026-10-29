from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from auto_pair_benchmark import estimate_from_matches, match_candidates, selected_rows
from benchmark_tracking import (
    DATASETS,
    grayscale,
    image_resolver,
    list_images,
    read_rows,
    velocity_coefficients,
)


PARAMETER_GRID = {
    "minimum_displacement": (2.0, 3.0, 4.0, 5.0),
    "minimum_count": (3, 5, 8),
    "dominance_ratio": (1.5, 2.0, 3.0),
    "positive_quantile": (0.55, 0.65, 0.75, 0.85, 0.95),
    "negative_quantile": (0.05, 0.15, 0.25, 0.35, 0.45),
}


def parameter_sets():
    keys = tuple(PARAMETER_GRID)
    for values in itertools.product(*(PARAMETER_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def build_cache(count: int, radius: int):
    records = []
    for dataset in DATASETS:
        rows = read_rows(dataset.labels)
        images = list_images(dataset.frames)
        resolve = image_resolver(images)
        coefficient = float(velocity_coefficients(rows)[0])
        print(f"Extracting {dataset.name} ...", flush=True)
        for direction, row in selected_rows(rows, count):
            before_path = resolve(row["before_file"], row["before_frame"])
            after_path = resolve(row["after_file"], row["after_frame"])
            matches = match_candidates(
                grayscale(before_path), grayscale(after_path), radius=radius
            )
            records.append({
                "dataset": dataset.name,
                "direction": direction,
                "before_frame": int(row["before_frame"]),
                "manual_velocity": float(row["velocity"]),
                "velocity_per_pixel": coefficient,
                "matches": [[float(dx), float(weight)] for dx, weight in matches],
            })
    return records


def predict(record, parameters):
    result = estimate_from_matches(
        [(float(dx), float(weight)) for dx, weight in record["matches"]],
        **parameters,
    )
    if result is None:
        return None, None
    return result["dx"] * record["velocity_per_pixel"], result


def tuning_score(records, parameters):
    errors = {"right": [], "left": [], "slow": []}
    sign_wrong = 0
    sign_count = 0
    missing = 0
    for record in records:
        predicted, _ = predict(record, parameters)
        if predicted is None:
            missing += 1
            continue
        manual = record["manual_velocity"]
        errors[record["direction"]].append(abs(predicted - manual))
        if abs(manual) > 10:
            sign_count += 1
            sign_wrong += int(np.sign(predicted) != np.sign(manual))
    if any(not values for values in errors.values() if records):
        return float("inf")
    direction_mae = {
        key: float(np.mean(values)) if values else 0.0 for key, values in errors.items()
    }
    balanced_mae = (
        1.5 * direction_mae["right"]
        + direction_mae["left"]
        + 0.75 * direction_mae["slow"]
    ) / 3.25
    sign_penalty = 50.0 * sign_wrong / max(sign_count, 1)
    missing_penalty = 50.0 * missing / max(len(records), 1)
    return balanced_mae + sign_penalty + missing_penalty


def metrics(records):
    errors = np.asarray([row["abs_error"] for row in records if row["abs_error"] is not None])
    sign_rows = [row for row in records if abs(row["manual_velocity"]) > 10 and row["predicted_velocity"] is not None]
    sign_ok = sum(np.sign(row["manual_velocity"]) == np.sign(row["predicted_velocity"]) for row in sign_rows)
    return {
        "n": len(records),
        "mae": float(errors.mean()),
        "median_ae": float(np.median(errors)),
        "p90_ae": float(np.quantile(errors, 0.9)),
        "sign_ok": int(sign_ok),
        "sign_n": len(sign_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--cache", type=Path, default=Path("candidate_cache.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("loso_results"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fixed-directional", action="store_true")
    args = parser.parse_args()

    if args.cache.exists() and not args.rebuild_cache:
        records = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        records = build_cache(args.count, args.radius)
        args.cache.write_text(json.dumps(records), encoding="utf-8")

    predictions = []
    parameter_rows = []
    for held_out in [dataset.name for dataset in DATASETS]:
        training = [row for row in records if row["dataset"] != held_out]
        test = [row for row in records if row["dataset"] == held_out]
        if args.fixed_directional:
            best = {
                "minimum_displacement": 2.0,
                "minimum_count": 3,
                "dominance_ratio": 2.0,
                "positive_quantile": 0.85,
                "negative_quantile": 0.25,
            }
        else:
            best = min(parameter_sets(), key=lambda parameters: tuning_score(training, parameters))
        parameter_rows.append({"held_out_dataset": held_out, **best})
        for record in test:
            predicted, detail = predict(record, best)
            predictions.append({
                "dataset": held_out,
                "direction": record["direction"],
                "before_frame": record["before_frame"],
                "manual_velocity": record["manual_velocity"],
                "predicted_velocity": predicted,
                "abs_error": None if predicted is None else abs(predicted - record["manual_velocity"]),
                "support": 0 if detail is None else int(detail["support"]),
                "trusted_matches": 0 if detail is None else int(detail["trusted_matches"]),
                "support_fraction": 0 if detail is None else detail["support_fraction"],
                "direction_consistency": 0 if detail is None else detail["direction_consistency"],
                "spread_px": "" if detail is None else detail["spread_px"],
                "confidence": 0 if detail is None else detail["confidence"],
                "confidence_label": "unmeasurable" if detail is None else detail["confidence_label"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "loso_predictions.csv"
    parameter_path = args.output_dir / "loso_parameters.csv"
    summary_path = args.output_dir / "loso_summary.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with parameter_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)

    summary = []
    for dataset in [dataset.name for dataset in DATASETS] + ["ALL"]:
        source = predictions if dataset == "ALL" else [row for row in predictions if row["dataset"] == dataset]
        for direction in ("right", "left", "slow", "all"):
            group = source if direction == "all" else [row for row in source if row["direction"] == direction]
            if group:
                summary.append({"dataset": dataset, "direction": direction, **metrics(group)})
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        if row["dataset"] == "ALL":
            print(row)
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
