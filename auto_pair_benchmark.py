from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from benchmark_tracking import (
    DATASETS,
    Dataset,
    evenly_spaced,
    grayscale,
    image_resolver,
    list_images,
    local_match,
    read_rows,
    velocity_coefficients,
)


def candidate_centers(
    before: np.ndarray,
    after: np.ndarray,
    radius: int,
    count: int,
    step: int = 48,
) -> list[tuple[int, int, float]]:
    """Select textured, changing regions without using manual particle coordinates."""
    height, width = before.shape
    difference = np.abs(after - before)
    candidates: list[tuple[float, int, int]] = []
    x_min, x_max = max(radius + 155, 350), min(width - radius - 155, 2400)
    y_min, y_max = max(radius + 60, 25), min(height - radius - 60, 1050)
    half_score_window = max(10, radius // 2)
    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            ys = slice(y - half_score_window, y + half_score_window + 1)
            xs = slice(x - half_score_window, x + half_score_window + 1)
            motion = float(np.mean(difference[ys, xs]))
            texture = float(np.std(before[ys, xs]))
            score = motion * min(texture / 0.08, 1.0)
            candidates.append((score, x, y))

    chosen: list[tuple[int, int, float]] = []

    def add_candidate(score: float, x: int, y: int) -> bool:
        if all(math.hypot(x - old_x, y - old_y) >= step * 0.8 for old_x, old_y, _ in chosen):
            chosen.append((x, y, score))
            return True
        return False

    # Reserve candidates in every horizontal band so a narrow moving layer is
    # not displaced by widespread static texture elsewhere in the image.
    band_height = step * 3
    bands: dict[int, list[tuple[float, int, int]]] = {}
    for candidate in candidates:
        band = (candidate[2] - y_min) // band_height
        bands.setdefault(band, []).append(candidate)
    per_band = max(4, count // max(len(bands), 1))
    for band in sorted(bands):
        accepted = 0
        for score, x, y in sorted(bands[band], reverse=True):
            if add_candidate(score, x, y):
                accepted += 1
            if accepted >= per_band:
                break

    for score, x, y in sorted(candidates, reverse=True):
        add_candidate(score, x, y)
        if len(chosen) >= count:
            break
    chosen = chosen[:count]
    return chosen


def weighted_consensus(values: list[tuple[float, float]], tolerance: float = 4.0):
    if not values:
        return None
    displacement = np.asarray([item[0] for item in values])
    weights = np.asarray([item[1] for item in values])
    support = np.asarray([
        weights[np.abs(displacement - value) <= tolerance].sum()
        for value in displacement
    ])
    seed = displacement[int(np.argmax(support))]
    mask = np.abs(displacement - seed) <= tolerance
    if int(mask.sum()) < 2:
        return None
    ordered = np.argsort(displacement[mask])
    selected_values = displacement[mask][ordered]
    selected_weights = weights[mask][ordered]
    midpoint = selected_weights.sum() / 2
    estimate = float(selected_values[np.searchsorted(np.cumsum(selected_weights), midpoint)])
    return estimate, int(mask.sum()), float(mask.sum() / len(values))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    target = ordered_weights.sum() * quantile
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), target)])


def directional_motion(
    values: list[tuple[float, float]],
    minimum_displacement: float = 2.0,
    minimum_count: int = 3,
    dominance_ratio: float = 2.0,
    moving_quantile: float = 0.85,
    positive_quantile: float | None = 0.85,
    negative_quantile: float | None = 0.25,
):
    """Prefer a coherent moving population over the much larger static background."""
    if not values:
        return None
    displacement = np.asarray([item[0] for item in values])
    weights = np.asarray([item[1] for item in values])
    positive = displacement >= minimum_displacement
    negative = displacement <= -minimum_displacement
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    positive_quantile = moving_quantile if positive_quantile is None else positive_quantile
    negative_quantile = 1.0 - moving_quantile if negative_quantile is None else negative_quantile
    if positive_count >= minimum_count and positive_count >= dominance_ratio * max(negative_count, 1):
        estimate = weighted_quantile(displacement[positive], weights[positive], positive_quantile)
        return estimate, positive_count, positive_count / len(values)
    if negative_count >= minimum_count and negative_count >= dominance_ratio * max(positive_count, 1):
        estimate = weighted_quantile(displacement[negative], weights[negative], negative_quantile)
        return estimate, negative_count, negative_count / len(values)
    return weighted_consensus(values)


def match_candidates(
    before: np.ndarray,
    after: np.ndarray,
    radius: int = 32,
    candidate_count: int = 80,
) -> list[tuple[float, float]]:
    centers = candidate_centers(before, after, radius, candidate_count)
    matches: list[tuple[float, float]] = []
    for x, y, activity in centers:
        try:
            dx, dy, correlation, margin = local_match(
                before, after, (x, y), radius, search_x=150, search_y=55
            )
            reverse_dx, reverse_dy, _, _ = local_match(
                after, before, (x + dx, y + dy), radius, search_x=150, search_y=55
            )
        except ValueError:
            continue
        round_trip = math.hypot(dx + reverse_dx, dy + reverse_dy)
        if round_trip > 3.0 or margin < 0.015:
            continue
        quality = max(margin, 0.001) * max(correlation + 1.0, 0.05) * max(activity, 1e-5)
        matches.append((dx, quality))

    return matches


def estimate_from_matches(
    matches: list[tuple[float, float]],
    minimum_displacement: float = 2.0,
    minimum_count: int = 3,
    dominance_ratio: float = 2.0,
    moving_quantile: float = 0.85,
    positive_quantile: float | None = 0.85,
    negative_quantile: float | None = 0.25,
) -> dict[str, float] | None:
    consensus = directional_motion(
        matches,
        minimum_displacement=minimum_displacement,
        minimum_count=minimum_count,
        dominance_ratio=dominance_ratio,
        moving_quantile=moving_quantile,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
    )
    if consensus is None:
        return None
    dx, support, support_fraction = consensus
    displacements = np.asarray([item[0] for item in matches], dtype=float)
    moving = np.abs(displacements) >= minimum_displacement
    if abs(dx) >= minimum_displacement:
        selected = moving & (np.sign(displacements) == np.sign(dx))
        direction_consistency = float(selected.sum() / max(int(moving.sum()), 1))
    else:
        selected = np.abs(displacements - dx) <= 4.0
        direction_consistency = float(selected.sum() / max(len(displacements), 1))
    selected_values = displacements[selected]
    spread_px = (
        float(np.median(np.abs(selected_values - dx))) if selected_values.size else float("inf")
    )
    trusted_score = min(len(matches) / 30.0, 1.0)
    support_score = min(float(support) / 8.0, 1.0)
    spread_score = 0.0 if not np.isfinite(spread_px) else 1.0 / (1.0 + spread_px / 8.0)
    confidence = float(np.clip(
        0.20 * trusted_score
        + 0.20 * support_score
        + 0.35 * direction_consistency
        + 0.25 * spread_score,
        0.0,
        1.0,
    ))
    confidence_label = "high" if confidence >= 0.925 else "medium" if confidence >= 0.876 else "low"
    return {
        "dx": dx,
        "support": float(support),
        "trusted_matches": float(len(matches)),
        "support_fraction": support_fraction,
        "direction_consistency": direction_consistency,
        "spread_px": spread_px,
        "confidence": confidence,
        "confidence_label": confidence_label,
    }


def estimate_pair(
    before: np.ndarray,
    after: np.ndarray,
    radius: int = 32,
    candidate_count: int = 80,
) -> dict[str, float] | None:
    matches = match_candidates(before, after, radius, candidate_count)
    return estimate_from_matches(matches)


def selected_rows(rows: list[dict[str, str]], count: int):
    result = []
    for direction, predicate in (
        ("right", lambda velocity: velocity > 10),
        ("left", lambda velocity: velocity < -10),
        ("slow", lambda velocity: abs(velocity) <= 10),
    ):
        group = [row for row in rows if predicate(float(row["velocity"]))]
        result.extend((direction, row) for row in evenly_spaced(group, count))
    return result


def evaluate_dataset(dataset: Dataset, count: int, radius: int):
    rows = read_rows(dataset.labels)
    images = list_images(dataset.frames)
    resolve = image_resolver(images)
    coefficient = velocity_coefficients(rows)
    records = []
    for direction, row in selected_rows(rows, count):
        before_path = resolve(row["before_file"], row["before_frame"])
        after_path = resolve(row["after_file"], row["after_frame"])
        result = estimate_pair(grayscale(before_path), grayscale(after_path), radius=radius)
        predicted = None if result is None else result["dx"] * float(coefficient[0])
        records.append({
            "dataset": dataset.name,
            "direction": direction,
            "before_frame": row["before_frame"],
            "manual_velocity": float(row["velocity"]),
            "predicted_velocity": predicted,
            "abs_error": None if predicted is None else abs(predicted - float(row["velocity"])),
            "support": 0 if result is None else int(result["support"]),
            "trusted_matches": 0 if result is None else int(result["trusted_matches"]),
            "support_fraction": 0 if result is None else result["support_fraction"],
            "confidence": 0 if result is None else result["confidence"],
            "confidence_label": "unmeasurable" if result is None else result["confidence_label"],
        })
    return records


def print_summary(records):
    for dataset in sorted({row["dataset"] for row in records}):
        print(f"\nDataset {dataset}")
        for direction in ("right", "left", "slow", "all"):
            subset = [
                row for row in records
                if row["dataset"] == dataset and (direction == "all" or row["direction"] == direction)
            ]
            valid = [row for row in subset if row["predicted_velocity"] is not None]
            if not subset:
                continue
            errors = np.asarray([row["abs_error"] for row in valid])
            if not valid:
                print(f"  {direction:5s}: coverage=0/{len(subset)}")
                continue
            sign_rows = [row for row in valid if abs(row["manual_velocity"]) > 10]
            sign_ok = sum(
                math.copysign(1, row["manual_velocity"])
                == math.copysign(1, row["predicted_velocity"])
                for row in sign_rows
            )
            print(
                f"  {direction:5s}: coverage={len(valid)}/{len(subset)} "
                f"MAE={errors.mean():.2f} median={np.median(errors):.2f} "
                f"p90={np.quantile(errors, .9):.2f} sign={sign_ok}/{len(sign_rows)}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("auto_pair_results.csv"))
    args = parser.parse_args()
    datasets = [d for d in DATASETS if not args.datasets or d.name in args.datasets]
    records = []
    for dataset in datasets:
        records.extend(evaluate_dataset(dataset, args.count, args.radius))
    print_summary(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
