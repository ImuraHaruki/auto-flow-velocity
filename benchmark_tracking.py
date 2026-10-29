from __future__ import annotations

import argparse
import csv
import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import match_template


@dataclass(frozen=True)
class Dataset:
    name: str
    frames: Path
    labels: Path


DATASETS = (
    Dataset(
        "A",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010164_timestamp_JST_12s"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010164_timestamp_JST_12s\flow_velocity_manual.csv"),
    ),
    Dataset(
        "B",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010202_timestamp_JST"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010202_timestamp_JST\flow_velocity_results4.csv"),
    ),
    Dataset(
        "C_GX010101",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010101_timestamp_JST"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010101_timestamp_JST\flow_velocity_results.csv"),
    ),
    Dataset(
        "D_GX010105",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010105_timestamp_JST"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010105_timestamp_JST\flow_velocity_results.csv"),
    ),
    Dataset(
        "E_GX010165",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010165_timestamp_JST"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010165_timestamp_JST\flow_velocity_results.csv"),
    ),
    Dataset(
        "F_GX010213",
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010213_timestamp_JST"),
        Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames\GX010213_timestamp_JST\flow_velocity_results.csv"),
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        (
            path for path in folder.iterdir()
            if path.suffix.casefold() in extensions and "preview" not in path.stem.casefold()
        ),
        key=natural_key,
    )


def trailing_number(name: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(match.group(1)) if match else None


def image_resolver(images: list[Path]):
    by_name = {path.name.casefold(): path for path in images}
    by_number: dict[int, list[Path]] = {}
    by_modulo: dict[int, list[Path]] = {}
    for path in images:
        number = trailing_number(path.name)
        if number is not None:
            by_number.setdefault(number, []).append(path)
            by_modulo.setdefault(number % 10000, []).append(path)

    def resolve(csv_name: str, frame_number: str) -> Path:
        exact = by_name.get(Path(csv_name).name.casefold())
        if exact is not None:
            return exact
        number = trailing_number(csv_name)
        if number is not None:
            candidates = by_number.get(number, []) or by_modulo.get(number % 10000, [])
            if len(candidates) == 1:
                return candidates[0]
        return images[int(frame_number) - 1]

    return resolve


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


@functools.lru_cache(maxsize=8)
def grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def quadratic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -1.0, 1.0))


def local_match(
    before: np.ndarray,
    after: np.ndarray,
    point: tuple[float, float],
    patch_radius: int,
    search_x: int,
    search_y: int,
) -> tuple[float, float, float, float]:
    x = int(round(point[0]))
    y = int(round(point[1]))
    height, width = before.shape
    radius = patch_radius
    if not (radius <= x < width - radius and radius <= y < height - radius):
        raise ValueError("template outside image")
    template = before[y - radius:y + radius + 1, x - radius:x + radius + 1]

    left = max(0, x - radius - search_x)
    right = min(width, x + radius + search_x + 1)
    top = max(0, y - radius - search_y)
    bottom = min(height, y + radius + search_y + 1)
    search = after[top:bottom, left:right]
    correlation = match_template(search, template, pad_input=False)
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    best = float(correlation[peak_y, peak_x])
    subpixel_x = 0.0
    subpixel_y = 0.0
    if 0 < peak_x < correlation.shape[1] - 1:
        subpixel_x = quadratic_peak_offset(
            float(correlation[peak_y, peak_x - 1]),
            best,
            float(correlation[peak_y, peak_x + 1]),
        )
    if 0 < peak_y < correlation.shape[0] - 1:
        subpixel_y = quadratic_peak_offset(
            float(correlation[peak_y - 1, peak_x]),
            best,
            float(correlation[peak_y + 1, peak_x]),
        )
    competing = correlation.copy()
    exclusion = max(3, patch_radius // 6)
    competing[
        max(0, peak_y - exclusion):peak_y + exclusion + 1,
        max(0, peak_x - exclusion):peak_x + exclusion + 1,
    ] = -np.inf
    second = float(np.max(competing)) if np.isfinite(competing).any() else -1.0
    matched_x = left + peak_x + radius + subpixel_x
    matched_y = top + peak_y + radius + subpixel_y
    return float(matched_x - point[0]), float(matched_y - point[1]), best, best - second


def velocity_coefficients(rows: list[dict[str, str]]) -> np.ndarray:
    displacement = np.array([
        (
            float(row["after_x_px"]) - float(row["before_x_px"]),
            float(row["after_y_px"]) - float(row["before_y_px"]),
        )
        for row in rows
    ])
    velocity = np.array([float(row["velocity"]) for row in rows])
    return np.linalg.lstsq(displacement, velocity, rcond=None)[0]


def evaluate(dataset: Dataset, count_per_direction: int, radii: list[int]) -> None:
    rows = read_rows(dataset.labels)
    images = list_images(dataset.frames)
    resolve_image = image_resolver(images)
    coefficient = velocity_coefficients(rows)
    selected = []
    for direction, predicate in (
        ("right", lambda velocity: velocity > 10),
        ("left", lambda velocity: velocity < -10),
        ("slow", lambda velocity: abs(velocity) <= 10),
    ):
        group = [row for row in rows if predicate(float(row["velocity"]))]
        for row in evenly_spaced(group, count_per_direction):
            selected.append((direction, row))

    print(f"\nDataset {dataset.name}: labels={len(rows)}, samples={len(selected)}")
    velocities = np.asarray([float(row["velocity"]) for row in rows])
    print(
        f"frames={len(images)}, velocity min/median/max="
        f"{velocities.min():.2f}/{np.median(velocities):.2f}/{velocities.max():.2f}, "
        f"right/left/slow={(velocities > 10).sum()}/{(velocities < -10).sum()}/{(np.abs(velocities) <= 10).sum()}"
    )
    print(f"velocity coefficients: x={coefficient[0]:.8f}, y={coefficient[1]:.8f}")
    for radius in radii:
        records = []
        for direction, row in selected:
            before_path = resolve_image(row["before_file"], row["before_frame"])
            after_path = resolve_image(row["after_file"], row["after_frame"])
            try:
                dx, dy, correlation, peak_margin = local_match(
                    grayscale(before_path),
                    grayscale(after_path),
                    (float(row["before_x_px"]), float(row["before_y_px"])),
                    patch_radius=radius,
                    search_x=150,
                    search_y=55,
                )
                reverse_dx, reverse_dy, _, _ = local_match(
                    grayscale(after_path),
                    grayscale(before_path),
                    (float(row["before_x_px"]) + dx, float(row["before_y_px"]) + dy),
                    patch_radius=radius,
                    search_x=150,
                    search_y=55,
                )
            except ValueError:
                continue
            round_trip_error = math.hypot(dx + reverse_dx, dy + reverse_dy)
            predicted_velocity = float(np.dot((dx, dy), coefficient))
            manual_velocity = float(row["velocity"])
            records.append(
                (direction, predicted_velocity, manual_velocity, correlation, peak_margin, round_trip_error)
            )

        print(f"  radius={radius}px")
        for mode, subset in (
            ("raw", records),
            ("trusted", [r for r in records if r[5] <= 3.0 and r[4] >= 0.015]),
        ):
            print(f"    {mode}: accepted={len(subset)}/{len(records)}")
            for direction in ("right", "left", "slow", "all"):
                group = subset if direction == "all" else [r for r in subset if r[0] == direction]
                if not group:
                    continue
                errors = np.array([abs(r[1] - r[2]) for r in group])
                correlations = np.array([r[3] for r in group])
                sign_ok = [
                    math.copysign(1, r[1]) == math.copysign(1, r[2])
                    for r in group if abs(r[2]) > 10
                ]
                print(
                    f"      {direction:5s} n={len(group):3d} "
                    f"MAE={errors.mean():7.2f} median={np.median(errors):7.2f} "
                    f"p90={np.quantile(errors, .9):7.2f} corr={np.median(correlations):.3f} "
                    f"sign={sum(sign_ok)}/{len(sign_ok)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--radii", type=int, nargs="+", default=[16, 24, 32, 48])
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    for dataset in DATASETS:
        if args.datasets and dataset.name not in args.datasets:
            continue
        evaluate(dataset, args.count, args.radii)


if __name__ == "__main__":
    main()
