from __future__ import annotations

from pathlib import Path
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_tracking import image_resolver, list_images


ROOT = Path(r"C:\Users\haruh\Documents\Codex\flow_velocity_gui\auto_velocity_prototype\manual_series_combined_20260831")
AUTO_PATH = ROOT / "manual_series_auto_velocity_combined.csv"
MANUAL_PATH = ROOT / "manual_measurements_combined.csv"


def temporal_filter(group: pd.DataFrame, *, window: int, zero_abs: float,
                    ref_motion: float, rescue_conf: float, rescue_abs: float,
                    rescue_rel: float = 0.25) -> pd.DataFrame:
    out = group.sort_values("before_frame").copy().reset_index(drop=True)
    raw = out["raw_velocity_cm_s"].astype(float).to_numpy()
    conf = out["confidence"].astype(float).to_numpy()
    base = conf >= 0.85
    anchors = np.flatnonzero(base & (np.abs(raw) >= 10.0))
    reference = np.full(len(out), np.nan)

    for i in range(len(out)):
        left = anchors[(anchors < i) & (anchors >= i - window)]
        right = anchors[(anchors > i) & (anchors <= i + window)]
        if not len(left) or not len(right):
            continue
        li, ri = int(left[-1]), int(right[0])
        lv, rv = raw[li], raw[ri]
        if np.sign(lv) != np.sign(rv):
            continue
        allowed_jump = max(25.0, 0.35 * max(abs(lv), abs(rv)))
        if abs(lv - rv) > allowed_jump:
            continue
        fraction = (i - li) / (ri - li)
        reference[i] = lv + fraction * (rv - lv)

    static_lock = base & (np.abs(raw) <= zero_abs) & (np.abs(reference) >= ref_motion)
    tolerance = np.maximum(rescue_abs, rescue_rel * np.abs(reference))
    trend_match = (
        ~base
        & (conf >= rescue_conf)
        & np.isfinite(reference)
        & (np.sign(raw) == np.sign(reference))
        & (np.abs(raw - reference) <= tolerance)
    )
    accepted = (base & ~static_lock) | trend_match

    out["temporal_reference_cm_s"] = reference
    out["temporal_status"] = np.where(
        static_lock, "rejected_static_lock",
        np.where(trend_match, "rescued_temporal", np.where(accepted, "measured_base", "unmeasurable")),
    )
    out["velocity_temporal_cm_s"] = np.where(accepted, raw, np.nan)
    return out


def temporal_filter_cluster(group: pd.DataFrame, *, window: int, zero_abs: float,
                            ref_motion: float, rescue_conf: float, rescue_abs: float,
                            min_count: int = 5, dominance: float = 0.75,
                            rescue_rel: float = 0.25) -> pd.DataFrame:
    """Use a robust local population, including low-confidence moving candidates."""
    out = group.sort_values("before_frame").copy().reset_index(drop=True)
    raw = out["raw_velocity_cm_s"].astype(float).to_numpy()
    conf = out["confidence"].astype(float).to_numpy()
    base = conf >= 0.85
    reference = np.full(len(out), np.nan)

    for i in range(len(out)):
        lo, hi = max(0, i - window), min(len(out), i + window + 1)
        indices = np.arange(lo, hi)
        indices = indices[indices != i]
        usable = indices[(conf[indices] >= 0.65) & (np.abs(raw[indices]) >= 10.0)]
        if len(usable) < min_count:
            continue
        values = raw[usable]
        positive = values > 0
        positive_count = int(positive.sum())
        negative_count = len(values) - positive_count
        if positive_count >= min_count and positive_count / len(values) >= dominance:
            selected = values[positive]
        elif negative_count >= min_count and negative_count / len(values) >= dominance:
            selected = values[~positive]
        else:
            continue
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        if mad > max(12.0, 0.20 * abs(median)):
            continue
        reference[i] = median

    static_lock = base & (np.abs(raw) <= zero_abs) & (np.abs(reference) >= ref_motion)
    tolerance = np.maximum(rescue_abs, rescue_rel * np.abs(reference))
    trend_match = (
        ~base
        & (conf >= rescue_conf)
        & np.isfinite(reference)
        & (np.sign(raw) == np.sign(reference))
        & (np.abs(raw - reference) <= tolerance)
    )
    accepted = (base & ~static_lock) | trend_match
    out["temporal_reference_cm_s"] = reference
    out["temporal_status"] = np.where(
        static_lock, "rejected_static_lock",
        np.where(trend_match, "rescued_temporal", np.where(accepted, "measured_base", "unmeasurable")),
    )
    out["velocity_temporal_cm_s"] = np.where(accepted, raw, np.nan)
    return out


def manual_targets(auto: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset, mgroup in manual.groupby("dataset"):
        folder = Path(mgroup.iloc[0]["manual_csv_path"]).parent
        images = list_images(folder)
        resolve = image_resolver(images)
        index_by_path = {p.resolve(): i for i, p in enumerate(images)}
        agroup = auto[auto["dataset"] == dataset].sort_values("before_frame")
        auto_indices = agroup["before_frame"].to_numpy(dtype=int) - 1
        for _, row in mgroup.iterrows():
            before_path = resolve(str(row["before_file"]), str(row["before_frame"]))
            actual_index = index_by_path[before_path.resolve()]
            pos = int(np.argmin(np.abs(auto_indices - actual_index)))
            distance = abs(int(auto_indices[pos]) - actual_index)
            if distance > 5:
                continue
            records.append({
                "dataset": dataset,
                "auto_before_frame": int(agroup.iloc[pos]["before_frame"]),
                "manual_velocity": float(row["velocity"]),
            })
    rows = pd.DataFrame(records)
    return rows.groupby(["dataset", "auto_before_frame"], as_index=False).agg(
        manual_velocity=("manual_velocity", "median"),
        manual_count=("manual_velocity", "size"),
    )


def build_bins(filtered: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    predictions = filtered[[
        "dataset", "before_frame", "raw_velocity_cm_s", "confidence",
        "velocity_temporal_cm_s", "temporal_status",
    ]].copy()
    predictions = predictions.rename(columns={
        "before_frame": "auto_before_frame",
        "raw_velocity_cm_s": "raw_velocity",
        "velocity_temporal_cm_s": "temporal_velocity",
    })
    predictions["base_velocity"] = predictions["raw_velocity"].where(predictions["confidence"] >= 0.85)
    return targets.merge(predictions, on=["dataset", "auto_before_frame"], how="left")


def metrics(bins: pd.DataFrame, column: str) -> dict[str, float]:
    valid = bins[bins[column].notna()].copy()
    errors = np.abs(valid[column] - valid["manual_velocity"])
    false_zero = valid[(np.abs(valid[column]) <= 5) & (np.abs(valid["manual_velocity"]) >= 20)]
    return {
        "n": len(valid),
        "coverage": len(valid) / len(bins),
        "mae": errors.mean(),
        "median_ae": errors.median(),
        "false_zero": len(false_zero),
    }


def evaluate(auto: pd.DataFrame, targets: pd.DataFrame, params: dict, filter_func=temporal_filter) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = pd.concat(
        [filter_func(group, **params) for _, group in auto.groupby("dataset")],
        ignore_index=True,
    )
    bins = build_bins(filtered, targets)
    rows = []
    for dataset in [*sorted(bins["dataset"].unique()), "ALL"]:
        group = bins if dataset == "ALL" else bins[bins["dataset"] == dataset]
        base = metrics(group, "base_velocity")
        temporal = metrics(group, "temporal_velocity")
        rows.append({
            "dataset": dataset,
            **{f"base_{k}": v for k, v in base.items()},
            **{f"temporal_{k}": v for k, v in temporal.items()},
        })
    return filtered, pd.DataFrame(rows)


def main():
    auto = pd.read_csv(AUTO_PATH)
    manual = pd.read_csv(MANUAL_PATH)
    targets = manual_targets(auto, manual)
    candidates = []
    for window, zero_abs, ref_motion, rescue_conf, rescue_abs in itertools.product(
        [4, 6, 8, 10], [3.0, 5.0], [15.0, 20.0], [0.72, 0.75, 0.78, 0.80], [10.0, 15.0, 20.0]
    ):
        params = dict(window=window, zero_abs=zero_abs, ref_motion=ref_motion,
                      rescue_conf=rescue_conf, rescue_abs=rescue_abs)
        filtered, summary = evaluate(auto, targets, params)
        total = summary[summary["dataset"] == "ALL"].iloc[0]
        # Favor lower error and fewer false-zero matches, then greater coverage.
        score = total["temporal_mae"] + 2.0 * total["temporal_false_zero"] - 5.0 * total["temporal_coverage"]
        candidates.append((score, params, total.to_dict()))
    candidates.sort(key=lambda item: item[0])
    for score, params, total in candidates[:12]:
        print(f"score={score:.4f} params={params} total={total}")

    best_params = candidates[0][1]
    filtered, summary = evaluate(auto, targets, best_params)
    print("\nBEST", best_params)
    print(summary.to_string(index=False))
    print("\nSTATUS COUNTS")
    print(filtered.groupby(["dataset", "temporal_status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
