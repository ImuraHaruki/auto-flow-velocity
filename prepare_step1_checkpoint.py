from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
UNLABELED = ROOT / "all_unlabeled_results_20260829" / "all_unlabeled_velocity.csv"
MANUAL = ROOT / "manual_series_combined_20260831" / "manual_series_auto_velocity_combined.csv"
OUTPUT = ROOT / "all_patterns_step1_temporal_v2_20260831"
RAW_DIR = OUTPUT / "raw"
CHECKPOINT = RAW_DIR / "checkpoint.csv"
CALIBRATION_MAP = OUTPUT / "calibration_map.json"

FIELDS = [
    "dataset", "before_frame", "after_frame", "before_file", "after_file",
    "time_s", "delta_time_s", "sample_step_frames", "frame_gap",
    "cm_per_pixel", "calibration_group", "displacement_px", "velocity_cm_s",
    "raw_velocity_cm_s", "status", "support", "trusted_matches",
    "support_fraction", "direction_consistency", "spread_px", "confidence",
    "confidence_label",
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 0:
        print(f"Checkpoint already exists; preserving {CHECKPOINT}")
        return
    unlabeled = pd.read_csv(UNLABELED)
    manual = pd.read_csv(MANUAL)
    unlabeled["calibration_group"] = unlabeled["calibration_group"].fillna("existing_group_calibration")
    manual["calibration_group"] = "manual_dataset_specific"
    combined = pd.concat([unlabeled, manual], ignore_index=True, sort=False)
    combined["sample_step_frames"] = 1
    combined = combined.sort_values(["dataset", "before_frame"])

    calibration = {}
    for dataset, group in combined.groupby("dataset"):
        calibration[dataset] = {
            "cm_per_pixel": float(group.iloc[0]["cm_per_pixel"]),
            "calibration_group": str(group.iloc[0]["calibration_group"]),
        }
    CALIBRATION_MAP.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = combined[FIELDS].where(pd.notna(combined[FIELDS]), "").to_dict("records")
    with CHECKPOINT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Seeded {len(rows)} reusable rows across {len(calibration)} datasets")
    print(CHECKPOINT)
    print(CALIBRATION_MAP)


if __name__ == "__main__":
    main()
