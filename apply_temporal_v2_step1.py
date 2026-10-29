from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "all_patterns_step1_temporal_v2_20260831"
RAW_INPUT = OUTPUT_DIR / "raw" / "all_unlabeled_velocity.csv"
MANUAL_INPUT = ROOT / "manual_series_combined_20260831" / "manual_measurements_combined.csv"
PER_SEQUENCE_DIR = OUTPUT_DIR / "per_sequence"
GRAPH_DIR = OUTPUT_DIR / "graphs"
OVERVIEW_DIR = OUTPUT_DIR / "overview"

sys.path.insert(0, str(ROOT / "manual_series_combined_20260831"))
from evaluate_temporal_filter import temporal_filter_cluster
import apply_temporal_v2_all_patterns as plotting


# Same physical +/-0.125 s neighborhood as v2 at 5-frame cadence.
PARAMS = {
    "window": 30,
    "zero_abs": 3.0,
    "ref_motion": 20.0,
    "rescue_conf": 0.75,
    "rescue_abs": 10.0,
    "min_count": 25,
    "dominance": 0.75,
}


def main() -> None:
    for folder in (OUTPUT_DIR, PER_SEQUENCE_DIR, GRAPH_DIR, OVERVIEW_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_INPUT)
    manual = pd.read_csv(MANUAL_INPUT)
    manual_datasets = set(manual["dataset"].unique())
    raw["manual_data_available"] = raw["dataset"].isin(manual_datasets)
    raw["original_status"] = raw["status"]
    raw["original_velocity_cm_s"] = raw["velocity_cm_s"]
    raw["source_group"] = "step1_image_pair_estimation"
    raw["fps"] = 240.0
    raw["confidence_threshold"] = 0.85

    groups = []
    for index, (dataset, group) in enumerate(raw.groupby("dataset", sort=True), start=1):
        filtered = temporal_filter_cluster(group, **PARAMS)
        filtered["dataset"] = dataset
        groups.append(filtered)
        if index == 1 or index % 10 == 0 or index == raw["dataset"].nunique():
            print(f"Temporal filtering {index}/{raw['dataset'].nunique()}", flush=True)
    result = pd.concat(groups, ignore_index=True, sort=False)
    result["velocity_final_cm_s"] = result["velocity_temporal_cm_s"]
    result["final_status"] = np.where(result["velocity_final_cm_s"].notna(), "measured", "unmeasurable")
    result["temporal_residual_cm_s"] = np.abs(
        result["raw_velocity_cm_s"] - result["temporal_reference_cm_s"]
    )
    result = result.sort_values(["dataset", "before_frame"]).reset_index(drop=True)

    output_csv = OUTPUT_DIR / "all_patterns_velocity_step1_temporal_v2.csv"
    summary_csv = OUTPUT_DIR / "sequence_summary_step1_temporal_v2.csv"
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = plotting.build_summary(result)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    plotting.OVERVIEW_DIR = OVERVIEW_DIR
    for index, (dataset, group) in enumerate(result.groupby("dataset", sort=True), start=1):
        group.to_csv(PER_SEQUENCE_DIR / f"{dataset}_velocity_step1_temporal_v2.csv", index=False, encoding="utf-8-sig")
        plotting.plot_sequence(group, manual[manual["dataset"] == dataset], GRAPH_DIR / f"{dataset}_step1_temporal_v2.png")
        if index == 1 or index % 10 == 0 or index == result["dataset"].nunique():
            print(f"Graphs {index}/{result['dataset'].nunique()}", flush=True)
    plotting.plot_overviews(result, manual)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "local moving-population temporal consistency v2",
        "parameters": PARAMS,
        "fps": 240.0,
        "measurement_start_step_frames": 1,
        "measurement_start_interval_s": 1 / 240,
        "adaptive_pair_gap_frames": [1, 4],
        "source_raw_csv": str(RAW_INPUT),
        "datasets": int(result["dataset"].nunique()),
        "rows": int(len(result)),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"FINAL_OUTPUT {output_csv.resolve()}", flush=True)


if __name__ == "__main__":
    main()
