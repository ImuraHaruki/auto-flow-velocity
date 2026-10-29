from __future__ import annotations

from pathlib import Path
import sys

from run_all_unlabeled_sequences import main as run_raw
from apply_temporal_v2_step1 import main as run_temporal


PROJECT = Path(__file__).resolve().parent
FRAMES_ROOT = Path(r"E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames")
OUTPUT = PROJECT / "all_patterns_step1_temporal_v2_20260831"
RAW_OUTPUT = OUTPUT / "raw"
CALIBRATION_MAP = OUTPUT / "calibration_map.json"


def main() -> None:
    if "--check" in sys.argv:
        print(f"frames_root_exists={FRAMES_ROOT.exists()}")
        print(FRAMES_ROOT)
        return
    sys.argv = [
        "run_all_unlabeled_sequences.py",
        str(FRAMES_ROOT),
        "--output-dir", str(RAW_OUTPUT),
        "--step", "1",
        "--max-gap", "4",
        "--minimum-confidence", "0.85",
        "--workers", "12",
        "--progress-every", "100",
        "--include-manual",
        "--calibration-map", str(CALIBRATION_MAP),
    ]
    run_raw()
    run_temporal()


if __name__ == "__main__":
    main()
