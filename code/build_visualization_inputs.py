"""Create lightweight, real CSV inputs for classroom Web visualizations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PACKAGE_ROOT / "data/teaching_samples"
OUTPUT_DIR = PACKAGE_ROOT / "data/visualization_inputs"
RAW_SOURCE = PROJECT_ROOT / "data/datasets/raw-data/omizunagidori/Omizunagidori2019_raw_data_C5_lbs0001.csv"

CLASSES = ["stationary", "bathing", "flight_take_off", "flight_cruising", "foraging_dive", "dipping"]
RAW_LABELS = {
    "stationary": "stationary",
    "bathing": "bathing",
    "flight_take_off": "flight_take_off",
    "flight_cruising": "flight_cruising",
    "foraging_dive": "foraging_dive",
    "dipping": "surface_seizing",
}


def first_contiguous_run(frame: pd.DataFrame, label: str, points: int = 124) -> pd.DataFrame:
    match = frame["label"].eq(label)
    groups = match.ne(match.shift()).cumsum()
    for _, block in frame[match].groupby(groups[match], sort=False):
        if len(block) >= points:
            return block.iloc[:points].copy()
    raise RuntimeError(f"No contiguous {points}-point raw segment found for {label}")


def build() -> None:
    web_dir = OUTPUT_DIR / "web_ready_25hz"
    raw_dir = OUTPUT_DIR / "raw_segments_31hz"
    web_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SAMPLE_DIR / "samples_metadata.csv")
    acceleration = pd.read_csv(SAMPLE_DIR / "samples_acceleration.csv")
    combined = []
    for teaching_class in CLASSES:
        sample = metadata[metadata["teaching_class"] == teaching_class].iloc[0]
        curve = acceleration[acceleration["sample_id"] == sample["sample_id"]].copy()
        curve.insert(0, "timestamp", pd.to_datetime(curve["unixtime"], unit="s", utc=True))
        curve["label"] = sample["source_label"]
        curve["animal_id"] = sample["animal_id"]
        web_frame = curve[["timestamp", "acc_x", "acc_y", "acc_z", "label", "sample_id", "animal_id"]]
        web_frame.to_csv(web_dir / f"{teaching_class}_example.csv", index=False, encoding="utf-8-sig")
        combined.append(web_frame)
    pd.concat(combined, ignore_index=True).to_csv(
        OUTPUT_DIR / "six_classes_web_input.csv", index=False, encoding="utf-8-sig"
    )
    raw = pd.read_csv(RAW_SOURCE, usecols=["timestamp", "acc_x", "acc_y", "acc_z", "label"], low_memory=False)
    for teaching_class, source_label in RAW_LABELS.items():
        segment = first_contiguous_run(raw, source_label)
        segment.insert(0, "animal_id", "OM1901")
        segment.insert(1, "teaching_class", teaching_class)
        segment.to_csv(raw_dir / f"{teaching_class}_raw_segment.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    build()

