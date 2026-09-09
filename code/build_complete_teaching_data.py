"""Build the complete timestamp-aligned teaching dataset from project data.

Outputs all feature windows (42,526 in the current project), their 50-point
acceleration windows, per-window metadata, and a reproducibility manifest.
No model training or source-data modification is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "data" / "_complete_teaching_samples_build"
FEATURE_ROOT = PROJECT_ROOT / "data/datasets/logbot_data/feature_extraction/acc_features/omizunagidori"
ACC_ROOT = PROJECT_ROOT / "data/datasets/preprocessed_data/omizunagidori"

LABEL_TO_CLASS = {
    "preening": "stationary",
    "stationary": "stationary",
    "bathing": "bathing",
    "flight_take_off": "flight_take_off",
    "flight_cruising": "flight_cruising",
    "foraging_dive": "foraging_dive",
    "surface_seizing": "dipping",
}


def timestamp_ms(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series) * 1000).round().astype("int64")


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    acceleration_path = OUTPUT_DIR / "samples_acceleration.csv"
    wrote_acceleration_header = False
    sample_number = 0

    for feature_path in sorted(FEATURE_ROOT.glob("*.csv")):
        animal_id = feature_path.stem
        features = pd.read_csv(feature_path, low_memory=False)
        acceleration = pd.read_csv(
            ACC_ROOT / f"{animal_id}.csv",
            usecols=["datetime", "unixtime", "acc_x", "acc_y", "acc_z", "label"],
            low_memory=False,
        )
        acceleration["timestamp_ms"] = timestamp_ms(acceleration["unixtime"])
        end_index = dict(zip(acceleration["timestamp_ms"], acceleration.index, strict=False))

        feature_output_rows = []
        metadata_rows = []
        acceleration_rows = []
        for _, feature_row in features.iterrows():
            end_ms = int(round(float(feature_row["unixtime"]) * 1000))
            if end_ms not in end_index:
                raise RuntimeError(f"No acceleration timestamp match: {animal_id}/{end_ms}")
            end = int(end_index[end_ms])
            start = end - 49
            if start < 0:
                raise RuntimeError(f"Incomplete 50-point window: {animal_id}/{end_ms}")
            window = acceleration.iloc[start : end + 1].copy()
            if len(window) != 50:
                raise RuntimeError(f"Window length is not 50: {animal_id}/{end_ms}")

            sample_number += 1
            sample_id = f"OM-{sample_number:06d}"
            teaching_class = LABEL_TO_CLASS[str(feature_row["label"])]
            feature_record = feature_row.to_dict()
            feature_record["sample_id"] = sample_id
            feature_output_rows.append(feature_record)
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "animal_id": animal_id,
                    "teaching_class": teaching_class,
                    "source_label": str(feature_row["label"]),
                    "label_id": int(feature_row["label_id"]),
                    "start_unixtime": float(window["unixtime"].iloc[0]),
                    "end_unixtime": float(window["unixtime"].iloc[-1]),
                    "start_datetime": str(window["datetime"].iloc[0]),
                    "end_datetime": str(window["datetime"].iloc[-1]),
                    "sampling_rate_hz": 25,
                    "window_points": 50,
                    "source_acceleration_csv": str((ACC_ROOT / f"{animal_id}.csv").relative_to(PROJECT_ROOT)),
                    "source_feature_csv": str(feature_path.relative_to(PROJECT_ROOT)),
                }
            )
            window.insert(0, "sample_id", sample_id)
            window.insert(1, "point", range(50))
            window.insert(2, "animal_id", animal_id)
            acceleration_rows.append(
                window[["sample_id", "point", "animal_id", "datetime", "unixtime", "acc_x", "acc_y", "acc_z", "label"]]
            )

        pd.concat(acceleration_rows, ignore_index=True).to_csv(
            acceleration_path,
            mode="a",
            header=not wrote_acceleration_header,
            index=False,
            encoding="utf-8-sig" if not wrote_acceleration_header else "utf-8",
        )
        wrote_acceleration_header = True
        metadata_parts.append(pd.DataFrame(metadata_rows))
        feature_parts.append(pd.DataFrame(feature_output_rows))
        print(f"{animal_id}: {len(features)} windows")

    metadata = pd.concat(metadata_parts, ignore_index=True)
    feature_data = pd.concat(feature_parts, ignore_index=True)
    feature_data = feature_data[["sample_id"] + [c for c in feature_data.columns if c != "sample_id"]]
    metadata.to_csv(OUTPUT_DIR / "samples_metadata.csv", index=False, encoding="utf-8-sig")
    feature_data.to_csv(OUTPUT_DIR / "samples_features_119.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "dataset_scope": "all labelled omizunagidori feature windows available in the project",
        "synthetic_data": False,
        "sample_count": int(len(metadata)),
        "acceleration_row_count": int(len(metadata) * 50),
        "points_per_sample": 50,
        "sampling_rate_hz": 25,
        "feature_table_columns_including_metadata": int(len(feature_data.columns) - 1),
        "animal_count": int(metadata["animal_id"].nunique()),
        "animals": sorted(metadata["animal_id"].unique().tolist()),
        "class_counts": metadata["teaching_class"].value_counts().sort_index().to_dict(),
        "source_feature_directory": str(FEATURE_ROOT.relative_to(PROJECT_ROOT)),
        "source_acceleration_directory": str(ACC_ROOT.relative_to(PROJECT_ROOT)),
        "matching_rule": "animal_id + feature unixtime equals the final point of a 50-point preprocessed acceleration window",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
