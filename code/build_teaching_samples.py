"""Build a small, timestamp-aligned classroom sample set from real project data.

This script does not train a model.  It matches each 119-feature row to the
labelled NPZ window whose final timestamp is identical.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "data" / "teaching_samples"

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data/datasets/logbot_data/feature_extraction/acc_features/omizunagidori"
)
NPZ_ROOT = PROJECT_ROOT / "data/datasets/npz_format/labelled/omizunagidori"

LABEL_TO_CLASS = {
    "preening": "stationary",
    "stationary": "stationary",
    "bathing": "bathing",
    "flight_take_off": "flight_take_off",
    "flight_cruising": "flight_cruising",
    "foraging_dive": "foraging_dive",
    "surface_seizing": "dipping",
}

DISPLAY_NAME = {
    "stationary": "Stationary",
    "bathing": "Bathing",
    "flight_take_off": "Take-off",
    "flight_cruising": "Cruising Flight",
    "foraging_dive": "Foraging Dive",
    "dipping": "Dipping",
}

# A compact set of LOIO test individuals. Together they provide at least ten
# examples of every teaching class while requiring only three model pairs.
TARGETS = {
    "stationary": [("OM1901", 10)],
    "bathing": [("OM2211", 10)],
    "flight_take_off": [("OM1901", 9), ("OM2214", 1)],
    "flight_cruising": [("OM1901", 10)],
    "foraging_dive": [("OM1901", 10)],
    "dipping": [("OM1901", 10)],
}


def timestamp_key(value: float) -> int:
    """Convert a Unix timestamp to integer milliseconds for stable matching."""
    return int(round(float(value) * 1000))


def load_feature_rows(animal_id: str) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_ROOT / f"{animal_id}.csv")
    frame["teaching_class"] = frame["label"].map(LABEL_TO_CLASS)
    frame["timestamp_ms"] = frame["unixtime"].map(timestamp_key)
    return frame


def index_npz_windows(animal_id: str) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in sorted((NPZ_ROOT / animal_id).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            timestamp = archive["timestamp"]
            index[timestamp_key(timestamp.reshape(-1)[-1])] = path
    return index


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_features: list[pd.Series] = []
    curve_rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []

    for teaching_class, requests in TARGETS.items():
        class_count = 0
        for animal_id, requested in requests:
            features = load_feature_rows(animal_id)
            candidates = features[features["teaching_class"] == teaching_class]
            npz_index = index_npz_windows(animal_id)
            matched = candidates[candidates["timestamp_ms"].isin(npz_index)].head(requested)
            if len(matched) != requested:
                raise RuntimeError(
                    f"{animal_id}/{teaching_class}: requested {requested}, matched {len(matched)}"
                )

            for _, feature_row in matched.iterrows():
                class_count += 1
                sample_id = f"{teaching_class}-{class_count:02d}"
                npz_path = npz_index[int(feature_row["timestamp_ms"])]
                with np.load(npz_path, allow_pickle=False) as archive:
                    x = archive["X"].reshape(50, 3)
                    timestamp = archive["timestamp"].reshape(50)

                row = feature_row.copy()
                row["sample_id"] = sample_id
                selected_features.append(row)
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "animal_id": animal_id,
                        "teaching_class": teaching_class,
                        "display_name": DISPLAY_NAME[teaching_class],
                        "source_label": feature_row["label"],
                        "label_id": int(feature_row["label_id"]),
                        "start_unixtime": float(timestamp[0]),
                        "end_unixtime": float(timestamp[-1]),
                        "sampling_rate_hz": 25,
                        "window_points": 50,
                        "source_npz": str(npz_path.relative_to(PROJECT_ROOT)),
                        "source_feature_csv": str(
                            (FEATURE_ROOT / f"{animal_id}.csv").relative_to(PROJECT_ROOT)
                        ),
                    }
                )
                for point, (ts, values) in enumerate(zip(timestamp, x, strict=True)):
                    curve_rows.append(
                        {
                            "sample_id": sample_id,
                            "point": point,
                            "unixtime": float(ts),
                            "acc_x": float(values[0]),
                            "acc_y": float(values[1]),
                            "acc_z": float(values[2]),
                        }
                    )

    metadata_frame = pd.DataFrame(metadata)
    curves_frame = pd.DataFrame(curve_rows)
    features_frame = pd.DataFrame(selected_features)
    feature_order = ["sample_id"] + [c for c in features_frame.columns if c != "sample_id"]
    features_frame = features_frame[feature_order]

    metadata_frame.to_csv(OUTPUT_DIR / "samples_metadata.csv", index=False, encoding="utf-8-sig")
    curves_frame.to_csv(OUTPUT_DIR / "samples_acceleration.csv", index=False, encoding="utf-8-sig")
    features_frame.to_csv(OUTPUT_DIR / "samples_features_119.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "source": "Real omizunagidori project data; no synthetic samples",
        "matching_rule": "animal_id + final window Unix timestamp (millisecond precision)",
        "sample_count": len(metadata_frame),
        "points_per_sample": 50,
        "sampling_rate_hz": 25,
        "class_counts": metadata_frame["teaching_class"].value_counts().sort_index().to_dict(),
        "animals": sorted(metadata_frame["animal_id"].unique().tolist()),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_pairs = sorted(metadata_frame["animal_id"].unique())
    for animal_id in model_pairs:
        xgb_source = (
            PROJECT_ROOT
            / "data/model-output/I03/ex-d90/om-50/xgboost/"
            / "xgboost-feats-119-smote-true/seed0"
            / f"{animal_id}.pickle"
        )
        dcl_source = (
            PROJECT_ROOT
            / "data/model-output/I03/ex-d00/om-50/dcl-sa/seed0"
            / animal_id
        )
        shutil.copy2(xgb_source, PACKAGE_ROOT / "models/xgboost" / xgb_source.name)
        target = PACKAGE_ROOT / "models/dcl_sa" / animal_id
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dcl_source / "checkpoints_dir/best_model_weights.pt", target / "best_model_weights.pt")
        shutil.copy2(dcl_source / "config.yaml", target / "config.yaml")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
