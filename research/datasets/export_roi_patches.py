from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import load_hsi_window
from hsi_water_detection_app.data.preprocessing import normalize_cube

EXPECTED_PATCH_H = 64
EXPECTED_PATCH_W = 64


def apply_sensor_band_policy(cube: np.ndarray, sensor: str) -> np.ndarray:
    if cube is None:
        raise ValueError("cube is None")

    sensor = (sensor or "").lower()

    if sensor == "hyperion":
        if cube.shape[0] == 242:
            bad = set(HYPERION_BAD_BANDS_0BASED)
            keep = [i for i in range(cube.shape[0]) if i not in bad]
            cube = cube[keep]
            print(f"[INFO] apply_sensor_band_policy: Hyperion 242 -> {cube.shape[0]} bands")
        else:
            print(f"[INFO] apply_sensor_band_policy: skip bad-band removal because bands={cube.shape[0]}")
    else:
        print(f"[INFO] apply_sensor_band_policy: no special rule for sensor={sensor}")

    return cube


def is_placeholder_row(row: dict) -> bool:
    fields = [
        row.get("row_start"),
        row.get("row_stop"),
        row.get("col_start"),
        row.get("col_stop"),
    ]
    return any("TODO" in str(x) for x in fields)


def export_one(row: dict, output_dir: str | Path) -> None:
    sample_id = row["sample_id"]
    input_path = row["input_path"]
    sensor = row.get("sensor", "hyperion")

    row_start = int(row["row_start"])
    row_stop = int(row["row_stop"])
    col_start = int(row["col_start"])
    col_stop = int(row["col_stop"])

    cube, meta = load_hsi_window(
        input_path,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
    )

    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape={cube.shape}")

    if cube.shape[1] != EXPECTED_PATCH_H or cube.shape[2] != EXPECTED_PATCH_W:
        raise ValueError(
            f"patch shape mismatch: got {cube.shape}, "
            f"expected bands x {EXPECTED_PATCH_H} x {EXPECTED_PATCH_W}"
        )

    cube = apply_sensor_band_policy(cube, sensor=sensor)
    cube = normalize_cube(cube)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{sample_id}.npz"
    json_path = output_dir / f"{sample_id}.json"

    label = row["label"]
    confidence = float(row.get("confidence", 1.0))
    split = row.get("split", "train")

    np.savez_compressed(
        npz_path,
        cube=cube.astype(np.float32),
        label=np.array(label),
        confidence=np.array(confidence, dtype=np.float32),
        split=np.array(split),
        sample_id=np.array(sample_id),
    )

    # Preserve any extra manifest fields in the sidecar json so downstream
    # dataset/eval code can access regenerated metadata without depending on
    # the original CSV at runtime.
    meta_out = dict(row)
    meta_out.update(
        {
            "sample_id": sample_id,
            "scene_id": row.get("scene_id"),
            "input_path": input_path,
            "sensor": sensor,
            "row_start": row_start,
            "row_stop": row_stop,
            "col_start": col_start,
            "col_stop": col_stop,
            "label": label,
            "confidence": confidence,
            "split": split,
            "source": row.get("source"),
            "notes": row.get("notes"),
            "cube_shape": list(cube.shape),
        }
    )
    json_path.write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] exported sample: {sample_id} -> {npz_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    exported = 0
    skipped = 0

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        sample_id = row.get("sample_id", "<unknown>")
        try:
            if is_placeholder_row(row):
                print(f"[WARN] skipped placeholder row: {sample_id}")
                skipped += 1
                continue

            export_one(row, args.output_dir)
            exported += 1
        except Exception as e:
            print(f"[WARN] skipped invalid row: {sample_id} ({e!r})")
            skipped += 1

    print(f"[INFO] exported {exported} ROI patches to {args.output_dir}")
    print(f"[INFO] skipped {skipped} rows")


if __name__ == "__main__":
    main()
