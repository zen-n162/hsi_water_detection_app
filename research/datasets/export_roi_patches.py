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


def has_todo_placeholder(row: dict[str, str]) -> bool:
    for k in ["row_start", "row_stop", "col_start", "col_stop"]:
        if "TODO_" in str(row.get(k, "")):
            return True
    return False


def parse_int_field(row: dict[str, str], key: str) -> int:
    return int(str(row[key]).strip())


def export_one(row: dict[str, str], output_dir: Path) -> Path:
    input_path = row["input_path"]
    row_start = parse_int_field(row, "row_start")
    row_stop = parse_int_field(row, "row_stop")
    col_start = parse_int_field(row, "col_start")
    col_stop = parse_int_field(row, "col_stop")
    sensor = row.get("sensor", "hyperion")

    cube, meta = load_hsi_window(
        input_path,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
    )

    if cube is None:
        raise RuntimeError(f"failed to load cube from {input_path}")

    if cube.shape[1] != EXPECTED_PATCH_H or cube.shape[2] != EXPECTED_PATCH_W:
        raise ValueError(
            f"patch shape mismatch: got {cube.shape}, expected bands x {EXPECTED_PATCH_H} x {EXPECTED_PATCH_W}"
        )

    cube = apply_sensor_band_policy(cube, sensor=sensor)
    cube = normalize_cube(cube)

    sample_id = row["sample_id"]
    out_npz = output_dir / f"{sample_id}.npz"
    out_json = output_dir / f"{sample_id}.json"

    np.savez_compressed(
        out_npz,
        cube=cube.astype(np.float32),
        label=np.array(row["label"]),
        confidence=np.float32(row["confidence"]),
    )

    out_json.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "scene_id": row["scene_id"],
                "input_path": input_path,
                "sensor": sensor,
                "roi": {
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "col_start": col_start,
                    "col_stop": col_stop,
                },
                "label": row["label"],
                "confidence": float(row["confidence"]),
                "meta": meta,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] exported sample: {sample_id} -> {out_npz}")
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ROI patches from wetness manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("annotations/manifests/wetness_manifest.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/processed/wetness_pretrain"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    with args.manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id", "(unknown)")

            if has_todo_placeholder(row):
                print(f"[WARN] skipped placeholder row: {sample_id}")
                skipped += 1
                continue

            try:
                export_one(row, args.output_dir)
                exported += 1
            except Exception as e:
                print(f"[WARN] skipped invalid row: {sample_id} ({e!r})")
                skipped += 1

    print(f"[INFO] exported {exported} ROI patches to {args.output_dir}")
    print(f"[INFO] skipped {skipped} rows")


if __name__ == "__main__":
    main()
