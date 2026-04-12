from __future__ import annotations

import argparse
import csv
from pathlib import Path


HEADER = [
    "sample_id",
    "scene_id",
    "input_path",
    "sensor",
    "row_start",
    "row_stop",
    "col_start",
    "col_stop",
    "label",
    "confidence",
    "source",
    "split",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize or extend wetness manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/manifests/wetness_manifest.csv"),
        help="Path to wetness manifest CSV",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create manifest with header if it does not exist",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.init and not args.output.exists():
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        print(f"[INFO] initialized manifest: {args.output}")
        return

    print("[INFO] build_wetness_manifest.py placeholder")
    print("[INFO] later: ingest UI annotations / proxy labels / external ROI tables")


if __name__ == "__main__":
    main()
