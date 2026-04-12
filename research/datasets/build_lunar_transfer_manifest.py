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
    "tag",
    "confidence",
    "source",
    "split",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize lunar transfer manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/manifests/lunar_transfer_manifest.csv"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        print(f"[INFO] initialized lunar transfer manifest: {args.output}")
    else:
        print(f"[INFO] already exists: {args.output}")


if __name__ == "__main__":
    main()
