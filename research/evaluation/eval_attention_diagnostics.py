from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize spectral attention CSV outputs")
    parser.add_argument(
        "--input_glob",
        type=str,
        default="outputs/**/*.csv",
        help="Glob to spectral attention CSV files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("experiments/metrics/attention_diagnostics.csv"),
    )
    args = parser.parse_args()

    files = sorted(Path(".").glob(args.input_glob))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        if "spectral_attention" not in f.name:
            continue
        try:
            arr = np.genfromtxt(f, delimiter=",", names=True)
            if arr.size == 0:
                continue
            attn = arr[arr.dtype.names[-1]]
            rows.append(
                {
                    "path": str(f),
                    "mean_attention": float(np.mean(attn)),
                    "max_attention": float(np.max(attn)),
                    "argmax_index": int(np.argmax(attn)),
                }
            )
        except Exception as e:
            rows.append({"path": str(f), "error": repr(e)})

    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["path"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] wrote attention diagnostics to {args.output_csv}")


if __name__ == "__main__":
    main()
