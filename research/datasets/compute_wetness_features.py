from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_cube(cube: np.ndarray) -> dict:
    mean_spectrum = cube.mean(axis=(1, 2))
    std_spectrum = cube.std(axis=(1, 2))
    return {
        "bands": int(cube.shape[0]),
        "height": int(cube.shape[1]),
        "width": int(cube.shape[2]),
        "mean_spectrum": mean_spectrum.tolist(),
        "std_spectrum": std_spectrum.tolist(),
        "global_mean": float(cube.mean()),
        "global_std": float(cube.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple wetness features from exported ROI patches")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("datasets/processed/wetness_pretrain"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/interim/roi_exports"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npz"))
    for f in files:
        data = np.load(f, allow_pickle=True)
        cube = data["cube"]
        summary = summarize_cube(cube)
        out_json = args.output_dir / f"{f.stem}_features.json"
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] feature summaries written to {args.output_dir}")


if __name__ == "__main__":
    main()
