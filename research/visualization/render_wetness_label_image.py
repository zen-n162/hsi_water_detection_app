from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import pyplot as plt


LABEL_TO_ID = {
    "background": 0,
    "dry": 1,
    "wet": 2,
    "uncertain": 3,
}

# RGB preview colors
LABEL_TO_RGB = {
    0: (0, 0, 0),          # background: black
    1: (255, 165, 0),      # dry: orange
    2: (0, 255, 255),      # wet: cyan
    3: (255, 0, 255),      # uncertain: magenta
}


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return rows


def filter_rows(rows: list[dict[str, str]], split: str | None) -> list[dict[str, str]]:
    if split is None:
        return rows
    return [r for r in rows if str(r.get("split", "")).strip() == split]


def build_label_mask(
    rows: list[dict[str, str]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    for row in rows:
        label_name = str(row["label"]).strip().lower()
        if label_name not in LABEL_TO_ID:
            print(f"[WARN] unknown label skipped: {label_name}")
            continue

        r0 = int(row["row_start"])
        r1 = int(row["row_stop"])
        c0 = int(row["col_start"])
        c1 = int(row["col_stop"])

        # clip to image bounds
        r0 = max(0, min(r0, height))
        r1 = max(0, min(r1, height))
        c0 = max(0, min(c0, width))
        c1 = max(0, min(c1, width))

        if r0 >= r1 or c0 >= c1:
            print(f"[WARN] invalid ROI skipped: {row.get('sample_id', 'unknown')}")
            continue

        mask[r0:r1, c0:c1] = LABEL_TO_ID[label_name]

    return mask


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in LABEL_TO_RGB.items():
        rgb[mask == label_id] = color
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ground-truth wetness label image from manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv"),
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--reference_tif",
        type=Path,
        required=True,
        help="Reference raster to get height/width/profile",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=[None, "train", "val", "test"],
        help="Optional split filter",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/label_visualization_v3"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest_rows(args.manifest)
    rows = filter_rows(rows, args.split)

    if not rows:
        raise RuntimeError(
            f"No rows found after split filtering. split={args.split!r}"
        )

    with rasterio.open(args.reference_tif) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    mask = build_label_mask(rows, height=height, width=width)
    rgb = mask_to_rgb(mask)

    split_suffix = args.split if args.split else "all"
    tif_path = args.output_dir / f"wetness_label_mask_{split_suffix}.tif"
    png_path = args.output_dir / f"wetness_label_preview_{split_suffix}.png"
    legend_path = args.output_dir / f"wetness_label_legend_{split_suffix}.json"

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(tif_path, "w", **out_profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "wetness_label_mask")

    plt.imsave(png_path, rgb)

    legend = {
        "label_to_id": LABEL_TO_ID,
        "label_to_rgb": {str(k): list(v) for k, v in LABEL_TO_RGB.items()},
        "split": args.split,
        "reference_tif": str(args.reference_tif),
        "manifest": str(args.manifest),
        "output_mask_tif": str(tif_path),
        "output_preview_png": str(png_path),
        "image_shape": [height, width],
        "crs": str(crs) if crs is not None else None,
        "transform": list(transform) if transform is not None else None,
        "n_rows_used": len(rows),
    }
    legend_path.write_text(
        json.dumps(legend, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] rows used: {len(rows)}")
    print(f"[INFO] saved label mask tif: {tif_path}")
    print(f"[INFO] saved preview png: {png_path}")
    print(f"[INFO] saved legend json: {legend_path}")


if __name__ == "__main__":
    main()
