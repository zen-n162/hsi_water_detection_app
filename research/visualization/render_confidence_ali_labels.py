from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_single_band(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    return arr, profile


def build_class_view(conf: np.ndarray, nodata_value: int = 255) -> np.ndarray:
    """
    Convert confidence_ali values into display classes.

    output classes:
      0 -> dry / background
      1 -> wet tier 1
      2 -> wet tier 2
      3 -> wet tier 3
      4 -> ignore / invalid
    """
    out = np.full(conf.shape, 4, dtype=np.uint8)
    out[conf == 0] = 0
    out[conf == 1] = 1
    out[conf == 2] = 2
    out[conf == 3] = 3
    out[conf == nodata_value] = 4
    return out


def save_label_png(
    class_arr: np.ndarray,
    output_png: Path,
    title: str,
) -> None:
    cmap = ListedColormap(
        [
            "#4d4d4d",  # dry
            "#6baed6",  # wet tier 1
            "#3182bd",  # wet tier 2
            "#08519c",  # wet tier 3
            "#f0f0f0",  # ignore
        ]
    )
    bounds = np.arange(-0.5, 5.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(class_arr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(
        [
            "dry(0)",
            "wet_tier1(1)",
            "wet_tier2(2)",
            "wet_tier3(3)",
            "ignore(255)",
        ]
    )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def save_preview_npz(class_arr: np.ndarray, output_npz: Path) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, label_image=class_arr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render confidence_ali.tif into a visible label image."
    )
    ap.add_argument(
        "--input_tif",
        type=Path,
        required=True,
        help="Path to confidence_ali.tif",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/label_visualization_v3"),
        help="Directory to save rendered outputs",
    )
    ap.add_argument(
        "--title",
        default="confidence_ali label visualization",
        help="Figure title",
    )
    args = ap.parse_args()

    conf, profile = load_single_band(args.input_tif)
    class_arr = build_class_view(conf, nodata_value=profile.get("nodata", 255) or 255)

    png_path = args.output_dir / "confidence_ali_labels.png"
    npz_path = args.output_dir / "confidence_ali_labels_preview.npz"
    summary_path = args.output_dir / "confidence_ali_labels_summary.txt"

    save_label_png(class_arr, png_path, args.title)
    save_preview_npz(class_arr, npz_path)

    unique, counts = np.unique(conf, return_counts=True)
    summary_lines = [
        f"input_tif={args.input_tif}",
        f"shape={conf.shape}",
        f"dtype={conf.dtype}",
        "value_counts:",
    ]
    summary_lines += [f"  {int(v)}: {int(c)}" for v, c in zip(unique, counts)]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] saved png: {png_path}")
    print(f"[INFO] saved preview npz: {npz_path}")
    print(f"[INFO] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
