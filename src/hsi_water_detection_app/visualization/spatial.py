from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _optional_import_rasterio():
    try:
        import rasterio  # type: ignore
        from affine import Affine  # type: ignore
        return rasterio, Affine
    except ImportError:
        return None, None


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amin, amax):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _make_rgb_from_cube(
    cube: Optional[np.ndarray],
    rgb_bands: Sequence[int] = (3, 9, 17),
) -> Optional[np.ndarray]:
    """
    cube: (C, H, W) -> returns (H, W, 3) in [0,1]
    """
    if cube is None:
        return None

    if cube.ndim != 3:
        raise ValueError("cube must have shape (bands, height, width)")

    c, h, w = cube.shape
    band_ids = []
    for b in rgb_bands:
        if 0 <= int(b) < c:
            band_ids.append(int(b))
        else:
            band_ids.append(min(max(int(b), 0), c - 1))

    rgb = np.stack([cube[band_ids[0]], cube[band_ids[1]], cube[band_ids[2]]], axis=-1)
    rgb = _normalize01(rgb)
    return rgb


def save_probability_map(
    prob_map: Optional[np.ndarray],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_geotiff: bool = True,
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if prob_map is None:
        dummy_txt = out_path.with_suffix(".txt")
        dummy_txt.write_text("probability map is None\n", encoding="utf-8")
        print(f"[INFO] probability map missing; wrote placeholder: {dummy_txt}")
        return

    np.save(out_path, prob_map)
    print(f"[INFO] probability map saved to: {out_path}")

    if metadata is not None:
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[INFO] probability metadata saved to: {meta_path}")

    if not save_geotiff:
        return

    rasterio, Affine = _optional_import_rasterio()
    if rasterio is None or Affine is None:
        print("[INFO] rasterio/affine not installed; GeoTIFF export skipped")
        return

    if metadata is None:
        print("[INFO] metadata missing; GeoTIFF export skipped")
        return

    transform_tuple = metadata.get("transform")
    crs = metadata.get("crs")

    if transform_tuple is None or crs is None:
        print("[INFO] transform/crs missing; GeoTIFF export skipped")
        return

    transform = Affine(*transform_tuple)
    tif_path = out_path.with_suffix(".tif")

    profile = {
        "driver": "GTiff",
        "height": prob_map.shape[0],
        "width": prob_map.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    print(f"[INFO] GeoTIFF probability map saved to: {tif_path}")


def save_spatial_attention_map(
    spatial_attn: Optional[np.ndarray],
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    cube: Optional[np.ndarray] = None,
    rgb_bands: Sequence[int] = (3, 9, 17),
    save_geotiff: bool = True,
) -> None:
    """
    Saves:
      - spatial_attention.npy
      - spatial_attention.png
      - spatial_attention.tif (if georef available)
      - spatial_attention_overlay.png (if cube available)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if spatial_attn is None:
        print("[INFO] save_spatial_attention_map skipped because spatial_attn is None")
        return

    spatial_attn = _normalize01(spatial_attn)

    npy_path = out_dir / "spatial_attention.npy"
    np.save(npy_path, spatial_attn.astype(np.float32))
    print(f"[INFO] spatial attention saved to: {npy_path}")

    png_path = out_dir / "spatial_attention.png"
    plt.imsave(png_path, spatial_attn, cmap="jet", vmin=0.0, vmax=1.0)
    print(f"[INFO] spatial attention PNG saved to: {png_path}")

    if save_geotiff:
        rasterio, Affine = _optional_import_rasterio()
        if rasterio is not None and Affine is not None and metadata is not None:
            transform_tuple = metadata.get("transform")
            crs = metadata.get("crs")
            if transform_tuple is not None and crs is not None:
                transform = Affine(*transform_tuple)
                tif_path = out_dir / "spatial_attention.tif"
                profile = {
                    "driver": "GTiff",
                    "height": spatial_attn.shape[0],
                    "width": spatial_attn.shape[1],
                    "count": 1,
                    "dtype": "float32",
                    "crs": crs,
                    "transform": transform,
                    "compress": "lzw",
                }
                with rasterio.open(tif_path, "w", **profile) as dst:
                    dst.write(spatial_attn.astype(np.float32), 1)
                print(f"[INFO] spatial attention GeoTIFF saved to: {tif_path}")
            else:
                print("[INFO] transform/crs missing; spatial attention GeoTIFF skipped")
        else:
            print("[INFO] rasterio/affine unavailable; spatial attention GeoTIFF skipped")

    rgb = _make_rgb_from_cube(cube, rgb_bands=rgb_bands)
    if rgb is not None:
        overlay_path = out_dir / "spatial_attention_overlay.png"

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.imshow(spatial_attn, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"[INFO] spatial attention overlay saved to: {overlay_path}")
