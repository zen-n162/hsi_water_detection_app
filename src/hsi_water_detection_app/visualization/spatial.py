from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _optional_import_rasterio():
    try:
        import rasterio  # type: ignore
        from affine import Affine  # type: ignore
        return rasterio, Affine
    except ImportError:
        return None, None


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
