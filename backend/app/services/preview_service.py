from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amin, amax):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def make_probability_png(
    prob_map_npy: str | Path,
    out_png: str | Path,
) -> Optional[Path]:
    prob_map_npy = Path(prob_map_npy)
    out_png = Path(out_png)

    if not prob_map_npy.exists():
        return None

    arr = np.load(prob_map_npy)
    arr = _normalize01(arr)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png, arr, cmap="viridis", vmin=0.0, vmax=1.0)
    return out_png


def make_pseudocolor_png(
    spatial_overlay_png: str | Path,
    spatial_attention_png: str | Path,
    out_png: str | Path,
) -> Optional[Path]:
    """
    Temporary practical implementation:
    if true pseudocolor source is not separately saved yet,
    derive a background-like preview from the spatial overlay or fallback image.

    Priority:
      1) spatial_attention_overlay.png
      2) spatial_attention.png
    """
    spatial_overlay_png = Path(spatial_overlay_png)
    spatial_attention_png = Path(spatial_attention_png)
    out_png = Path(out_png)

    src = None
    if spatial_overlay_png.exists():
        src = spatial_overlay_png
    elif spatial_attention_png.exists():
        src = spatial_attention_png

    if src is None:
        return None

    img = plt.imread(src)

    # If overlay exists, try to soften the heatmap influence and keep background-like preview.
    if img.ndim == 3 and img.shape[-1] >= 3:
        rgb = img[..., :3].astype(np.float32)
        rgb = _normalize01(rgb)
    else:
        gray = _normalize01(img.astype(np.float32))
        rgb = np.stack([gray, gray, gray], axis=-1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png, rgb)
    return out_png
