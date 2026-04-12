from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import load_hsi_bounds, load_hsi_cube, load_hsi_window
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amin, amax):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _stretch_band(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if np.isclose(lo, hi):
        return _normalize01(arr)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def _safe_rgb_bands(num_bands: int, preferred=(3, 9, 17)) -> tuple[int, int, int]:
    if num_bands <= 0:
        return (0, 0, 0)
    if all(b < num_bands for b in preferred):
        return preferred

    # fallback: spread across spectrum
    if num_bands == 1:
        return (0, 0, 0)
    if num_bands == 2:
        return (0, 1, 1)

    return (0, num_bands // 2, num_bands - 1)


def load_preview_cube(
    *,
    input_path: str,
    sensor: str,
    row_start: int | None,
    row_stop: int | None,
    col_start: int | None,
    col_stop: int | None,
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
) -> Optional[np.ndarray]:
    if all(v is not None for v in [xmin, ymin, xmax, ymax]):
        cube, _ = load_hsi_bounds(
            input_path,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
    elif all(v is not None for v in [row_start, row_stop, col_start, col_stop]):
        cube, _ = load_hsi_window(
            input_path,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
        )
    else:
        cube, _ = load_hsi_cube(
            input_path,
            allow_dummy=False,
            sensor=sensor if sensor != "auto" else None,
        )

    if cube is None:
        return None

    if sensor == "hyperion":
        cube = remove_bad_bands(cube, bad_band_indices=HYPERION_BAD_BANDS_0BASED)

    cube = normalize_cube(cube)
    return cube


def make_pseudocolor_png_from_cube(
    cube: np.ndarray,
    out_png: str | Path,
    rgb_bands: tuple[int, int, int] = (3, 9, 17),
) -> Optional[Path]:
    if cube is None or cube.ndim != 3:
        return None

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bands, _, _ = cube.shape
    r_idx, g_idx, b_idx = _safe_rgb_bands(bands, rgb_bands)

    rgb = np.stack(
        [
            _stretch_band(cube[r_idx]),
            _stretch_band(cube[g_idx]),
            _stretch_band(cube[b_idx]),
        ],
        axis=-1,
    ).astype(np.float32)

    plt.imsave(out_png, rgb)
    return out_png


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


def make_probability_overlay_png(
    prob_map_npy: str | Path,
    pseudocolor_png: str | Path,
    out_png: str | Path,
    alpha: float = 0.45,
) -> Optional[Path]:
    prob_map_npy = Path(prob_map_npy)
    pseudocolor_png = Path(pseudocolor_png)
    out_png = Path(out_png)

    if not prob_map_npy.exists() or not pseudocolor_png.exists():
        return None

    prob = np.load(prob_map_npy).astype(np.float32)
    prob = _normalize01(prob)

    bg = plt.imread(pseudocolor_png).astype(np.float32)
    if bg.ndim == 2:
        bg = np.stack([bg, bg, bg], axis=-1)
    if bg.shape[-1] == 4:
        bg = bg[..., :3]

    heat = cm.viridis(prob)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * bg + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png, overlay)
    return out_png
