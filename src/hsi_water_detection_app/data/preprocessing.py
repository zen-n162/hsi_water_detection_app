from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def remove_bad_bands(
    cube: Optional[np.ndarray], bad_band_indices: Iterable[int]
) -> Optional[np.ndarray]:
    """
    Remove bad bands from cube shaped as (bands, height, width).
    """
    if cube is None:
        print("[INFO] remove_bad_bands skipped because cube is None")
        return None

    bad_band_indices = set(int(i) for i in bad_band_indices)
    num_bands = cube.shape[0]
    keep_indices = [i for i in range(num_bands) if i not in bad_band_indices]

    if not keep_indices:
        print("[WARN] all bands would be removed; returning original cube")
        return cube

    out = cube[keep_indices, :, :]
    print(f"[INFO] remove_bad_bands: {num_bands} -> {out.shape[0]} bands")
    return out


def normalize_cube(cube: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Per-band min-max normalization for cube shaped as (bands, height, width).
    """
    if cube is None:
        print("[INFO] normalize_cube skipped because cube is None")
        return None

    cube = cube.astype(np.float32, copy=False)
    out = np.empty_like(cube, dtype=np.float32)

    for band_idx in range(cube.shape[0]):
        band = cube[band_idx]
        band_min = np.nanmin(band)
        band_max = np.nanmax(band)

        if np.isclose(band_max, band_min):
            out[band_idx] = 0.0
        else:
            out[band_idx] = (band - band_min) / (band_max - band_min)

    print("[INFO] normalize_cube completed")
    return out
