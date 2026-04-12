from __future__ import annotations

import numpy as np


def mean_spectrum(cube: np.ndarray) -> np.ndarray:
    return cube.mean(axis=(1, 2))


def band_depth(center: float, left: float, right: float, wavelengths: np.ndarray, spectrum: np.ndarray) -> float:
    def nearest_idx(v: float) -> int:
        return int(np.argmin(np.abs(wavelengths - v)))

    ic = nearest_idx(center)
    il = nearest_idx(left)
    ir = nearest_idx(right)

    continuum = np.interp(wavelengths[ic], [wavelengths[il], wavelengths[ir]], [spectrum[il], spectrum[ir]])
    if continuum == 0:
        return 0.0
    return float(1.0 - spectrum[ic] / continuum)
