from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HSIMetadata:
    sensor: str
    crs: Optional[str]
    transform: Optional[tuple]
    width: int
    height: int
    bands: int
    dtype: str
    wavelengths_nm: Optional[np.ndarray] = None
    fwhm_nm: Optional[np.ndarray] = None
    bad_band_mask: Optional[np.ndarray] = None
    band_names: Optional[list[str]] = None
    source_path: Optional[str] = None
    header_path: Optional[str] = None
    roi: Optional[dict] = None


@dataclass
class HSICube:
    data: np.ndarray  # (bands, rows, cols)
    meta: HSIMetadata
