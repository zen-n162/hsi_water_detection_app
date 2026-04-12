from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from hsi_water_detection_app.data.adapters.base import BaseHSIAdapter
from hsi_water_detection_app.data.metadata import HSICube, HSIMetadata
from hsi_water_detection_app.data.roi import ROI
from hsi_water_detection_app.data.wavelength_io import load_wavelength_sidecar


HYPERION_BAD_BANDS_1BASED = (
    list(range(1, 8)) +
    list(range(58, 77)) +
    list(range(225, 243))
)
HYPERION_BAD_BANDS_0BASED = np.asarray([b - 1 for b in HYPERION_BAD_BANDS_1BASED], dtype=np.int32)


class HyperionAdapter(BaseHSIAdapter):
    sensor_name = "hyperion"

    def __init__(self, raster_loader, bounds_loader, pixel_loader):
        self._raster_loader = raster_loader
        self._bounds_loader = bounds_loader
        self._pixel_loader = pixel_loader

    def can_handle(self, path: Path, meta_hint: dict[str, Any]) -> bool:
        count = meta_hint.get("count")
        name = path.name.lower()
        return count == 242 or "eo1" in name or "hyperion" in name

    def load(self, path: Path, roi: Optional[ROI] = None) -> HSICube:
        if roi is None:
            cube, raw = self._raster_loader(path)
        elif roi.mode == "bounds":
            cube, raw = self._bounds_loader(path, roi.xmin, roi.ymin, roi.xmax, roi.ymax)
        elif roi.mode == "pixel":
            cube, raw = self._pixel_loader(path, roi.row_start, roi.row_stop, roi.col_start, roi.col_stop)
        else:
            raise ValueError(f"Unsupported ROI mode: {roi.mode}")

        sidecar = path.with_name(path.stem + "_wavelengths.csv")
        wavelengths = load_wavelength_sidecar(sidecar)

        meta = HSIMetadata(
            sensor="hyperion",
            crs=raw.get("crs"),
            transform=raw.get("transform"),
            width=raw["width"],
            height=raw["height"],
            bands=raw["count"],
            dtype=raw["dtype"],
            wavelengths_nm=wavelengths,
            bad_band_mask=np.isin(np.arange(raw["count"]), HYPERION_BAD_BANDS_0BASED),
            source_path=str(path),
            header_path=str(sidecar) if sidecar.exists() else None,
            roi=raw.get("roi"),
        )
        return HSICube(data=cube, meta=meta)
