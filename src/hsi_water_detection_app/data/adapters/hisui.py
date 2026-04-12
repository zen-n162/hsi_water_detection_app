from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hsi_water_detection_app.data.adapters.base import BaseHSIAdapter
from hsi_water_detection_app.data.metadata import HSICube, HSIMetadata
from hsi_water_detection_app.data.roi import ROI
from hsi_water_detection_app.data.wavelength_io import load_wavelength_sidecar


class HISUIAdapter(BaseHSIAdapter):
    sensor_name = "hisui"

    def __init__(self, raster_loader, bounds_loader, pixel_loader):
        self._raster_loader = raster_loader
        self._bounds_loader = bounds_loader
        self._pixel_loader = pixel_loader

    def can_handle(self, path: Path, meta_hint: dict[str, Any]) -> bool:
        count = meta_hint.get("count")
        name = path.name.lower()
        return count == 185 or "hisui" in name

    def load(self, path: Path, roi: Optional[ROI] = None) -> HSICube:
        if roi is None:
            cube, raw = self._raster_loader(path)
        elif roi.mode == "bounds":
            cube, raw = self._bounds_loader(path, roi.xmin, roi.ymin, roi.xmax, roi.ymax)
        elif roi.mode == "pixel":
            cube, raw = self._pixel_loader(path, roi.row_start, roi.row_stop, roi.col_start, roi.col_stop)
        else:
            raise ValueError(f"Unsupported ROI mode: {roi.mode}")

        candidates = [
            path.with_name(path.stem + "_wavelengths.csv"),
            path.with_name(path.stem + "_wavelengths.npy"),
            path.with_suffix(".hdr"),
        ]
        wavelengths = None
        header_path = None
        for c in candidates:
            if c.exists():
                wavelengths = load_wavelength_sidecar(c)
                header_path = str(c)
                if wavelengths is not None:
                    break

        meta = HSIMetadata(
            sensor="hisui",
            crs=raw.get("crs"),
            transform=raw.get("transform"),
            width=raw["width"],
            height=raw["height"],
            bands=raw["count"],
            dtype=raw["dtype"],
            wavelengths_nm=wavelengths,
            source_path=str(path),
            header_path=header_path,
            roi=raw.get("roi"),
        )
        return HSICube(data=cube, meta=meta)
