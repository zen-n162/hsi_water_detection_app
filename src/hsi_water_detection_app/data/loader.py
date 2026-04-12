from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from hsi_water_detection_app.data.adapters.hisui import HISUIAdapter
from hsi_water_detection_app.data.adapters.hyperion import HyperionAdapter
from hsi_water_detection_app.data.metadata import HSICube, HSIMetadata
from hsi_water_detection_app.data.roi import ROI
from hsi_water_detection_app.data.wavelength_io import load_wavelength_sidecar


def _optional_import_rasterio():
    try:
        import rasterio  # type: ignore
        return rasterio
    except ImportError:
        return None


def _inspect_raster_quick(path: Path) -> dict[str, Any]:
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        return {"suffix": path.suffix.lower()}
    try:
        with rasterio.open(path) as src:
            return {
                "suffix": path.suffix.lower(),
                "count": src.count,
                "crs": str(src.crs) if src.crs else None,
                "width": src.width,
                "height": src.height,
            }
    except Exception:
        return {"suffix": path.suffix.lower()}


def _load_raster_full(path: Path):
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is not installed.")
    with rasterio.open(path) as src:
        cube = src.read()
        meta = {
            "input_path": str(path),
            "status": "loaded",
            "format": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]) if src.count > 0 else None,
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform) if src.transform else None,
        }
    return cube.astype(np.float32, copy=False), meta


def _load_raster_pixel_roi(path: Path, row_start: int, row_stop: int, col_start: int, col_stop: int):
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is not installed.")
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        window = Window(col_off=col_start, row_off=row_start, width=col_stop-col_start, height=row_stop-row_start)
        cube = src.read(window=window)
        meta = {
            "input_path": str(path),
            "status": "loaded_window",
            "format": src.driver,
            "width": cube.shape[2],
            "height": cube.shape[1],
            "count": cube.shape[0],
            "dtype": str(cube.dtype),
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.window_transform(window)),
            "roi": {
                "mode": "pixel",
                "row_start": row_start,
                "row_stop": row_stop,
                "col_start": col_start,
                "col_stop": col_stop,
            },
        }
    return cube.astype(np.float32, copy=False), meta


def _load_raster_bounds_roi(path: Path, xmin: float, ymin: float, xmax: float, ymax: float):
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is not installed.")
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        window = window.round_offsets().round_lengths()
        cube = src.read(window=window)
        meta = {
            "input_path": str(path),
            "status": "loaded_bounds",
            "format": src.driver,
            "width": cube.shape[2],
            "height": cube.shape[1],
            "count": cube.shape[0],
            "dtype": str(cube.dtype),
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.window_transform(window)),
            "roi": {
                "mode": "bounds",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            },
        }
    return cube.astype(np.float32, copy=False), meta


def _load_hdf5(path: Path, dataset_key: Optional[str] = None):
    with h5py.File(path, "r") as f:
        if dataset_key is None:
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3:
                    dataset_key = k
                    break
        if dataset_key is None:
            raise ValueError(f"No 3D dataset found in HDF5 file: {path}")

        dset = f[dataset_key]
        arr = dset[()]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D dataset, got {arr.shape}")

        if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
            cube = np.transpose(arr, (2, 0, 1))
        else:
            cube = arr

        wavelengths = None
        for attr_key in ("wavelength", "wavelengths", "band_wavelengths"):
            if attr_key in dset.attrs:
                wavelengths = np.asarray(dset.attrs[attr_key], dtype=np.float32).reshape(-1)
                break

        meta = HSIMetadata(
            sensor="hdf5_generic",
            crs=None,
            transform=None,
            width=cube.shape[2],
            height=cube.shape[1],
            bands=cube.shape[0],
            dtype=str(cube.dtype),
            wavelengths_nm=wavelengths,
            source_path=str(path),
        )
        return HSICube(data=cube.astype(np.float32, copy=False), meta=meta)


def parse_header_wavelengths(path: str):
    return load_wavelength_sidecar(path)


def load_hsi_cube(
    path: str,
    *,
    dataset_key: Optional[str] = None,
    allow_dummy: bool = True,
    dummy_shape: Tuple[int, int, int] = (16, 64, 64),
    sensor: Optional[str] = None,
    roi: Optional[ROI] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    p = Path(path)

    if not p.exists():
        print(f"[WARN] input file does not exist yet: {p}")
        if allow_dummy:
            bands, height, width = dummy_shape
            cube = np.zeros((bands, height, width), dtype=np.float32)
            return cube, {
                "input_path": str(p),
                "status": "dummy_generated",
                "format": "dummy",
                "shape_cube": tuple(cube.shape),
            }
        return None, {"input_path": str(p), "status": "missing", "format": "unknown"}

    if p.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        obj = _load_hdf5(p, dataset_key=dataset_key)
        return obj.data, obj.meta.__dict__

    hint = _inspect_raster_quick(p)

    adapters = [
        HyperionAdapter(_load_raster_full, _load_raster_bounds_roi, _load_raster_pixel_roi),
        HISUIAdapter(_load_raster_full, _load_raster_bounds_roi, _load_raster_pixel_roi),
    ]

    if sensor is not None:
        adapters = sorted(adapters, key=lambda a: a.sensor_name != sensor)

    for adapter in adapters:
        if adapter.can_handle(p, hint):
            obj = adapter.load(p, roi=roi)
            print(f"[INFO] loaded cube with shape {obj.data.shape} from {p} using adapter={adapter.sensor_name}")
            return obj.data, obj.meta.__dict__

    cube, meta = _load_raster_full(p)
    print(f"[INFO] loaded cube with shape {cube.shape} from {p} using adapter=generic")
    return cube, meta


def load_hsi_window(path: str, row_start: int, row_stop: int, col_start: int, col_stop: int):
    roi = ROI(mode="pixel", row_start=row_start, row_stop=row_stop, col_start=col_start, col_stop=col_stop)
    return load_hsi_cube(path, allow_dummy=False, roi=roi)


def load_hsi_bounds(path: str, xmin: float, ymin: float, xmax: float, ymax: float):
    roi = ROI(mode="bounds", xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return load_hsi_cube(path, allow_dummy=False, roi=roi)
