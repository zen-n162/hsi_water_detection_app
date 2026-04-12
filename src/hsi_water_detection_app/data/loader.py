from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import h5py


def _optional_import_rasterio():
    try:
        import rasterio  # type: ignore
        return rasterio
    except ImportError:
        return None


def _guess_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    if suffix in {".hdr", ".img", ".dat", ".bin"}:
        return "envi_or_raw"
    return "unknown"


def _find_envi_binary_from_hdr(hdr_path: Path) -> Optional[Path]:
    """
    If user passes .hdr directly, try to find the matching raw binary file.
    Common candidates: same stem with .img/.dat/.bin or no suffix variant.
    """
    candidates = [
        hdr_path.with_suffix(".img"),
        hdr_path.with_suffix(".dat"),
        hdr_path.with_suffix(".bin"),
        hdr_path.with_suffix(""),
    ]
    for c in candidates:
        if c.exists() and c != hdr_path:
            return c
    return None


def _parse_envi_header_text(header_text: str) -> Dict[str, Any]:
    """
    Parse a minimal subset of ENVI header fields.
    """
    meta: Dict[str, Any] = {}

    lines = header_text.splitlines()
    joined = "\n".join(lines)

    def _extract_brace_block(key: str) -> Optional[str]:
        key_idx = joined.lower().find(key.lower())
        if key_idx == -1:
            return None
        brace_start = joined.find("{", key_idx)
        brace_end = joined.find("}", brace_start)
        if brace_start == -1 or brace_end == -1:
            return None
        return joined[brace_start + 1 : brace_end]

    # wavelength
    wave_block = _extract_brace_block("wavelength")
    if wave_block is not None:
        values = []
        for item in wave_block.replace("\n", " ").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                continue
        if values:
            meta["wavelengths"] = np.asarray(values, dtype=np.float32)

    # fwhm
    fwhm_block = _extract_brace_block("fwhm")
    if fwhm_block is not None:
        values = []
        for item in fwhm_block.replace("\n", " ").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                continue
        if values:
            meta["fwhm"] = np.asarray(values, dtype=np.float32)

    # bad band list
    bbl_block = _extract_brace_block("bbl")
    if bbl_block is not None:
        values = []
        for item in bbl_block.replace("\n", " ").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(float(item)))
            except ValueError:
                continue
        if values:
            meta["bbl"] = np.asarray(values, dtype=np.int32)

    return meta


def parse_header_wavelengths(path: str) -> Optional[np.ndarray]:
    """
    Parse ENVI-style wavelength metadata from .hdr.
    """
    header_path = Path(path)

    if not header_path.exists():
        print(f"[WARN] header file does not exist: {header_path}")
        return None

    text = header_path.read_text(encoding="utf-8", errors="ignore")
    parsed = _parse_envi_header_text(text)
    wavelengths = parsed.get("wavelengths")

    if wavelengths is None:
        print(f"[INFO] no wavelength field found in: {header_path}")
        return None

    print(f"[INFO] parsed {len(wavelengths)} wavelengths from: {header_path}")
    return wavelengths


def _load_geotiff_or_envi(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError(
            "rasterio is not installed. Install it before reading GeoTIFF/ENVI files."
        )

    open_path = path

    # If user passed .hdr directly, try to resolve matching binary.
    if path.suffix.lower() == ".hdr":
        candidate = _find_envi_binary_from_hdr(path)
        if candidate is None:
            raise FileNotFoundError(
                f"ENVI .hdr was provided but no matching binary file was found for: {path}"
            )
        open_path = candidate
        print(f"[INFO] resolved ENVI binary from header: {open_path}")

    with rasterio.open(open_path) as src:
        cube = src.read()  # (bands, height, width)
        meta: Dict[str, Any] = {
            "input_path": str(path),
            "opened_path": str(open_path),
            "status": "loaded",
            "format": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]) if src.count > 0 else None,
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform) if src.transform else None,
        }

        # Try sidecar/header wavelengths for ENVI-like data
        hdr_candidates = []
        if path.suffix.lower() == ".hdr":
            hdr_candidates.append(path)
        hdr_candidates.append(open_path.with_suffix(".hdr"))

        wavelengths = None
        for hdr in hdr_candidates:
            if hdr.exists():
                wavelengths = parse_header_wavelengths(str(hdr))
                if wavelengths is not None:
                    break

        if wavelengths is not None:
            meta["wavelengths"] = wavelengths

        return cube, meta


def _find_first_3d_dataset(h5obj: h5py.Group, prefix: str = "") -> Optional[str]:
    """
    Find first 3D dataset path in HDF5 recursively.
    """
    for key in h5obj.keys():
        obj = h5obj[key]
        current_path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            return current_path
        if isinstance(obj, h5py.Group):
            found = _find_first_3d_dataset(obj, current_path)
            if found is not None:
                return found
    return None


def _load_hdf5(path: Path, dataset_key: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        if dataset_key is None:
            dataset_key = _find_first_3d_dataset(f)
            if dataset_key is None:
                raise ValueError(f"No 3D dataset found in HDF5 file: {path}")

        dset = f[dataset_key]
        arr = dset[()]  # h5py datasets support NumPy-like slicing

        # normalize axis order to (bands, height, width)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D dataset, got shape {arr.shape} for key {dataset_key}")

        # Heuristic:
        # if last axis is smallest, assume (height, width, bands) and transpose
        if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
            cube = np.transpose(arr, (2, 0, 1))
        else:
            cube = arr

        meta: Dict[str, Any] = {
            "input_path": str(path),
            "status": "loaded",
            "format": "HDF5",
            "dataset_key": dataset_key,
            "shape_original": tuple(arr.shape),
            "shape_cube": tuple(cube.shape),
            "dtype": str(arr.dtype),
        }

        # Try to get wavelengths from attrs
        wavelengths = None
        for attr_key in ("wavelength", "wavelengths", "band_wavelengths"):
            if attr_key in dset.attrs:
                raw = dset.attrs[attr_key]
                wavelengths = np.asarray(raw, dtype=np.float32).reshape(-1)
                break

        if wavelengths is not None:
            meta["wavelengths"] = wavelengths

        return cube.astype(np.float32, copy=False), meta


def load_hsi_cube(
    path: str,
    *,
    dataset_key: Optional[str] = None,
    allow_dummy: bool = True,
    dummy_shape: Tuple[int, int, int] = (16, 64, 64),
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Load an HSI cube from GeoTIFF / ENVI / HDF5.

    Returns:
        cube: ndarray shaped (bands, height, width) or None
        meta: metadata dict

    If file does not exist and allow_dummy=True, returns a zero dummy cube.
    """
    file_path = Path(path)

    if not file_path.exists():
        print(f"[WARN] input file does not exist yet: {file_path}")
        if allow_dummy:
            bands, height, width = dummy_shape
            cube = np.zeros((bands, height, width), dtype=np.float32)
            return cube, {
                "input_path": str(file_path),
                "status": "dummy_generated",
                "format": "dummy",
                "shape_cube": tuple(cube.shape),
            }
        return None, {
            "input_path": str(file_path),
            "status": "missing",
            "format": "unknown",
        }

    fmt = _guess_format(file_path)

    if fmt in {"geotiff", "envi_or_raw"}:
        cube, meta = _load_geotiff_or_envi(file_path)
    elif fmt == "hdf5":
        cube, meta = _load_hdf5(file_path, dataset_key=dataset_key)
    else:
        raise ValueError(f"Unsupported input format for file: {file_path}")

    print(f"[INFO] loaded cube with shape {cube.shape} from {file_path}")
    return cube.astype(np.float32, copy=False), meta

def load_hsi_window(
    path: str,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
):
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is not installed.")

    from pathlib import Path
    from rasterio.windows import Window

    file_path = Path(path)
    with rasterio.open(file_path) as src:
        window = Window(
            col_off=col_start,
            row_off=row_start,
            width=col_stop - col_start,
            height=row_stop - row_start,
        )
        cube = src.read(window=window)
        meta = {
            "input_path": str(file_path),
            "status": "loaded_window",
            "format": src.driver,
            "width": cube.shape[2],
            "height": cube.shape[1],
            "count": cube.shape[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.window_transform(window)) if src.transform else None,
        }
    return cube.astype(np.float32, copy=False), meta
