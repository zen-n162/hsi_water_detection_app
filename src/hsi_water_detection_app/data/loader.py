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
    if suffix in {".csv", ".txt", ".npy"}:
        return "wavelength_sidecar"
    return "unknown"


def _find_envi_binary_from_hdr(hdr_path: Path) -> Optional[Path]:
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


def _find_sidecar_hdr(path: Path) -> Optional[Path]:
    candidates = [
        path.with_suffix(".hdr"),
        Path(str(path) + ".hdr"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _extract_brace_block(text: str, key: str) -> Optional[str]:
    key_idx = text.lower().find(key.lower())
    if key_idx == -1:
        return None
    brace_start = text.find("{", key_idx)
    brace_end = text.find("}", brace_start)
    if brace_start == -1 or brace_end == -1:
        return None
    return text[brace_start + 1 : brace_end]


def _extract_scalar_value(text: str, key: str) -> Optional[str]:
    lines = text.splitlines()
    for line in lines:
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        if lhs.strip().lower() == key.lower():
            return rhs.strip().strip("{}").strip()
    return None


def _parse_float_list(block: str) -> Optional[np.ndarray]:
    values = []
    for item in block.replace("\n", " ").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def _parse_int_list(block: str) -> Optional[np.ndarray]:
    values = []
    for item in block.replace("\n", " ").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(float(item)))
        except ValueError:
            continue
    if not values:
        return None
    return np.asarray(values, dtype=np.int32)


def _convert_wavelength_units(
    wavelengths: np.ndarray,
    units: Optional[str],
    target_units: str = "nm",
) -> np.ndarray:
    """
    Normalize wavelength units.
    Supported input:
      - micrometers / um / microns
      - nanometers / nm
      - index / unknown -> unchanged
    """
    if units is None:
        return wavelengths

    u = units.strip().lower()

    if u in {"nanometers", "nanometer", "nm"}:
        current = "nm"
    elif u in {"micrometers", "micrometer", "um", "μm", "microns", "micron"}:
        current = "um"
    else:
        # unknown/index/wavenumber/... は今はそのまま返す
        return wavelengths

    if current == target_units:
        return wavelengths

    if current == "um" and target_units == "nm":
        return wavelengths * 1000.0

    if current == "nm" and target_units == "um":
        return wavelengths / 1000.0

    return wavelengths


def _parse_envi_header_text(header_text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    wave_block = _extract_brace_block(header_text, "wavelength")
    if wave_block is not None:
        wavelengths = _parse_float_list(wave_block)
        if wavelengths is not None:
            meta["wavelengths_raw"] = wavelengths

    fwhm_block = _extract_brace_block(header_text, "fwhm")
    if fwhm_block is not None:
        fwhm = _parse_float_list(fwhm_block)
        if fwhm is not None:
            meta["fwhm_raw"] = fwhm

    bbl_block = _extract_brace_block(header_text, "bbl")
    if bbl_block is not None:
        bbl = _parse_int_list(bbl_block)
        if bbl is not None:
            meta["bbl"] = bbl

    units = _extract_scalar_value(header_text, "wavelength units")
    if units is not None:
        meta["wavelength_units_raw"] = units

    if "wavelengths_raw" in meta:
        wavelengths_nm = _convert_wavelength_units(
            meta["wavelengths_raw"],
            meta.get("wavelength_units_raw"),
            target_units="nm",
        )
        meta["wavelengths"] = wavelengths_nm
        meta["wavelength_units"] = "nm"

    if "fwhm_raw" in meta:
        fwhm_nm = _convert_wavelength_units(
            meta["fwhm_raw"],
            meta.get("wavelength_units_raw"),
            target_units="nm",
        )
        meta["fwhm"] = fwhm_nm

    return meta


def _read_header_metadata(header_path: Path) -> Dict[str, Any]:
    text = header_path.read_text(encoding="utf-8", errors="ignore")
    meta = _parse_envi_header_text(text)
    meta["header_path"] = str(header_path)
    return meta


def parse_header_wavelengths(path: str) -> Optional[np.ndarray]:
    """
    Read wavelengths from:
      - ENVI .hdr
      - .csv / .txt / .npy sidecar
    Returns wavelengths in nm where possible.
    """
    p = Path(path)

    if not p.exists():
        print(f"[WARN] header file does not exist: {p}")
        return None

    suffix = p.suffix.lower()

    if suffix == ".hdr":
        meta = _read_header_metadata(p)
        wavelengths = meta.get("wavelengths")
        if wavelengths is None:
            print(f"[INFO] no wavelength field found in: {p}")
            return None
        print(f"[INFO] parsed {len(wavelengths)} wavelengths from header: {p}")
        return wavelengths

    if suffix == ".npy":
        arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        print(f"[INFO] loaded {len(arr)} wavelengths from npy: {p}")
        return arr

    if suffix in {".csv", ".txt"}:
        values = []
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            parts = [x.strip() for x in line.replace(",", " ").split()]
            for token in parts:
                try:
                    values.append(float(token))
                except ValueError:
                    continue
        if not values:
            print(f"[INFO] no numeric wavelength values found in: {p}")
            return None
        arr = np.asarray(values, dtype=np.float32)
        print(f"[INFO] loaded {len(arr)} wavelengths from text sidecar: {p}")
        return arr

    print(f"[INFO] unsupported header/wavelength sidecar format: {p}")
    return None


def _load_geotiff_or_envi(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError(
            "rasterio is not installed. Install it before reading GeoTIFF/ENVI files."
        )

    open_path = path

    if path.suffix.lower() == ".hdr":
        candidate = _find_envi_binary_from_hdr(path)
        if candidate is None:
            raise FileNotFoundError(
                f"ENVI .hdr was provided but no matching binary file was found for: {path}"
            )
        open_path = candidate
        print(f"[INFO] resolved ENVI binary from header: {open_path}")

    with rasterio.open(open_path) as src:
        cube = src.read()
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

        # header auto-discovery
        header_candidates = []
        if path.suffix.lower() == ".hdr":
            header_candidates.append(path)

        # wavelength/header auto-discovery
        sidecar = _find_wavelength_sidecar(open_path)
        if sidecar is not None:
            if sidecar.suffix.lower() == ".hdr":
                hdr_meta = _read_header_metadata(sidecar)
                meta.update(hdr_meta)
                if "wavelengths" in hdr_meta:
                    print(f"[INFO] attached wavelengths from sidecar header: {sidecar}")
            else:
                wavelengths = parse_header_wavelengths(str(sidecar))
                if wavelengths is not None:
                    meta["wavelengths"] = wavelengths
                    meta["wavelength_units"] = "nm"
                    meta["header_path"] = str(sidecar)
                    print(f"[INFO] attached wavelengths from sidecar file: {sidecar}")

        seen = set()
        for hdr in header_candidates:
            if hdr in seen:
                continue
            seen.add(hdr)
            if hdr.exists():
                hdr_meta = _read_header_metadata(hdr)
                meta.update(hdr_meta)
                if "wavelengths" in hdr_meta:
                    print(f"[INFO] attached wavelengths from sidecar header: {hdr}")
                break

        return cube.astype(np.float32, copy=False), meta


def _find_first_3d_dataset(h5obj: h5py.Group, prefix: str = "") -> Optional[str]:
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
        arr = dset[()]

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D dataset, got shape {arr.shape} for key {dataset_key}")

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

        wavelengths = None
        wavelength_units = None
        for attr_key in ("wavelength", "wavelengths", "band_wavelengths"):
            if attr_key in dset.attrs:
                wavelengths = np.asarray(dset.attrs[attr_key], dtype=np.float32).reshape(-1)
                break

        for unit_key in ("wavelength_units", "wavelength unit", "units"):
            if unit_key in dset.attrs:
                raw = dset.attrs[unit_key]
                wavelength_units = raw.decode() if isinstance(raw, bytes) else str(raw)
                break

        if wavelengths is not None:
            wavelengths = _convert_wavelength_units(wavelengths, wavelength_units, target_units="nm")
            meta["wavelengths"] = wavelengths
            meta["wavelength_units"] = "nm"

        return cube.astype(np.float32, copy=False), meta


def load_hsi_cube(
    path: str,
    *,
    dataset_key: Optional[str] = None,
    allow_dummy: bool = True,
    dummy_shape: Tuple[int, int, int] = (16, 64, 64),
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
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
        cube = src.read(window=window)  # (bands, rows, cols)
        meta = {
            "input_path": str(file_path),
            "status": "loaded_window",
            "format": src.driver,
            "width": cube.shape[2],
            "height": cube.shape[1],
            "count": cube.shape[0],
            "dtype": str(cube.dtype),
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.window_transform(window)),
            "roi": {
                "row_start": row_start,
                "row_stop": row_stop,
                "col_start": col_start,
                "col_stop": col_stop,
            },
        }
    return cube.astype(np.float32, copy=False), meta


def load_hsi_bounds(
    path: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an ROI from projected/georeferenced coordinates such as UTM.

    Parameters
    ----------
    path : str
        Path to GeoTIFF/BigTIFF/ENVI-readable raster.
    xmin, ymin, xmax, ymax : float
        Bounding box in dataset CRS coordinates.
        For UTM data, these are typically meters.

    Returns
    -------
    cube : np.ndarray
        Array of shape (bands, rows, cols)
    meta : dict
        Metadata for the cropped ROI, including updated transform.
    """
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is not installed.")

    from rasterio.windows import from_bounds

    file_path = Path(path)

    with rasterio.open(file_path) as src:
        if src.transform is None:
            raise ValueError(f"No affine transform available in dataset: {file_path}")

        # Build window from geospatial bounds
        window = from_bounds(
            left=xmin,
            bottom=ymin,
            right=xmax,
            top=ymax,
            transform=src.transform,
        )

        # Round offsets/shape to integer pixel boundaries
        window = window.round_offsets().round_lengths()

        cube = src.read(window=window)  # (bands, rows, cols)

        meta: Dict[str, Any] = {
            "input_path": str(file_path),
            "status": "loaded_bounds",
            "format": src.driver,
            "width": cube.shape[2],
            "height": cube.shape[1],
            "count": cube.shape[0],
            "dtype": str(cube.dtype),
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.window_transform(window)),
            "bounds": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            },
        }

    return cube.astype(np.float32, copy=False), meta


def _find_wavelength_sidecar(path: Path) -> Optional[Path]:
    stem = path.with_suffix("")
    candidates = [
        stem.with_suffix(".hdr"),
        Path(str(stem) + "_wavelengths.npy"),
        Path(str(stem) + "_wavelengths.csv"),
        Path(str(stem) + "_wavelengths.txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
