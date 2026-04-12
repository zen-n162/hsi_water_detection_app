from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import (
    load_hsi_bounds,
    load_hsi_cube,
    load_hsi_window,
    parse_header_wavelengths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_BASE_URL = "http://127.0.0.1:8000"


def to_public_url(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    return f"{PUBLIC_BASE_URL}/{rel.as_posix()}"


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


def _infer_sensor(sensor: str, input_path: str) -> str:
    if sensor != "auto":
        return sensor
    s = input_path.lower()
    if "eo1" in s or "hyperion" in s:
        return "hyperion"
    if "hisui" in s:
        return "hisui"
    return "generic"


def _load_cube_and_meta(
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
):
    if all(v is not None for v in [xmin, ymin, xmax, ymax]):
        cube, meta = load_hsi_bounds(
            input_path,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
    elif all(v is not None for v in [row_start, row_stop, col_start, col_stop]):
        cube, meta = load_hsi_window(
            input_path,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
        )
    else:
        cube, meta = load_hsi_cube(
            input_path,
            allow_dummy=False,
            sensor=sensor if sensor != "generic" else None,
        )
    return cube, meta


def _apply_bad_band_rule(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray],
    sensor: str,
):
    if cube is None:
        return cube, wavelengths

    if sensor != "hyperion":
        return cube, wavelengths

    before = cube.shape[0]
    bad_set = set(HYPERION_BAD_BANDS_0BASED)
    keep_indices = [i for i in range(before) if i not in bad_set]
    cube = cube[keep_indices]

    if wavelengths is not None and len(wavelengths) == before:
        wavelengths = wavelengths[keep_indices]

    return cube, wavelengths


def _resolve_preview_band(
    *,
    preview_band: int | None,
    preview_wavelength: float | None,
    wavelengths: Optional[np.ndarray],
    num_bands: int,
) -> tuple[int, Optional[float]]:
    if preview_band is not None:
        band = max(0, min(int(preview_band), num_bands - 1))
        wl = float(wavelengths[band]) if wavelengths is not None and band < len(wavelengths) else None
        return band, wl

    if preview_wavelength is not None and wavelengths is not None and len(wavelengths) > 0:
        idx = int(np.argmin(np.abs(wavelengths - float(preview_wavelength))))
        return idx, float(wavelengths[idx])

    band = min(10, num_bands - 1)
    wl = float(wavelengths[band]) if wavelengths is not None and band < len(wavelengths) else None
    return band, wl


def build_grayscale_preview(
    *,
    input_path: str,
    header_path: str | None,
    sensor: str,
    preview_band: int | None,
    preview_wavelength: float | None,
    row_start: int | None,
    row_stop: int | None,
    col_start: int | None,
    col_stop: int | None,
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
):
    sensor = _infer_sensor(sensor, input_path)

    cube, meta = _load_cube_and_meta(
        input_path=input_path,
        sensor=sensor,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )

    wavelengths = None
    if header_path:
        wavelengths = parse_header_wavelengths(header_path)
    if wavelengths is None:
        wavelengths = meta.get("wavelengths")

    cube, wavelengths = _apply_bad_band_rule(cube, wavelengths, sensor)

    if cube is None:
        raise RuntimeError("Failed to load HSI cube for grayscale preview")

    band_idx, band_wavelength = _resolve_preview_band(
        preview_band=preview_band,
        preview_wavelength=preview_wavelength,
        wavelengths=wavelengths,
        num_bands=cube.shape[0],
    )

    band_img = _stretch_band(cube[band_idx])

    outdir = PROJECT_ROOT / "outputs" / "preview_ui" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / "grayscale_preview.png"
    plt.imsave(png_path, band_img, cmap="gray", vmin=0.0, vmax=1.0)

    result = {
        "ok": True,
        "sensor": sensor,
        "preview_band_index": int(band_idx),
        "preview_wavelength_nm": band_wavelength,
        "image_height": int(band_img.shape[0]),
        "image_width": int(band_img.shape[1]),
        "output_dir": str(outdir),
        "files": {
            "grayscale_preview_png": str(png_path),
        },
        "urls": {
            "grayscale_preview_png": to_public_url(png_path),
        },
        "meta": {
            "crs": meta.get("crs"),
            "transform": meta.get("transform"),
            "roi": meta.get("roi"),
        },
    }

    (outdir / "preview_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result
