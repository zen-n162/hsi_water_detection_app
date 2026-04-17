from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.preview_service import build_grayscale_preview
from backend.app.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.post("/preview/grayscale")
async def preview_grayscale(
    hsi_file: UploadFile | None = File(default=None),
    wavelength_file: UploadFile | None = File(default=None),
    input_path: str | None = Form(default=None),
    sensor: str = Form("auto"),
    preview_band: int | None = Form(default=None),
    preview_band_index: int | None = Form(default=None),
    preview_wavelength: float | None = Form(default=None),
    row_start: int | None = Form(default=None),
    row_stop: int | None = Form(default=None),
    col_start: int | None = Form(default=None),
    col_stop: int | None = Form(default=None),
    xmin: float | None = Form(default=None),
    ymin: float | None = Form(default=None),
    xmax: float | None = Form(default=None),
    ymax: float | None = Form(default=None),
):
    """
    Preview grayscale endpoint.

    Accepted input styles:
      1) uploaded HSI file (hsi_file)
      2) existing server-side path (input_path)

    Frontend compatibility:
      - preview_band_index or preview_band
      - returns top-level grayscale_preview_url
    """

    effective_preview_band = preview_band_index if preview_band_index is not None else preview_band

    # case 1: use existing server-side path
    if input_path:
        if not settings.allow_server_file_paths:
            raise PermissionError("input_path is disabled in public mode. Upload the HSI file instead.")
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise HTTPException(status_code=400, detail=f"input_path not found: {input_path}")

        wavelength_path = None
        if wavelength_file is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                wavelength_path = tmpdir_path / wavelength_file.filename
                with wavelength_path.open("wb") as f:
                    shutil.copyfileobj(wavelength_file.file, f)

                result = build_grayscale_preview(
                    input_path=str(input_path_obj),
                    header_path=str(wavelength_path),
                    sensor=sensor,
                    preview_band=effective_preview_band,
                    preview_wavelength=preview_wavelength,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
                return result

        result = build_grayscale_preview(
            input_path=str(input_path_obj),
            header_path=None,
            sensor=sensor,
            preview_band=effective_preview_band,
            preview_wavelength=preview_wavelength,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        return result

    # case 2: use uploaded file
    if hsi_file is None:
        raise HTTPException(
            status_code=400,
            detail="Either hsi_file or input_path must be provided.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        hsi_path = tmpdir_path / hsi_file.filename
        with hsi_path.open("wb") as f:
            shutil.copyfileobj(hsi_file.file, f)

        wavelength_path = None
        if wavelength_file is not None:
            wavelength_path = tmpdir_path / wavelength_file.filename
            with wavelength_path.open("wb") as f:
                shutil.copyfileobj(wavelength_file.file, f)

        result = build_grayscale_preview(
            input_path=str(hsi_path),
            header_path=str(wavelength_path) if wavelength_path else None,
            sensor=sensor,
            preview_band=effective_preview_band,
            preview_wavelength=preview_wavelength,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        return result
