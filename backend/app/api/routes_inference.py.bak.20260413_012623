from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.services.inference_service import run_inference_pipeline

router = APIRouter()

@router.post("/inference/run")
async def run_inference(
    hsi_file: UploadFile = File(...),
    wavelength_file: UploadFile | None = File(default=None),
    sensor: str = Form("auto"),
    device: str = Form("cpu"),
    model_checkpoint: str = Form("dummy.pth"),
    patch_size: int = Form(64),
    stride: int = Form(32),
    row_start: int | None = Form(default=None),
    row_stop: int | None = Form(default=None),
    col_start: int | None = Form(default=None),
    col_stop: int | None = Form(default=None),
    xmin: float | None = Form(default=None),
    ymin: float | None = Form(default=None),
    xmax: float | None = Form(default=None),
    ymax: float | None = Form(default=None),
):
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

        result = run_inference_pipeline(
            input_path=str(hsi_path),
            header_path=str(wavelength_path) if wavelength_path else None,
            sensor=sensor,
            device=device,
            model_checkpoint=model_checkpoint,
            patch_size=patch_size,
            stride=stride,
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
