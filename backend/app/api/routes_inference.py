from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.app.services.deploy_config_service import make_deploy_config_response
from backend.app.services.inference_service import run_inference_pipeline

router = APIRouter()


@router.get("/inference/deploy-config")
def get_inference_deploy_config(
    deploy_config_path: str | None = Query(default=None),
):
    return make_deploy_config_response(deploy_config_path=deploy_config_path)


@router.post("/inference/run")
async def run_inference(
    hsi_file: UploadFile | None = File(default=None),
    wavelength_file: UploadFile | None = File(default=None),

    input_path: str | None = Form(default=None),
    sensor: str = Form("auto"),
    device: str = Form("cpu"),
    deploy_config_path: str | None = Form(default=None),

    # legacy / optional
    model_checkpoint: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    temperature_json: str | None = Form(default=None),
    decision_threshold: float | None = Form(default=None),
    threshold: float | None = Form(default=None),  # frontend 互換
    manifest_path: str | None = Form(default=None),
    patch_dataset_path: str | None = Form(default=None),
    split_policy: str | None = Form(default=None),

    # HyperSIGMA dual checkpoint
    spat_checkpoint: str | None = Form(default=None),
    spec_checkpoint: str | None = Form(default=None),

    model_type: str | None = Form(default=None),
    patch_size: int | None = Form(default=None),
    stride: int | None = Form(default=None),

    preview_band_index: int | None = Form(default=None),  # 受けるだけで無害
    preview_wavelength: float | None = Form(default=None),  # 受けるだけで無害

    row_start: int | None = Form(default=None),
    row_stop: int | None = Form(default=None),
    col_start: int | None = Form(default=None),
    col_stop: int | None = Form(default=None),
    xmin: float | None = Form(default=None),
    ymin: float | None = Form(default=None),
    xmax: float | None = Form(default=None),
    ymax: float | None = Form(default=None),
):
    effective_threshold = decision_threshold if decision_threshold is not None else threshold

    # 1) server-side path input
    if input_path:
        result = run_inference_pipeline(
            input_path=input_path,
            header_path=None,
            sensor=sensor,
            device=device,
            deploy_config_path=deploy_config_path,
            model_checkpoint=model_checkpoint,
            spat_checkpoint=spat_checkpoint,
            spec_checkpoint=spec_checkpoint,
            temperature=temperature,
            temperature_json=temperature_json,
            decision_threshold=effective_threshold,
            manifest_path=manifest_path,
            patch_dataset_path=patch_dataset_path,
            split_policy=split_policy,
            model_type=model_type,
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

    # 2) uploaded file input
    if hsi_file is None:
        raise HTTPException(status_code=400, detail="Either hsi_file or input_path must be provided.")

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
            deploy_config_path=deploy_config_path,
            model_checkpoint=model_checkpoint,
            spat_checkpoint=spat_checkpoint,
            spec_checkpoint=spec_checkpoint,
            temperature=temperature,
            temperature_json=temperature_json,
            decision_threshold=effective_threshold,
            manifest_path=manifest_path,
            patch_dataset_path=patch_dataset_path,
            split_policy=split_policy,
            model_type=model_type,
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
