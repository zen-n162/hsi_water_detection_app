from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.app.services.deploy_config_service import make_deploy_config_response
from backend.app.services.inference_service import run_inference_pipeline
from backend.app.settings import get_settings

router = APIRouter()
settings = get_settings()


def _ensure_path_override_allowed(value: str | None, *, field_name: str):
    if value and not settings.allow_server_file_paths:
        raise PermissionError(f"{field_name} is disabled in external web mode.")


def _ensure_deploy_config_override_allowed(value: str | None):
    if value and not settings.allow_deploy_config_override:
        raise PermissionError("deploy_config_path override is disabled in external web mode.")


def _ensure_runtime_override_disabled(value, *, field_name: str):
    if value in {None, ""}:
        return
    if not settings.allow_deploy_config_override:
        raise PermissionError(f"{field_name} override is disabled in external web mode.")


@router.get("/inference/deploy-config")
def get_inference_deploy_config(
    deploy_config_path: str | None = Query(default=None),
):
    _ensure_deploy_config_override_allowed(deploy_config_path)
    return make_deploy_config_response(deploy_config_path=deploy_config_path)


@router.post("/inference/run")
async def run_inference(
    hsi_file: UploadFile | None = File(default=None),
    wavelength_file: UploadFile | None = File(default=None),
    label_file: UploadFile | None = File(default=None),
    mask_file: UploadFile | None = File(default=None),

    input_path: str | None = Form(default=None),
    sensor: str = Form("auto"),
    device: str | None = Form(default=None),
    deploy_config_path: str | None = Form(default=None),

    # legacy / optional
    model_checkpoint: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    temperature_json: str | None = Form(default=None),
    decision_threshold: float | None = Form(default=None),
    threshold: float | None = Form(default=None),  # frontend 互換
    manifest_path: str | None = Form(default=None),
    patch_dataset_path: str | None = Form(default=None),
    label_path: str | None = Form(default=None),
    mask_path: str | None = Form(default=None),
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
    effective_device = (device or settings.default_device).strip() or settings.default_device
    effective_threshold = decision_threshold if decision_threshold is not None else threshold
    _ensure_deploy_config_override_allowed(deploy_config_path)
    _ensure_path_override_allowed(input_path, field_name="input_path")
    _ensure_path_override_allowed(model_checkpoint, field_name="model_checkpoint")
    _ensure_path_override_allowed(temperature_json, field_name="temperature_json")
    _ensure_path_override_allowed(manifest_path, field_name="manifest_path")
    _ensure_path_override_allowed(patch_dataset_path, field_name="patch_dataset_path")
    _ensure_path_override_allowed(label_path, field_name="label_path")
    _ensure_path_override_allowed(mask_path, field_name="mask_path")
    _ensure_path_override_allowed(spat_checkpoint, field_name="spat_checkpoint")
    _ensure_path_override_allowed(spec_checkpoint, field_name="spec_checkpoint")
    _ensure_runtime_override_disabled(temperature, field_name="temperature")
    _ensure_runtime_override_disabled(effective_threshold, field_name="threshold")
    _ensure_runtime_override_disabled(split_policy, field_name="split_policy")
    _ensure_runtime_override_disabled(model_type, field_name="model_type")
    _ensure_runtime_override_disabled(patch_size, field_name="patch_size")
    _ensure_runtime_override_disabled(stride, field_name="stride")

    def _run_with_paths(
        *,
        resolved_input_path: str,
        resolved_header_path: str | None,
        resolved_label_path: str | None,
        resolved_mask_path: str | None,
    ):
        return run_inference_pipeline(
            input_path=resolved_input_path,
            header_path=resolved_header_path,
            sensor=sensor,
            device=effective_device,
            deploy_config_path=deploy_config_path,
            model_checkpoint=model_checkpoint,
            spat_checkpoint=spat_checkpoint,
            spec_checkpoint=spec_checkpoint,
            temperature=temperature,
            temperature_json=temperature_json,
            decision_threshold=effective_threshold,
            manifest_path=manifest_path,
            patch_dataset_path=patch_dataset_path,
            label_path=resolved_label_path,
            mask_path=resolved_mask_path,
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

    def _save_optional_upload(tmpdir_path: Path, upload: UploadFile | None) -> str | None:
        if upload is None:
            return None
        filename = Path(upload.filename or "uploaded_sidecar").name
        out_path = tmpdir_path / filename
        with out_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        return str(out_path)

    # 1) server-side path input
    if input_path:
        if wavelength_file is not None or label_file is not None or mask_file is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                uploaded_header_path = _save_optional_upload(tmpdir_path, wavelength_file)
                uploaded_label_path = _save_optional_upload(tmpdir_path, label_file)
                uploaded_mask_path = _save_optional_upload(tmpdir_path, mask_file)
                return _run_with_paths(
                    resolved_input_path=input_path,
                    resolved_header_path=uploaded_header_path,
                    resolved_label_path=uploaded_label_path or label_path,
                    resolved_mask_path=uploaded_mask_path or mask_path,
                )

        return _run_with_paths(
            resolved_input_path=input_path,
            resolved_header_path=None,
            resolved_label_path=label_path,
            resolved_mask_path=mask_path,
        )

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

        uploaded_label_path = _save_optional_upload(tmpdir_path, label_file)
        uploaded_mask_path = _save_optional_upload(tmpdir_path, mask_file)

        result = _run_with_paths(
            resolved_input_path=str(hsi_path),
            resolved_header_path=str(wavelength_path) if wavelength_path else None,
            resolved_label_path=uploaded_label_path or label_path,
            resolved_mask_path=uploaded_mask_path or mask_path,
        )
        return result
