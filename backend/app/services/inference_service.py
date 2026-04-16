from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any

from backend.app.services.deploy_config_service import (
    load_deploy_config_bundle,
)
from backend.app.services.preview_service import (
    load_preview_cube,
    make_probability_overlay_png,
    make_probability_png,
    make_pseudocolor_png_from_cube,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_INFERENCE_CONDA_ENV = "HyperSIGMA"


def to_public_url(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    return f"{PUBLIC_BASE_URL}/{rel.as_posix()}"


def _resolve_optional_path(p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_python_command() -> list[str]:
    conda_env = os.environ.get("HSI_INFERENCE_CONDA_ENV", DEFAULT_INFERENCE_CONDA_ENV).strip()
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return ["python"]


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def run_inference_pipeline(
    *,
    input_path: str,
    header_path: str | None,
    sensor: str,
    device: str,
    deploy_config_path: str | None = None,
    model_checkpoint: str | None,
    spat_checkpoint: str | None,
    spec_checkpoint: str | None,
    temperature: float | None = None,
    temperature_json: str | None = None,
    decision_threshold: float | None = None,
    manifest_path: str | None = None,
    patch_dataset_path: str | None = None,
    split_policy: str | None = None,
    model_type: str | None = None,
    patch_size: int | None = None,
    stride: int | None = None,
    row_start: int | None,
    row_stop: int | None,
    col_start: int | None,
    col_stop: int | None,
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
):
    output_dir = PROJECT_ROOT / "outputs" / "web_ui" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_bundle = load_deploy_config_bundle(deploy_config_path=deploy_config_path)
    deploy_resolved = deploy_bundle["resolved"]
    deploy_config_path_resolved = deploy_bundle["deploy_config_path"]

    resolved_model_checkpoint = _resolve_optional_path(
        _normalize_optional_string(model_checkpoint) or deploy_resolved.get("model_checkpoint")
    )
    resolved_spat_checkpoint = _resolve_optional_path(
        _normalize_optional_string(spat_checkpoint) or deploy_resolved.get("spat_checkpoint")
    )
    resolved_spec_checkpoint = _resolve_optional_path(
        _normalize_optional_string(spec_checkpoint) or deploy_resolved.get("spec_checkpoint")
    )
    resolved_temperature_json = _resolve_optional_path(
        _normalize_optional_string(temperature_json) or deploy_resolved.get("temperature_json")
    )
    resolved_temperature = _first_non_none(temperature, deploy_resolved.get("temperature"))
    resolved_threshold = _first_non_none(decision_threshold, deploy_resolved.get("threshold"))
    resolved_manifest_path = _resolve_optional_path(
        _normalize_optional_string(manifest_path) or deploy_resolved.get("manifest_path")
    )
    resolved_patch_dataset_path = _resolve_optional_path(
        _normalize_optional_string(patch_dataset_path) or deploy_resolved.get("dataset_path")
    )
    resolved_split_policy = _normalize_optional_string(split_policy) or deploy_resolved.get("split_policy")
    resolved_model_type = _normalize_optional_string(model_type) or deploy_resolved.get("model_type") or "ss"
    resolved_patch_size = int(_first_non_none(patch_size, deploy_resolved.get("patch_size"), 64))
    resolved_stride = int(_first_non_none(stride, deploy_resolved.get("stride"), 32))
    resolved_run_name = deploy_resolved.get("run_name")
    resolved_band_count = deploy_resolved.get("band_count")

    if not resolved_model_checkpoint:
        raise ValueError(
            f"model_checkpoint is required after resolving deploy config: {deploy_config_path_resolved}"
        )

    cmd = [
        *_build_python_command(),
        "-m",
        "hsi_water_detection_app.cli",
        "--deploy_config", deploy_config_path_resolved,
        "--input", input_path,
        "--sensor", sensor,
        "--device", device,
        "--model_type", resolved_model_type,
        "--patch_size", str(resolved_patch_size),
        "--stride", str(resolved_stride),
        "--output_dir", str(output_dir),
    ]

    if resolved_model_checkpoint:
        cmd += ["--model_checkpoint", resolved_model_checkpoint]
    if resolved_spat_checkpoint:
        cmd += ["--spat_checkpoint", resolved_spat_checkpoint]
    if resolved_spec_checkpoint:
        cmd += ["--spec_checkpoint", resolved_spec_checkpoint]
    if resolved_temperature is not None:
        cmd += ["--temperature", str(resolved_temperature)]
    if resolved_temperature_json:
        cmd += ["--temperature_json", resolved_temperature_json]
    if resolved_threshold is not None:
        cmd += ["--decision_threshold", str(resolved_threshold)]
    if resolved_manifest_path:
        cmd += ["--manifest_path", resolved_manifest_path]
    if resolved_patch_dataset_path:
        cmd += ["--patch_dataset_path", resolved_patch_dataset_path]
    if resolved_split_policy:
        cmd += ["--split_policy", str(resolved_split_policy)]

    if header_path:
        cmd += ["--header", header_path]

    if all(v is not None for v in [row_start, row_stop, col_start, col_stop]):
        cmd += [
            "--row_start", str(row_start),
            "--row_stop", str(row_stop),
            "--col_start", str(col_start),
            "--col_stop", str(col_stop),
        ]

    if all(v is not None for v in [xmin, ymin, xmax, ymax]):
        cmd += [
            "--xmin", str(xmin),
            "--ymin", str(ymin),
            "--xmax", str(xmax),
            "--ymax", str(ymax),
        ]

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    files = {
        "probability_map_json": output_dir / "probability_map.json",
        "probability_map_npy": output_dir / "probability_map.npy",
        "probability_map_tif": output_dir / "probability_map.tif",
        "spatial_attention_npy": output_dir / "spatial_attention.npy",
        "spatial_attention_png": output_dir / "spatial_attention.png",
        "spatial_attention_tif": output_dir / "spatial_attention.tif",
        "spatial_attention_overlay_png": output_dir / "spatial_attention_overlay.png",
        "spectral_attention_csv": output_dir / "spectral_attention.csv",
        "spectral_attention_png": output_dir / "spectral_attention.png",
        "run_config": output_dir / "run_config.json",
        "metadata_json": output_dir / "metadata.json",
        "api_result_json": output_dir / "api_result.json",
    }

    pseudocolor_png = output_dir / "pseudocolor.png"
    probability_map_png = output_dir / "probability_map.png"
    probability_overlay_png = output_dir / "probability_overlay.png"

    preview_error = None
    if proc.returncode == 0 and files["probability_map_npy"].exists():
        preview_cube = load_preview_cube(
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
        if preview_cube is not None:
            make_pseudocolor_png_from_cube(
                cube=preview_cube,
                out_png=pseudocolor_png,
                rgb_bands=(3, 9, 17),
            )

        make_probability_png(
            prob_map_npy=files["probability_map_npy"],
            out_png=probability_map_png,
        )

        make_probability_overlay_png(
            prob_map_npy=files["probability_map_npy"],
            pseudocolor_png=pseudocolor_png,
            out_png=probability_overlay_png,
            alpha=0.45,
        )
    else:
        preview_error = (
            "Skipped preview artifact generation because the CLI command failed "
            "or probability_map.npy was not produced."
        )

    if pseudocolor_png.exists():
        files["pseudocolor_png"] = pseudocolor_png
    if probability_map_png.exists():
        files["probability_map_png"] = probability_map_png
    if probability_overlay_png.exists():
        files["probability_overlay_png"] = probability_overlay_png

    result = {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_dir": str(output_dir),
        "deploy_config_path": deploy_config_path_resolved,
        "resolved_deploy_config_path": deploy_config_path_resolved,
        "resolved_model_checkpoint": resolved_model_checkpoint,
        "resolved_spat_checkpoint": resolved_spat_checkpoint,
        "resolved_spec_checkpoint": resolved_spec_checkpoint,
        "resolved_temperature_json": resolved_temperature_json,
        "resolved_temperature": resolved_temperature,
        "resolved_threshold": resolved_threshold,
        "resolved_manifest": resolved_manifest_path,
        "resolved_dataset": resolved_patch_dataset_path,
        "resolved_split_policy": resolved_split_policy,
        "resolved_run_name": resolved_run_name,
        "resolved_band_count": resolved_band_count,
        "resolved_model_type": resolved_model_type,
        "resolved_patch_size": resolved_patch_size,
        "resolved_stride": resolved_stride,
        "requested_device": device,
        "executed_device": device,
        "model_type": resolved_model_type,
        "preview_generation_error": preview_error,
        "files": {k: str(v) for k, v in files.items()},
        "urls": {k: to_public_url(v) for k, v in files.items() if v.exists()},
    }

    cli_run_config = _load_json_if_exists(output_dir / "run_config.json")
    result["resolved_temperature"] = cli_run_config.get("resolved_temperature", result["resolved_temperature"])
    result["resolved_temperature_json"] = cli_run_config.get("resolved_temperature_json", result["resolved_temperature_json"])
    result["resolved_threshold"] = cli_run_config.get("decision_threshold", result["resolved_threshold"])
    result["resolved_manifest"] = cli_run_config.get("manifest_path", result["resolved_manifest"])
    result["resolved_dataset"] = cli_run_config.get("patch_dataset_path", result["resolved_dataset"])
    result["resolved_split_policy"] = cli_run_config.get("split_policy", result["resolved_split_policy"])
    result["resolved_run_name"] = cli_run_config.get("resolved_run_name", result["resolved_run_name"])
    result["resolved_band_count"] = cli_run_config.get("band_count", result["resolved_band_count"])
    result["resolved_model_type"] = cli_run_config.get("model_type", result["resolved_model_type"])
    result["resolved_patch_size"] = cli_run_config.get("patch_size", result["resolved_patch_size"])
    result["resolved_stride"] = cli_run_config.get("stride", result["resolved_stride"])
    result["requested_device"] = cli_run_config.get("requested_device", result["requested_device"])
    result["executed_device"] = cli_run_config.get("executed_device", result["executed_device"])
    result["model_provenance"] = {
        "deploy_config_path": deploy_config_path_resolved,
        "deploy_name": deploy_resolved.get("deploy_name"),
        "run_name": cli_run_config.get("resolved_run_name", resolved_run_name),
        "manifest_path": cli_run_config.get("manifest_path", resolved_manifest_path),
        "patch_dataset_path": cli_run_config.get("patch_dataset_path", resolved_patch_dataset_path),
        "model_checkpoint_path": cli_run_config.get("model_checkpoint", resolved_model_checkpoint),
        "spat_checkpoint_path": cli_run_config.get("spat_checkpoint", resolved_spat_checkpoint),
        "spec_checkpoint_path": cli_run_config.get("spec_checkpoint", resolved_spec_checkpoint),
        "calibration_file_path": cli_run_config.get("resolved_temperature_json", resolved_temperature_json),
        "temperature": cli_run_config.get("resolved_temperature", resolved_temperature),
        "threshold": cli_run_config.get("decision_threshold", resolved_threshold),
        "split_policy": cli_run_config.get("split_policy", resolved_split_policy),
        "band_count": cli_run_config.get("band_count", resolved_band_count),
        "reference_models": deploy_resolved.get("reference_models", {}),
    }
    result["deploy_config"] = deploy_bundle["deploy_config"]
    result["deploy_reference_models"] = deploy_resolved.get("reference_models", {})
    result["created_from"] = deploy_resolved.get("created_from", {})

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deploy_config_path": deploy_config_path_resolved,
        "resolved_model_checkpoint": result["resolved_model_checkpoint"],
        "resolved_temperature_json": result["resolved_temperature_json"],
        "resolved_temperature": result["resolved_temperature"],
        "resolved_threshold": result["resolved_threshold"],
        "resolved_manifest": result["resolved_manifest"],
        "resolved_dataset": result["resolved_dataset"],
        "resolved_split_policy": result["resolved_split_policy"],
        "resolved_run_name": result["resolved_run_name"],
        "resolved_band_count": result["resolved_band_count"],
        "resolved_model_type": result["resolved_model_type"],
        "resolved_patch_size": result["resolved_patch_size"],
        "resolved_stride": result["resolved_stride"],
        "requested_device": result["requested_device"],
        "executed_device": result["executed_device"],
        "model_provenance": result["model_provenance"],
        "created_from": result["created_from"],
        "files": result["files"],
        "urls": result["urls"],
    }
    files["metadata_json"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "api_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["urls"] = {k: to_public_url(v) for k, v in files.items() if v.exists()}
    files["api_result_json"] = output_dir / "api_result.json"
    result["files"]["api_result_json"] = str(files["api_result_json"])
    result["urls"] = {k: to_public_url(v) for k, v in files.items() if v.exists()}
    metadata["files"] = result["files"]
    metadata["urls"] = result["urls"]
    files["metadata_json"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "api_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result
