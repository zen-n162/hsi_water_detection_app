from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.services.deploy_config_service import (
    load_deploy_config_bundle,
    make_public_deploy_summary,
    to_safe_path_label,
)
from backend.app.services.evaluation_service import apply_probability_mask, compute_label_metrics, load_label_array
from backend.app.services.preview_service import (
    load_preview_cube,
    make_label_png,
    make_probability_overlay_png,
    make_probability_png,
    make_pseudocolor_png_from_cube,
)
from backend.app.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)
_PUBLIC_FILE_KEYS = {
    "pseudocolor_png",
    "probability_map_png",
    "probability_overlay_png",
    "label_map_png",
    "spatial_attention_png",
    "spatial_attention_overlay_png",
    "spectral_attention_png",
    "metadata_json",
    "evaluation_metrics_json",
}


def to_public_url(path: Path) -> str:
    settings = get_settings()
    rel = path.resolve().relative_to(settings.output_root.resolve())
    public_path = f"{settings.output_url_prefix}/{rel.as_posix()}"
    if settings.public_base_url:
        return f"{settings.public_base_url}{public_path}"
    return public_path


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
    settings = get_settings()
    if settings.inference_python_bin:
        return [settings.inference_python_bin]
    if settings.inference_runtime == "conda":
        return ["conda", "run", "-n", settings.inference_conda_env, "python"]
    return [sys.executable]


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _existing_file_map(files: dict[str, Path]) -> dict[str, Path]:
    return {k: v for k, v in files.items() if v.exists()}


def _inference_output_root() -> Path:
    settings = get_settings()
    return settings.output_root if settings.output_root.name == "web_ui" else settings.output_root / "web_ui"


def _public_provenance(
    *,
    deploy_bundle: dict[str, Any],
    requested_device: str,
    executed_device: str,
    resolved_temperature: float | None,
) -> dict[str, Any]:
    summary = make_public_deploy_summary(deploy_bundle["resolved"])
    summary["deploy_config"] = to_safe_path_label(deploy_bundle["deploy_config_path"])
    summary["requested_device"] = requested_device
    summary["executed_device"] = executed_device
    summary["resolved_temperature"] = resolved_temperature
    summary["provenance_visibility"] = "public_safe"
    return summary


def _remove_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except IsADirectoryError:
        return


def _prune_external_output_files(files: dict[str, Path]) -> None:
    for key, path in files.items():
        if key not in _PUBLIC_FILE_KEYS:
            _remove_file(path)


def _safe_inference_mask_summary(
    mask_summary: dict[str, Any] | None,
    *,
    external_web_mode: bool,
) -> dict[str, Any] | None:
    if mask_summary is None:
        return None
    if not external_web_mode:
        return mask_summary

    safe_summary = dict(mask_summary)
    if safe_summary.get("mask_path"):
        safe_summary["mask_path"] = to_safe_path_label(safe_summary["mask_path"])
    return safe_summary


def _persist_masked_probability_outputs(
    *,
    probability_map_npy: Path,
    probability_map: np.ndarray,
    mask_summary: dict[str, Any],
) -> dict[str, Any]:
    probability_map = probability_map.astype(np.float32, copy=False)
    np.save(probability_map_npy, probability_map)

    probability_metadata_path = probability_map_npy.with_suffix(".json")
    probability_metadata = _load_json_if_exists(probability_metadata_path)
    if probability_metadata:
        probability_metadata["inference_mask"] = mask_summary
        probability_metadata_path.write_text(
            json.dumps(probability_metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    try:
        from hsi_water_detection_app.visualization.spatial import save_probability_map

        save_probability_map(
            probability_map,
            output_path=str(probability_map_npy),
            metadata=probability_metadata or None,
            save_geotiff=True,
        )
        mask_summary["probability_map_geotiff_rewritten"] = probability_map_npy.with_suffix(".tif").exists()
    except Exception as exc:
        mask_summary["probability_map_geotiff_rewritten"] = False
        mask_summary["probability_map_geotiff_rewrite_error"] = str(exc)

    return mask_summary


def run_inference_pipeline(
    *,
    input_path: str,
    header_path: str | None,
    sensor: str,
    device: str,
    deploy_config_path: str | None = None,
    model_checkpoint: str | None = None,
    spat_checkpoint: str | None = None,
    spec_checkpoint: str | None = None,
    temperature: float | None = None,
    temperature_json: str | None = None,
    decision_threshold: float | None = None,
    manifest_path: str | None = None,
    patch_dataset_path: str | None = None,
    split_policy: str | None = None,
    label_path: str | None = None,
    mask_path: str | None = None,
    model_type: str | None = None,
    patch_size: int | None = None,
    stride: int | None = None,
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    output_dir = _inference_output_root() / datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_bundle = load_deploy_config_bundle(deploy_config_path=deploy_config_path)
    deploy_resolved = deploy_bundle["resolved"]
    deploy_config_path_resolved = deploy_bundle["deploy_config_path"]
    effective_sensor = str(deploy_resolved.get("sensor") or sensor)

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

    if not resolved_model_checkpoint and not resolved_spat_checkpoint:
        raise ValueError("Either model_checkpoint or spat_checkpoint must resolve to a valid value.")

    cmd = [
        *_build_python_command(),
        "-m",
        "hsi_water_detection_app.cli",
        "--deploy_config", deploy_config_path_resolved,
        "--input", input_path,
        "--sensor", effective_sensor,
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
    env["PYTHONPATH"] = "src:."
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    files: dict[str, Path] = {
        "probability_map_json": output_dir / "probability_map.json",
        "probability_map_npy": output_dir / "probability_map.npy",
        "probability_map_tif": output_dir / "probability_map.tif",
        "spatial_attention_npy": output_dir / "spatial_attention.npy",
        "spatial_attention_png": output_dir / "spatial_attention.png",
        "spatial_attention_tif": output_dir / "spatial_attention.tif",
        "spatial_attention_overlay_png": output_dir / "spatial_attention_overlay.png",
        "spectral_attention_csv": output_dir / "spectral_attention.csv",
        "spectral_attention_png": output_dir / "spectral_attention.png",
        "run_config_json": output_dir / "run_config.json",
        "metadata_json": output_dir / "metadata.json",
        "api_result_json": output_dir / "api_result.json",
        "evaluation_metrics_json": output_dir / "evaluation_metrics.json",
        "inference_mask_json": output_dir / "inference_mask.json",
        "inference_valid_mask_npy": output_dir / "inference_valid_mask.npy",
    }

    pseudocolor_png = output_dir / "pseudocolor.png"
    probability_map_png = output_dir / "probability_map.png"
    probability_overlay_png = output_dir / "probability_overlay.png"
    label_map_png = output_dir / "label_map.png"

    inference_mask: dict[str, Any] | None = None
    inference_mask_error: str | None = None
    if mask_path and proc.returncode == 0 and files["probability_map_npy"].exists():
        try:
            masked_probability, valid_mask, inference_mask = apply_probability_mask(
                probability_map_npy=files["probability_map_npy"],
                mask_path=mask_path,
                output_json=files["inference_mask_json"],
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
            np.save(files["inference_valid_mask_npy"], valid_mask.astype(np.uint8))
            inference_mask = _persist_masked_probability_outputs(
                probability_map_npy=files["probability_map_npy"],
                probability_map=masked_probability,
                mask_summary=inference_mask,
            )
            files["inference_mask_json"].write_text(
                json.dumps(inference_mask, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            inference_mask_error = str(exc)
            _remove_file(files["inference_mask_json"])

    preview_error = None
    if proc.returncode == 0 and files["probability_map_npy"].exists():
        probability_shape = tuple(np.load(files["probability_map_npy"]).shape)
        label_visualization_error = None
        if label_path:
            try:
                label_arr = load_label_array(
                    label_path,
                    target_shape=probability_shape,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
                valid_mask = (
                    np.load(files["inference_valid_mask_npy"]).astype(bool)
                    if files["inference_valid_mask_npy"].exists()
                    else None
                )
                make_label_png(
                    label=label_arr,
                    out_png=label_map_png,
                    valid_mask=valid_mask,
                )
            except Exception as exc:
                label_visualization_error = str(exc)

        preview_cube = load_preview_cube(
            input_path=input_path,
            sensor=effective_sensor,
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
            valid_mask_npy=files["inference_valid_mask_npy"],
        )
        if pseudocolor_png.exists():
            make_probability_overlay_png(
                prob_map_npy=files["probability_map_npy"],
                pseudocolor_png=pseudocolor_png,
                out_png=probability_overlay_png,
                alpha=0.45,
                valid_mask_npy=files["inference_valid_mask_npy"],
            )
    else:
        label_visualization_error = None
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
    if label_map_png.exists():
        files["label_map_png"] = label_map_png

    evaluation_metrics: dict[str, Any] | None = None
    evaluation_error: str | None = None
    if label_path and proc.returncode == 0 and files["probability_map_npy"].exists():
        try:
            threshold_for_eval = float(resolved_threshold) if resolved_threshold is not None else 0.5
            evaluation_metrics = compute_label_metrics(
                probability_map_npy=files["probability_map_npy"],
                threshold=threshold_for_eval,
                label_path=label_path,
                mask_path=mask_path,
                output_json=files["evaluation_metrics_json"],
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        except Exception as exc:
            evaluation_error = str(exc)
            _remove_file(files["evaluation_metrics_json"])

    cli_run_config = _load_json_if_exists(output_dir / "run_config.json")

    resolved_temperature = cli_run_config.get("resolved_temperature", resolved_temperature)
    resolved_temperature_json = cli_run_config.get("resolved_temperature_json", resolved_temperature_json)
    resolved_threshold = cli_run_config.get("decision_threshold", resolved_threshold)
    resolved_manifest_path = cli_run_config.get("manifest_path", resolved_manifest_path)
    resolved_patch_dataset_path = cli_run_config.get("patch_dataset_path", resolved_patch_dataset_path)
    resolved_split_policy = cli_run_config.get("split_policy", resolved_split_policy)
    resolved_run_name = cli_run_config.get("resolved_run_name", resolved_run_name)
    resolved_band_count = cli_run_config.get("band_count", resolved_band_count)
    resolved_model_type = cli_run_config.get("model_type", resolved_model_type)
    resolved_patch_size = cli_run_config.get("patch_size", resolved_patch_size)
    resolved_stride = cli_run_config.get("stride", resolved_stride)
    requested_device = cli_run_config.get("requested_device", device)
    executed_device = cli_run_config.get("executed_device", device)

    if settings.is_external_web_mode:
        logger.info(
            "external_web_inference internal_provenance=%s",
            json.dumps(
                {
                    "deploy_config_path": deploy_config_path_resolved,
                    "resolved_model_checkpoint": resolved_model_checkpoint,
                    "resolved_temperature_json": resolved_temperature_json,
                    "resolved_manifest": resolved_manifest_path,
                    "resolved_dataset": resolved_patch_dataset_path,
                    "label_path": label_path,
                    "mask_path": mask_path,
                    "inference_mask_error": inference_mask_error,
                    "evaluation_error": evaluation_error,
                    "requested_device": requested_device,
                    "executed_device": executed_device,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    if settings.is_external_web_mode:
        _prune_external_output_files(files)

    existing_files = _existing_file_map(files)
    file_map = {k: str(v) for k, v in existing_files.items()}
    url_map = {k: to_public_url(v) for k, v in existing_files.items()}
    public_inference_mask = _safe_inference_mask_summary(
        inference_mask,
        external_web_mode=settings.is_external_web_mode,
    )

    public_provenance = _public_provenance(
        deploy_bundle=deploy_bundle,
        requested_device=requested_device,
        executed_device=executed_device,
        resolved_temperature=resolved_temperature,
    )

    result: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "stdout": (
            None
            if settings.is_external_web_mode
            else proc.stdout
        ),
        "stderr": (
            "Backend inference failed on the owner-operated GPU runtime. Check the backend PC logs."
            if settings.is_external_web_mode and proc.returncode != 0
            else proc.stderr
        ),
        "output_dir": (
            to_safe_path_label(output_dir)
            if settings.is_external_web_mode
            else str(output_dir)
        ),
        "resolved_output_root": to_safe_path_label(settings.output_root),
        "deploy_config_path": (
            to_safe_path_label(deploy_config_path_resolved)
            if settings.is_external_web_mode
            else deploy_config_path_resolved
        ),
        "resolved_deploy_config_path": (
            to_safe_path_label(deploy_config_path_resolved)
            if settings.is_external_web_mode
            else deploy_config_path_resolved
        ),
        "resolved_model_checkpoint": (
            None if settings.is_external_web_mode else resolved_model_checkpoint
        ),
        "resolved_spat_checkpoint": (
            None if settings.is_external_web_mode else resolved_spat_checkpoint
        ),
        "resolved_spec_checkpoint": (
            None if settings.is_external_web_mode else resolved_spec_checkpoint
        ),
        "resolved_temperature_json": (
            None if settings.is_external_web_mode else resolved_temperature_json
        ),
        "resolved_temperature": resolved_temperature,
        "resolved_threshold": resolved_threshold,
        "resolved_manifest": (
            None if settings.is_external_web_mode else resolved_manifest_path
        ),
        "resolved_dataset": (
            None if settings.is_external_web_mode else resolved_patch_dataset_path
        ),
        "resolved_label": (
            None if settings.is_external_web_mode else label_path
        ),
        "resolved_mask": (
            None if settings.is_external_web_mode else mask_path
        ),
        "resolved_split_policy": resolved_split_policy,
        "resolved_run_name": resolved_run_name,
        "resolved_band_count": resolved_band_count,
        "resolved_model_type": resolved_model_type,
        "resolved_patch_size": resolved_patch_size,
        "resolved_stride": resolved_stride,
        "requested_device": requested_device,
        "executed_device": executed_device,
        "model_type": resolved_model_type,
        "sensor": effective_sensor,
        "preview_generation_error": preview_error,
        "label_visualization_error": label_visualization_error,
        "inference_mask": public_inference_mask,
        "inference_mask_error": inference_mask_error,
        "files": (
            {}
            if settings.is_external_web_mode
            else file_map
        ),
        "urls": url_map,
        "provenance_visibility": (
            "public_safe" if settings.is_external_web_mode else "full_internal"
        ),

        # frontend が直接使うトップレベルURL
        "pseudocolor_url": url_map.get("pseudocolor_png"),
        "label_map_url": url_map.get("label_map_png"),
        "probability_map_url": url_map.get("probability_map_png"),
        "probability_overlay_url": url_map.get("probability_overlay_png"),
        "spatial_attention_url": url_map.get("spatial_attention_png"),
        "spatial_attention_overlay_url": url_map.get("spatial_attention_overlay_png"),
        "spectral_attention_url": url_map.get("spectral_attention_png"),
        "metadata_url": None,
        "api_result_url": None if settings.is_external_web_mode else url_map.get("api_result_json"),
        "inference_mask_url": url_map.get("inference_mask_json"),
        "evaluation_metrics": evaluation_metrics,
        "evaluation_error": evaluation_error,
        "evaluation_metrics_url": url_map.get("evaluation_metrics_json"),
    }

    if settings.is_external_web_mode:
        result["model_provenance"] = public_provenance
        result["deploy_config"] = make_public_deploy_summary(deploy_bundle["deploy_config"])
        result["deploy_reference_models"] = public_provenance.get("reference_models", {})
        result["created_from"] = {}
    else:
        result["model_provenance"] = {
            "deploy_config_path": deploy_config_path_resolved,
            "deploy_name": deploy_resolved.get("deploy_name"),
            "run_name": resolved_run_name,
            "output_root": str(settings.output_root),
            "manifest_path": resolved_manifest_path,
            "patch_dataset_path": resolved_patch_dataset_path,
            "model_checkpoint_path": resolved_model_checkpoint,
            "spat_checkpoint_path": resolved_spat_checkpoint,
            "spec_checkpoint_path": resolved_spec_checkpoint,
            "calibration_file_path": resolved_temperature_json,
            "temperature": resolved_temperature,
            "threshold": resolved_threshold,
            "label_path": label_path,
            "mask_path": mask_path,
            "inference_mask": inference_mask,
            "inference_mask_error": inference_mask_error,
            "split_policy": resolved_split_policy,
            "band_count": resolved_band_count,
            "reference_models": deploy_resolved.get("reference_models", {}),
        }
        result["deploy_config"] = deploy_bundle["deploy_config"]
        result["deploy_reference_models"] = deploy_resolved.get("reference_models", {})
        result["created_from"] = deploy_resolved.get("created_from", {})

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deploy_config_path": result["deploy_config_path"],
        "resolved_threshold": result["resolved_threshold"],
        "resolved_split_policy": result["resolved_split_policy"],
        "resolved_run_name": result["resolved_run_name"],
        "resolved_band_count": result["resolved_band_count"],
        "resolved_model_type": result["resolved_model_type"],
        "resolved_patch_size": result["resolved_patch_size"],
        "resolved_stride": result["resolved_stride"],
        "resolved_output_root": result["resolved_output_root"],
        "requested_device": result["requested_device"],
        "executed_device": result["executed_device"],
        "sensor": result["sensor"],
        "resolved_temperature": result["resolved_temperature"],
        "provenance_visibility": result["provenance_visibility"],
        "model_provenance": result["model_provenance"],
        "created_from": result["created_from"],
        "urls": result["urls"],
        "pseudocolor_url": result["pseudocolor_url"],
        "label_map_url": result["label_map_url"],
        "label_visualization_error": result["label_visualization_error"],
        "probability_map_url": result["probability_map_url"],
        "probability_overlay_url": result["probability_overlay_url"],
        "spatial_attention_url": result["spatial_attention_url"],
        "spatial_attention_overlay_url": result["spatial_attention_overlay_url"],
        "spectral_attention_url": result["spectral_attention_url"],
        "inference_mask": result["inference_mask"],
        "inference_mask_error": result["inference_mask_error"],
        "inference_mask_url": result["inference_mask_url"],
        "evaluation_metrics": result["evaluation_metrics"],
        "evaluation_error": result["evaluation_error"],
        "evaluation_metrics_url": result["evaluation_metrics_url"],
    }

    if not settings.is_external_web_mode:
        metadata.update(
            {
                "resolved_model_checkpoint": result["resolved_model_checkpoint"],
                "resolved_spat_checkpoint": result["resolved_spat_checkpoint"],
                "resolved_spec_checkpoint": result["resolved_spec_checkpoint"],
                "resolved_temperature_json": result["resolved_temperature_json"],
                "resolved_manifest": result["resolved_manifest"],
                "resolved_dataset": result["resolved_dataset"],
                "resolved_label": result["resolved_label"],
                "resolved_mask": result["resolved_mask"],
                "files": result["files"],
            }
        )

    metadata_path = output_dir / "metadata.json"
    metadata_url = to_public_url(metadata_path)
    metadata["metadata_url"] = metadata_url
    metadata["urls"] = {
        **metadata["urls"],
        "metadata_json": metadata_url,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["metadata_url"] = metadata_url
    result["urls"]["metadata_json"] = metadata_url
    if not settings.is_external_web_mode:
        result["files"]["metadata_json"] = str(metadata_path)
    if not settings.is_external_web_mode:
        api_result_path = output_dir / "api_result.json"
        api_result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        api_result_url = to_public_url(api_result_path)
        result["api_result_url"] = api_result_url
        result["urls"]["api_result_json"] = api_result_url
        result["files"]["api_result_json"] = str(api_result_path)

    return result
