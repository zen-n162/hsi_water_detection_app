from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_TOP_LEVEL_PATH_KEYS = {
    "manifest_path",
    "dataset_path",
    "model_checkpoint",
    "temperature_json",
    "output_root",
    "created_from_summary_path",
}
_CREATED_FROM_PATH_KEYS = {
    "pipeline_summary",
    "best_run_selection",
    "gui_default_recommendation",
    "leakage_comparison",
    "artifacts_manifest",
}
_REFERENCE_MODEL_PATH_KEYS = {
    "model_checkpoint",
    "threshold_json",
    "metrics_json",
}
_PUBLIC_DEPLOY_KEYS = {
    "deploy_name",
    "deploy_config_version",
    "runtime_profile",
    "run_name",
    "model_type",
    "threshold",
    "patch_size",
    "stride",
    "band_count",
    "sensor",
    "split_policy",
    "notes",
}
_PUBLIC_REFERENCE_MODEL_KEYS = {
    "role",
    "model_type",
    "threshold",
    "notes",
}


def _resolve_project_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _resolve_nested_paths(obj: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    resolved = dict(obj)
    for key in keys:
        if key in obj:
            resolved[key] = _resolve_project_path(obj.get(key))
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_safe_path_label(path_value: str | Path | None) -> str | None:
    if path_value in {None, ""}:
        return None

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return str(path)

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return path.name


def make_public_deploy_summary(config: dict[str, Any]) -> dict[str, Any]:
    summary = {key: config[key] for key in _PUBLIC_DEPLOY_KEYS if key in config}

    reference_models = config.get("reference_models") or {}
    if reference_models:
        summary["reference_models"] = {
            name: {
                key: model_cfg[key]
                for key in _PUBLIC_REFERENCE_MODEL_KEYS
                if key in model_cfg
            }
            for name, model_cfg in reference_models.items()
        }

    return summary


def load_deploy_config_bundle(deploy_config_path: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    config_path = Path(deploy_config_path) if deploy_config_path else settings.default_deploy_config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Deploy config not found: {config_path}")

    raw = _load_json(config_path)
    resolved = dict(raw)
    for key in _TOP_LEVEL_PATH_KEYS:
        if key in raw:
            resolved[key] = _resolve_project_path(raw.get(key))

    created_from = raw.get("created_from") or {}
    if created_from:
        resolved["created_from"] = _resolve_nested_paths(created_from, _CREATED_FROM_PATH_KEYS)

    reference_models = raw.get("reference_models") or {}
    if reference_models:
        resolved["reference_models"] = {
            name: _resolve_nested_paths(model_cfg, _REFERENCE_MODEL_PATH_KEYS)
            for name, model_cfg in reference_models.items()
        }

    runtime_overrides: dict[str, Any] = {}
    if settings.model_checkpoint_override:
        resolved["model_checkpoint"] = _resolve_project_path(settings.model_checkpoint_override)
        runtime_overrides["model_checkpoint"] = resolved["model_checkpoint"]
    if settings.temperature_json_override:
        resolved["temperature_json"] = _resolve_project_path(settings.temperature_json_override)
        runtime_overrides["temperature_json"] = resolved["temperature_json"]
    if settings.threshold_override is not None:
        resolved["threshold"] = settings.threshold_override
        runtime_overrides["threshold"] = settings.threshold_override
    if settings.manifest_path_override:
        resolved["manifest_path"] = _resolve_project_path(settings.manifest_path_override)
        runtime_overrides["manifest_path"] = resolved["manifest_path"]
    if settings.dataset_path_override:
        resolved["dataset_path"] = _resolve_project_path(settings.dataset_path_override)
        runtime_overrides["dataset_path"] = resolved["dataset_path"]
    if "output_root" not in resolved:
        resolved["output_root"] = str(settings.output_root)
    runtime_overrides["output_root"] = str(settings.output_root)

    return {
        "deploy_config_path": str(config_path),
        "deploy_config_relative": (
            str(config_path.relative_to(PROJECT_ROOT))
            if config_path.is_relative_to(PROJECT_ROOT)
            else str(config_path)
        ),
        "deploy_config": raw,
        "resolved": resolved,
        "runtime_overrides": runtime_overrides,
    }


def make_deploy_config_response(deploy_config_path: str | None = None) -> dict[str, Any]:
    bundle = load_deploy_config_bundle(deploy_config_path=deploy_config_path)
    settings = get_settings()
    resolved = bundle["resolved"]

    if settings.is_external_web_mode:
        deploy_config = make_public_deploy_summary(bundle["deploy_config"])
        resolved_payload = make_public_deploy_summary(resolved)
        reference_models = resolved_payload.get("reference_models", {})
        runtime_overrides = {}
        if settings.threshold_override is not None:
            runtime_overrides["threshold"] = settings.threshold_override
    else:
        deploy_config = bundle["deploy_config"]
        resolved_payload = resolved
        reference_models = resolved.get("reference_models", {})
        runtime_overrides = bundle.get("runtime_overrides", {})

    return {
        "deploy_config_path": (
            to_safe_path_label(bundle["deploy_config_path"])
            if settings.is_external_web_mode
            else bundle["deploy_config_path"]
        ),
        "deploy_config_relative": bundle["deploy_config_relative"],
        "deploy_config": deploy_config,
        "resolved": resolved_payload,
        "reference_models": reference_models,
        "runtime_overrides": runtime_overrides,
        "runtime": {
            "app_mode": settings.app_mode,
            "runtime_root": to_safe_path_label(settings.runtime_root),
            "allow_server_file_paths": settings.allow_server_file_paths,
            "allow_deploy_config_override": settings.allow_deploy_config_override,
            "output_root": to_safe_path_label(settings.output_root),
            "output_url_prefix": settings.output_url_prefix,
            "public_base_url": settings.public_base_url,
            "default_device": settings.default_device,
            "provenance_visibility": (
                "public_safe" if settings.is_external_web_mode else "full_internal"
            ),
        },
    }
