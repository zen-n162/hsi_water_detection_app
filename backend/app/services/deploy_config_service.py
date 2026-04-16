from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOY_CONFIG_RELATIVE = "configs/deploy/hypersigma_v3_calibrated.json"
DEFAULT_DEPLOY_CONFIG_PATH = PROJECT_ROOT / DEFAULT_DEPLOY_CONFIG_RELATIVE

_TOP_LEVEL_PATH_KEYS = {
    "manifest_path",
    "dataset_path",
    "model_checkpoint",
    "temperature_json",
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


def load_deploy_config_bundle(deploy_config_path: str | None = None) -> dict[str, Any]:
    config_path = Path(deploy_config_path) if deploy_config_path else DEFAULT_DEPLOY_CONFIG_PATH
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

    return {
        "deploy_config_path": str(config_path),
        "deploy_config_relative": str(config_path.relative_to(PROJECT_ROOT)),
        "deploy_config": raw,
        "resolved": resolved,
    }


def make_deploy_config_response(deploy_config_path: str | None = None) -> dict[str, Any]:
    bundle = load_deploy_config_bundle(deploy_config_path=deploy_config_path)
    resolved = bundle["resolved"]
    return {
        "deploy_config_path": bundle["deploy_config_path"],
        "deploy_config_relative": bundle["deploy_config_relative"],
        "deploy_config": bundle["deploy_config"],
        "resolved": resolved,
        "reference_models": resolved.get("reference_models", {}),
    }
