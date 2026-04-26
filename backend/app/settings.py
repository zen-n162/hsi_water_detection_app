from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DEFAULT_LOCAL_DEPLOY_CONFIG = "configs/deploy/hypersigma_confmask_thr008_splitquality_stress_seed13.json"
DEFAULT_PUBLIC_DEPLOY_CONFIG = "configs/deploy/hypersigma_v3_calibrated_web.json"
DEFAULT_LOCAL_GPU_WEB_DEPLOY_CONFIG = "configs/deploy/hypersigma_v3_local_gpu_web.json"
DEFAULT_LOCAL_OUTPUT_ROOT = "outputs"
DEFAULT_PUBLIC_OUTPUT_ROOT = "runtime/outputs"
DEFAULT_LOCAL_GPU_WEB_OUTPUT_ROOT = "outputs/web_ui"


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_env_file(path: Path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _bootstrap_backend_env():
    _load_env_file(BACKEND_ROOT / ".env")
    _load_env_file(BACKEND_ROOT / ".env.local")


_bootstrap_backend_env()


def _parse_bool(value: str | None, *, default: bool) -> bool:
    cleaned = _clean_env(value)
    if cleaned is None:
        return default
    return cleaned.lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    cleaned = _clean_env(value)
    if cleaned is None:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _env_first(*keys: str) -> str | None:
    for key in keys:
        cleaned = _clean_env(os.environ.get(key))
        if cleaned is not None:
            return cleaned
    return None


def _normalize_app_mode(value: str | None) -> str:
    cleaned = (_clean_env(value) or "local").lower()
    if cleaned in {"local_gpu_web", "local-gpu-web", "gpu_web", "gpu-web"}:
        return "local_gpu_web"
    if cleaned in {"public", "prod", "production", "web"}:
        return "public"
    return "local"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _default_deploy_config(app_mode: str) -> str:
    if app_mode == "local_gpu_web":
        return DEFAULT_LOCAL_GPU_WEB_DEPLOY_CONFIG
    if app_mode == "public":
        return DEFAULT_PUBLIC_DEPLOY_CONFIG
    return DEFAULT_LOCAL_DEPLOY_CONFIG


def _default_output_root(app_mode: str) -> str:
    if app_mode == "local_gpu_web":
        return DEFAULT_LOCAL_GPU_WEB_OUTPUT_ROOT
    if app_mode == "public":
        return DEFAULT_PUBLIC_OUTPUT_ROOT
    return DEFAULT_LOCAL_OUTPUT_ROOT


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    app_mode: str
    runtime_root: Path
    default_deploy_config: Path
    output_root: Path
    output_url_prefix: str
    public_base_url: str | None
    default_device: str
    cors_allow_origins: list[str]
    cors_allow_credentials: bool
    allow_server_file_paths: bool
    allow_deploy_config_override: bool
    inference_runtime: str
    inference_conda_env: str
    inference_python_bin: str | None
    model_checkpoint_override: str | None
    temperature_json_override: str | None
    threshold_override: float | None
    manifest_path_override: str | None
    dataset_path_override: str | None

    @property
    def is_public_mode(self) -> bool:
        return self.app_mode == "public"

    @property
    def is_local_gpu_web_mode(self) -> bool:
        return self.app_mode == "local_gpu_web"

    @property
    def is_external_web_mode(self) -> bool:
        return self.app_mode in {"public", "local_gpu_web"}


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    app_mode = _normalize_app_mode(os.environ.get("HSI_APP_MODE"))
    runtime_root = _resolve_project_path(_env_first("HSI_RUNTIME_ROOT") or "runtime")

    default_deploy_config = _resolve_project_path(
        _env_first("HSI_DEPLOY_CONFIG", "HSI_DEFAULT_DEPLOY_CONFIG") or _default_deploy_config(app_mode)
    )
    output_root = _resolve_project_path(
        _clean_env(os.environ.get("HSI_OUTPUT_ROOT")) or _default_output_root(app_mode)
    )

    cors_allow_origins = _parse_csv(os.environ.get("HSI_CORS_ALLOW_ORIGINS"))
    if not cors_allow_origins and app_mode == "local":
        cors_allow_origins = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]

    allow_server_file_paths = _parse_bool(
        os.environ.get("HSI_ALLOW_SERVER_FILE_PATHS"),
        default=(app_mode != "public"),
    )
    allow_deploy_config_override = _parse_bool(
        os.environ.get("HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE"),
        default=(app_mode != "public"),
    )

    output_url_prefix = _clean_env(os.environ.get("HSI_OUTPUT_URL_PREFIX")) or "/outputs"
    if not output_url_prefix.startswith("/"):
        output_url_prefix = f"/{output_url_prefix}"
    output_url_prefix = output_url_prefix.rstrip("/") or "/outputs"
    public_base_url = _clean_env(os.environ.get("HSI_PUBLIC_BASE_URL"))
    if public_base_url is not None:
        public_base_url = public_base_url.rstrip("/")

    inference_runtime = (_clean_env(os.environ.get("HSI_INFERENCE_RUNTIME")) or "current").lower()
    if inference_runtime not in {"current", "conda"}:
        inference_runtime = "current"

    default_device = (_clean_env(os.environ.get("HSI_DEFAULT_DEVICE")) or "").lower()
    if default_device not in {"cpu", "cuda"}:
        default_device = "cuda" if app_mode == "local_gpu_web" else ("cpu" if app_mode == "public" else "cuda")

    threshold_override_raw = _env_first("HSI_THRESHOLD")
    try:
        threshold_override = float(threshold_override_raw) if threshold_override_raw is not None else None
    except ValueError as exc:
        raise ValueError(f"Invalid HSI_THRESHOLD value: {threshold_override_raw}") from exc

    return AppSettings(
        project_root=PROJECT_ROOT,
        app_mode=app_mode,
        runtime_root=runtime_root,
        default_deploy_config=default_deploy_config,
        output_root=output_root,
        output_url_prefix=output_url_prefix,
        public_base_url=public_base_url,
        default_device=default_device,
        cors_allow_origins=cors_allow_origins,
        cors_allow_credentials=_parse_bool(
            os.environ.get("HSI_CORS_ALLOW_CREDENTIALS"),
            default=False,
        ),
        allow_server_file_paths=allow_server_file_paths,
        allow_deploy_config_override=allow_deploy_config_override,
        inference_runtime=inference_runtime,
        inference_conda_env=_clean_env(os.environ.get("HSI_INFERENCE_CONDA_ENV")) or "HyperSIGMA",
        inference_python_bin=_clean_env(os.environ.get("HSI_INFERENCE_PYTHON_BIN")),
        model_checkpoint_override=_env_first("HSI_MODEL_CHECKPOINT"),
        temperature_json_override=_env_first("HSI_TEMPERATURE_JSON"),
        threshold_override=threshold_override,
        manifest_path_override=_env_first("HSI_MANIFEST_PATH"),
        dataset_path_override=_env_first("HSI_DATASET_PATH"),
    )
