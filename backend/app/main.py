import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes_inference import router as inference_router
from backend.app.api.routes_preview import router as preview_router
from backend.app.services.deploy_config_service import to_safe_path_label
from backend.app.settings import get_settings

settings = get_settings()

app = FastAPI(title="HSI Water Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_dir = settings.output_root
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    settings.output_url_prefix,
    StaticFiles(directory=str(outputs_dir)),
    name="outputs",
)


def _json_error(
    *,
    status_code: int,
    detail: str,
    code: str,
    hint: str | None = None,
):
    safe_detail = detail
    if settings.is_external_web_mode and code in {"missing_runtime_asset", "invalid_request", "internal_error"}:
        safe_detail = {
            "missing_runtime_asset": "A required runtime asset is missing on the backend PC.",
            "invalid_request": "The request could not be processed by the backend service.",
            "internal_error": "The backend service hit an internal error. Check the backend PC logs.",
        }[code]

    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "detail": safe_detail,
            "code": code,
            "hint": hint,
        },
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(_request: Request, exc: PermissionError):
    return _json_error(
        status_code=403,
        detail=str(exc),
        code="forbidden_override",
        hint="Use file upload inputs and the server default deploy profile in external web mode.",
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(_request: Request, exc: FileNotFoundError):
    return _json_error(
        status_code=500,
        detail=str(exc),
        code="missing_runtime_asset",
        hint="Check the configured deploy profile and mounted runtime assets on the backend service.",
    )


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    return _json_error(
        status_code=400,
        detail=str(exc),
        code="invalid_request",
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    return _json_error(
        status_code=500,
        detail=str(exc),
        code="internal_error",
    )


@app.get("/health")
def health():
    deploy_config_ref = (
        to_safe_path_label(settings.default_deploy_config)
        if settings.is_external_web_mode
        else str(settings.default_deploy_config)
    )

    return {
        "ok": True,
        "app_mode": settings.app_mode,
        "default_device": settings.default_device,
        "public_base_url": settings.public_base_url,
        "runtime_root": to_safe_path_label(settings.runtime_root),
        "default_deploy_config": deploy_config_ref,
        "default_deploy_config_exists": settings.default_deploy_config.exists(),
        "output_root": to_safe_path_label(settings.output_root),
        "output_url_prefix": settings.output_url_prefix,
        "allow_server_file_paths": settings.allow_server_file_paths,
        "allow_deploy_config_override": settings.allow_deploy_config_override,
        "cors_allow_origins": settings.cors_allow_origins,
        "inference_runtime": settings.inference_runtime,
        "model_checkpoint_override": (
            None if settings.is_external_web_mode else settings.model_checkpoint_override
        ),
        "temperature_json_override": (
            None if settings.is_external_web_mode else settings.temperature_json_override
        ),
        "threshold_override": settings.threshold_override,
        "manifest_path_override": (
            None if settings.is_external_web_mode else settings.manifest_path_override
        ),
        "dataset_path_override": (
            None if settings.is_external_web_mode else settings.dataset_path_override
        ),
        "render_service": os.environ.get("RENDER_SERVICE_NAME"),
        "provenance_visibility": "public_safe" if settings.is_external_web_mode else "full_internal",
    }

app.include_router(inference_router)
app.include_router(preview_router)
