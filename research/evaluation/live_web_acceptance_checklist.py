from __future__ import annotations

import argparse
import json
import math
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "live_web_acceptance"
    / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
)
REQUIRED_METADATA_KEYS = [
    "resolved_model_checkpoint",
    "resolved_temperature_json",
    "resolved_threshold",
    "resolved_run_name",
    "requested_device",
    "executed_device",
]
REQUIRED_RESULT_URL_KEYS = [
    "pseudocolor_url",
    "probability_overlay_url",
    "spatial_attention_overlay_url",
    "spectral_attention_url",
    "metadata_url",
]


def _normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def _join_url(base: str, path_or_url: str | None) -> str | None:
    if not path_or_url:
        return None
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return urllib.parse.urljoin(f"{base}/", path_or_url.lstrip("/"))


def _read_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with {exc.code}: {body}") from exc


def _read_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with {exc.code}: {body}") from exc


def _read_bytes(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with {exc.code}: {body}") from exc


def _encode_multipart(fields: dict[str, Any], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        if value is None:
            continue
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for key, path in files.items():
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{key}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def _post_multipart_json(
    url: str,
    *,
    fields: dict[str, Any],
    files: dict[str, Path],
    timeout: float,
) -> dict[str, Any]:
    payload, boundary = _encode_multipart(fields, files)
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Content-Length", str(len(payload)))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with {exc.code}: {body}") from exc


def _assert_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the live public web acceptance checks against Netlify + Render."
    )
    parser.add_argument("--frontend-url", type=str, default=None, help="Netlify production URL.")
    parser.add_argument("--backend-url", type=str, required=True, help="Render production backend URL.")
    parser.add_argument(
        "--sample-hsi",
        type=str,
        required=True,
        help="Local HSI file uploaded during preview and inference checks.",
    )
    parser.add_argument(
        "--sample-wavelength",
        type=str,
        default=None,
        help="Optional local wavelength/header sidecar file.",
    )
    parser.add_argument("--sensor", type=str, default="hyperion")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preview-band-index", type=int, default=10)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser


def _make_roi(preview_json: dict[str, Any], roi_size: int) -> dict[str, int]:
    width = int(preview_json.get("image_width") or preview_json.get("width") or 0)
    height = int(preview_json.get("image_height") or preview_json.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Preview response does not include positive image dimensions.")

    row_start = max(0, math.floor(height * 0.2))
    col_start = max(0, math.floor(width * 0.2))
    row_stop = min(height, row_start + roi_size)
    col_stop = min(width, col_start + roi_size)

    return {
        "row_start": row_start,
        "row_stop": row_stop,
        "col_start": col_start,
        "col_stop": col_stop,
    }


def main() -> None:
    args = build_parser().parse_args()
    backend_url = _normalize_base_url(args.backend_url)
    frontend_url = _normalize_base_url(args.frontend_url) if args.frontend_url else None
    sample_hsi = Path(args.sample_hsi).expanduser()
    sample_wavelength = Path(args.sample_wavelength).expanduser() if args.sample_wavelength else None
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    _assert_file_exists(sample_hsi, "sample_hsi")
    if sample_wavelength is not None:
        _assert_file_exists(sample_wavelength, "sample_wavelength")

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frontend_url": frontend_url,
        "backend_url": backend_url,
        "sample_hsi": str(sample_hsi),
        "sample_wavelength": str(sample_wavelength) if sample_wavelength else None,
        "checks": {},
        "manual_frontend_checks": [
            "Open the Netlify site in a browser.",
            "Confirm the page title and header show 'HSI Water Detection UI'.",
            "Confirm the 'Resolved inference settings' panel is visible.",
            "Confirm the grayscale preview is visible after upload.",
            "Confirm the ROI selection box is visible and draggable.",
            "Confirm the Pseudo Color, H2O Detection Overlay, Spatial Attention, and Spectral Attention cards are rendered.",
            "Confirm provenance fields remain visible in the sidebar.",
        ],
    }

    if frontend_url:
        frontend_html = _read_text(frontend_url, timeout=args.timeout_seconds)
        summary["checks"]["frontend_root"] = {
            "ok": "<div id=\"root\"></div>" in frontend_html and "HSI Water Detection UI" in frontend_html,
            "url": frontend_url,
        }

    health = _read_json(f"{backend_url}/health", timeout=args.timeout_seconds)
    summary["checks"]["backend_health"] = {
        "ok": bool(health.get("ok")),
        "response": health,
    }

    deploy_info = _read_json(f"{backend_url}/inference/deploy-config", timeout=args.timeout_seconds)
    summary["checks"]["deploy_config"] = {
        "ok": bool(deploy_info.get("resolved")),
        "deploy_config_relative": deploy_info.get("deploy_config_relative"),
        "runtime": deploy_info.get("runtime", {}),
        "runtime_overrides": deploy_info.get("runtime_overrides", {}),
    }

    files = {"hsi_file": sample_hsi}
    if sample_wavelength is not None:
        files["wavelength_file"] = sample_wavelength

    preview_json = _post_multipart_json(
        f"{backend_url}/preview/grayscale",
        fields={
            "sensor": args.sensor,
            "preview_band_index": args.preview_band_index,
        },
        files=files,
        timeout=args.timeout_seconds,
    )
    grayscale_preview_url = _join_url(
        backend_url,
        preview_json.get("grayscale_preview_url")
        or preview_json.get("preview_url")
        or preview_json.get("grayscale_url"),
    )
    if not grayscale_preview_url:
        raise ValueError("Preview response did not contain a preview URL.")

    _read_bytes(grayscale_preview_url, timeout=args.timeout_seconds)
    roi = _make_roi(preview_json, args.roi_size)
    summary["checks"]["preview_grayscale"] = {
        "ok": True,
        "url": grayscale_preview_url,
        "width": preview_json.get("image_width") or preview_json.get("width"),
        "height": preview_json.get("image_height") or preview_json.get("height"),
        "roi": roi,
    }

    inference_json = _post_multipart_json(
        f"{backend_url}/inference/run",
        fields={
            "sensor": args.sensor,
            "device": args.device,
            "row_start": roi["row_start"],
            "row_stop": roi["row_stop"],
            "col_start": roi["col_start"],
            "col_stop": roi["col_stop"],
        },
        files=files,
        timeout=args.timeout_seconds,
    )

    result_urls: dict[str, str] = {}
    for key in REQUIRED_RESULT_URL_KEYS:
        url = _join_url(backend_url, inference_json.get(key))
        if not url:
            raise ValueError(f"Inference response missing required URL: {key}")
        _read_bytes(url, timeout=args.timeout_seconds)
        result_urls[key] = url

    metadata = _read_json(result_urls["metadata_url"], timeout=args.timeout_seconds)
    missing_metadata_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_metadata_keys:
        raise ValueError(f"metadata.json is missing required keys: {missing_metadata_keys}")

    summary["checks"]["inference_run"] = {
        "ok": bool(inference_json.get("ok")),
        "requested_device": inference_json.get("requested_device"),
        "executed_device": inference_json.get("executed_device"),
        "resolved_model_checkpoint": inference_json.get("resolved_model_checkpoint"),
        "resolved_temperature_json": inference_json.get("resolved_temperature_json"),
        "resolved_threshold": inference_json.get("resolved_threshold"),
        "result_urls": result_urls,
    }
    summary["checks"]["metadata"] = {
        "ok": not missing_metadata_keys,
        "missing_keys": missing_metadata_keys,
        "resolved_model_checkpoint": metadata.get("resolved_model_checkpoint"),
        "resolved_temperature_json": metadata.get("resolved_temperature_json"),
        "resolved_threshold": metadata.get("resolved_threshold"),
        "resolved_run_name": metadata.get("resolved_run_name"),
        "requested_device": metadata.get("requested_device"),
        "executed_device": metadata.get("executed_device"),
    }

    summary["ok"] = all(check.get("ok") for check in summary["checks"].values())
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
