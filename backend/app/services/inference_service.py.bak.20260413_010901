from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

from backend.app.services.preview_service import (
    load_preview_cube,
    make_probability_overlay_png,
    make_probability_png,
    make_pseudocolor_png_from_cube,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_BASE_URL = "http://127.0.0.1:8000"


def to_public_url(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    return f"{PUBLIC_BASE_URL}/{rel.as_posix()}"


def run_inference_pipeline(
    *,
    input_path: str,
    header_path: str | None,
    sensor: str,
    device: str,
    patch_size: int,
    stride: int,
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

    cmd = [
        "python",
        "-m",
        "hsi_water_detection_app.cli",
        "--input", input_path,
        "--model_checkpoint", "dummy.pth",
        "--sensor", sensor,
        "--device", device,
        "--patch_size", str(patch_size),
        "--stride", str(stride),
        "--output_dir", str(output_dir),
    ]

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
    }

    # Create preview images from original cube + output arrays
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

    pseudocolor_png = output_dir / "pseudocolor.png"
    probability_map_png = output_dir / "probability_map.png"
    probability_overlay_png = output_dir / "probability_overlay.png"

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
        "files": {k: str(v) for k, v in files.items()},
        "urls": {k: to_public_url(v) for k, v in files.items() if v.exists()},
    }

    (output_dir / "api_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result
