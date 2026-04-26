from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import numpy as np

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import load_hsi_cube
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands
from hsi_water_detection_app.inference.patch_infer import generate_patches
from hsi_water_detection_app.inference.reconstruct import reconstruct_from_patches
from hsi_water_detection_app.models.hyper_sigma import load_model
from research.evaluation.run_fullscene_threshold_sanity import (
    DEFAULT_CONFIDENCE,
    DEFAULT_INPUT,
    DEFAULT_VALID_MASK,
    _build_thresholds,
    _load_runs,
    _load_split_repeat_rows,
    _load_temperature,
    _load_threshold,
    _map_scores,
    _merge_repeat_metadata,
    _parse_area_caps,
    _read_raster,
    _run_prevalence,
    _summarize_map,
    _write_summary,
)


def _infer_probability_map_with_progress(
    *,
    cube: np.ndarray,
    model: Any,
    patch_size: int,
    stride: int,
    temperature: float,
    seed: str,
    progress_every: int,
) -> np.ndarray:
    patches = generate_patches(cube, patch_size=patch_size, stride=stride)
    patch_outputs: list[dict[str, Any]] = []
    total = len(patches)
    for idx, item in enumerate(patches, start=1):
        result = model.infer_patch(
            item["patch"],
            return_attn=False,
            temperature=temperature,
        )
        patch_outputs.append(
            {
                "top": item["top"],
                "left": item["left"],
                "height": item["height"],
                "width": item["width"],
                "prob_map": result["prob_map"],
            }
        )
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"[INFO] seed{seed} calibrated inference {idx}/{total} patches")

    out = reconstruct_from_patches(
        patch_outputs,
        image_shape=(cube.shape[1], cube.shape[2]),
        patch_size=patch_size,
        stride=stride,
    )
    if out is None:
        raise RuntimeError("full-scene probability reconstruction returned None")
    return out.astype(np.float32)


def _save_probability_figure(output_png: Path, prob_map: np.ndarray, valid: np.ndarray, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(np.where(valid, prob_map, np.nan), cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="wet probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def run_one(args: argparse.Namespace, run: dict[str, Any], cube: np.ndarray, y_true: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    seed = str(run["seed"])
    checkpoint = Path(run["model_checkpoint"])
    threshold_path = Path(run["threshold_json"]) if run.get("threshold_json") else None
    temperature_path = Path(run["temperature_json"]) if run.get("temperature_json") else None
    threshold = _load_threshold(threshold_path)
    temperature = _load_temperature(temperature_path)

    model = load_model(
        model_checkpoint=str(checkpoint),
        device=args.device,
        in_channels=cube.shape[0],
        patch_size=args.patch_size,
        model_type=args.model_type,
    )

    seed_dir = args.output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    calibrated_map = _infer_probability_map_with_progress(
        cube=cube,
        model=model,
        patch_size=args.patch_size,
        stride=args.stride,
        temperature=temperature,
        seed=seed,
        progress_every=args.progress_every,
    )
    np.save(seed_dir / "calibrated_probability_map.npy", calibrated_map)
    _save_probability_figure(
        seed_dir / f"seed{seed}_calibrated_probability.png",
        calibrated_map,
        valid,
        f"seed{seed} calibrated_probability",
    )

    calibrated_thresholds = _build_thresholds(
        scores=_map_scores(calibrated_map, valid),
        val_f1_threshold=threshold,
        run=run,
        area_caps=args.area_caps,
    )
    out = {
        "seed": seed,
        "mode": "fullscene_calibrated_only",
        "model_checkpoint": str(checkpoint),
        "temperature_json": None if temperature_path is None else str(temperature_path),
        "best_val_threshold_json": None if threshold_path is None else str(threshold_path),
        "temperature": temperature,
        "val_f1_tuned_threshold": threshold,
        "train_prevalence": _run_prevalence(run, "train_prevalence"),
        "validation_prevalence": _run_prevalence(run, "validation_prevalence"),
        "area_caps": args.area_caps,
        "calibrated": _summarize_map(
            name="calibrated",
            prob_map=calibrated_map,
            y_true=y_true,
            valid=valid,
            thresholds=calibrated_thresholds,
            random_baseline_repeats=args.random_baseline_repeats,
            random_baseline_seed_base=args.random_baseline_seed,
            run_seed=seed,
        ),
    }
    (seed_dir / "fullscene_threshold_sanity.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    del model
    gc.collect()
    try:
        import torch

        if args.device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibrated-only full-scene HyperSIGMA map generation")
    parser.add_argument("--runs_json", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--sensor", type=str, default="hyperion")
    parser.add_argument("--model_type", type=str, default="ss", choices=["ss", "sa"])
    parser.add_argument("--split_repeat_summary_json", type=Path, default=None)
    parser.add_argument("--area_cap", type=float, default=0.20)
    parser.add_argument("--area_caps", type=str, default=None)
    parser.add_argument("--random_baseline_repeats", type=int, default=64)
    parser.add_argument("--random_baseline_seed", type=int, default=20260426)
    parser.add_argument("--progress_every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.area_caps = _parse_area_caps(args.area_caps, args.area_cap)
    cube, _ = load_hsi_cube(str(args.input), allow_dummy=False, sensor=args.sensor)
    if args.sensor == "hyperion" and cube.shape[0] == 242:
        cube = remove_bad_bands(cube, bad_band_indices=HYPERION_BAD_BANDS_0BASED)
    cube = normalize_cube(cube)

    confidence = _read_raster(args.confidence_tif)
    valid_mask = _read_raster(args.valid_mask_tif)
    if confidence.shape != valid_mask.shape:
        raise ValueError(f"confidence/mask shape mismatch: {confidence.shape} vs {valid_mask.shape}")
    if confidence.shape != tuple(cube.shape[1:]):
        raise ValueError(f"label/cube shape mismatch: labels={confidence.shape}, cube={cube.shape}")

    valid = valid_mask == 1
    y_true = np.isin(confidence, [1, 2, 3]).astype(np.int64)
    runs = _merge_repeat_metadata(_load_runs(args.runs_json), _load_split_repeat_rows(args.split_repeat_summary_json))
    rows = [run_one(args, run, cube, y_true, valid) for run in runs]
    _write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
