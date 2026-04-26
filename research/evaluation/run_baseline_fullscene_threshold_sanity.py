from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import torch
from torch import nn

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import load_hsi_cube
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands
from hsi_water_detection_app.inference.patch_infer import generate_patches
from research.evaluation.run_fullscene_threshold_sanity import (
    DEFAULT_CONFIDENCE,
    DEFAULT_INPUT,
    DEFAULT_VALID_MASK,
    _build_thresholds,
    _load_runs,
    _load_split_repeat_rows,
    _load_threshold,
    _merge_repeat_metadata,
    _parse_area_caps,
    _read_raster,
    _select_patch_subset,
    _summarize_scores,
    _write_summary,
)


class BaselineNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.head(feat).squeeze(1)


def _load_baseline_model(checkpoint: Path, *, in_channels: int, device: str) -> BaselineNet:
    model = BaselineNet(in_channels=in_channels)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _infer_sampled_scores(
    *,
    cube: np.ndarray,
    model: BaselineNet,
    patch_size: int,
    stride: int,
    y_true: np.ndarray,
    valid: np.ndarray,
    max_patches: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    patches = generate_patches(cube, patch_size=patch_size, stride=stride)
    sampled = _select_patch_subset(patches, max_patches)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    valid_pixels = 0

    with torch.no_grad():
        for idx, item in enumerate(sampled):
            top = int(item["top"])
            left = int(item["left"])
            h = int(item["height"])
            w = int(item["width"])
            patch_valid = valid[top : top + h, left : left + w]
            if not np.any(patch_valid):
                continue

            patch = np.asarray(item["patch"], dtype=np.float32)
            x = torch.from_numpy(patch).unsqueeze(0).to(device)
            logit = model(x)
            score = float(torch.sigmoid(logit).detach().cpu().item())

            patch_labels = y_true[top : top + h, left : left + w][patch_valid]
            local_rows, local_cols = np.nonzero(patch_valid)
            labels.append(patch_labels.astype(np.int64))
            scores.append(np.full(int(patch_valid.sum()), score, dtype=np.float64))
            rows.append((local_rows + top).astype(np.int32))
            cols.append((local_cols + left).astype(np.int32))
            valid_pixels += int(patch_valid.sum())
            if (idx + 1) % 25 == 0:
                print(f"[INFO] baseline sampled inference progress {idx + 1}/{len(sampled)} patches")

    if not scores:
        raise RuntimeError("No valid pixels found in sampled full-scene patches")
    meta = {
        "total_generated_patches": len(patches),
        "sampled_patches": len(sampled),
        "sampled_valid_pixels": valid_pixels,
        "score_granularity": "one_scalar_per_patch_repeated_over_valid_pixels",
    }
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(rows), np.concatenate(cols), meta


def run_one(args: argparse.Namespace, run: dict[str, Any], cube: np.ndarray, y_true: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    seed = str(run["seed"])
    checkpoint = Path(run["model_checkpoint"])
    threshold_path = Path(run["threshold_json"]) if run.get("threshold_json") else None
    threshold = _load_threshold(threshold_path)

    model = _load_baseline_model(
        checkpoint,
        in_channels=cube.shape[0],
        device=args.device,
    )

    seed_dir = args.output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    scores, labels, rows, cols, sample_meta = _infer_sampled_scores(
        cube=cube,
        model=model,
        patch_size=args.patch_size,
        stride=args.stride,
        y_true=y_true,
        valid=valid,
        max_patches=args.max_patches,
        device=args.device,
    )
    thresholds = _build_thresholds(
        scores=scores,
        val_f1_threshold=threshold,
        run=run,
        area_caps=args.area_caps,
    )
    np.savez_compressed(
        seed_dir / "sampled_scores.npz",
        scores=scores,
        labels=labels,
        rows=rows,
        cols=cols,
    )
    out = {
        "model_name": "baseline",
        "seed": seed,
        "mode": "sampled_fullscene_coverage",
        "model_checkpoint": str(checkpoint),
        "best_val_threshold_json": None if threshold_path is None else str(threshold_path),
        "temperature": None,
        "val_f1_tuned_threshold": threshold,
        "train_prevalence": run.get("train_prevalence"),
        "validation_prevalence": run.get("validation_prevalence"),
        "area_caps": args.area_caps,
        "raw": _summarize_scores(
            name="raw",
            scores=scores,
            labels=labels,
            thresholds=thresholds,
            sample_meta=sample_meta,
            random_baseline_repeats=args.random_baseline_repeats,
            random_baseline_seed_base=args.random_baseline_seed,
            run_seed=seed,
        ),
    }
    (seed_dir / "fullscene_threshold_sanity.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampled full-scene threshold sanity checks for baseline runs")
    parser.add_argument("--runs_json", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sensor", type=str, default="hyperion")
    parser.add_argument(
        "--max_patches",
        type=int,
        default=96,
        help="Uniform full-scene patch sample size. Baseline full-scene evaluation is sampled-only.",
    )
    parser.add_argument(
        "--split_repeat_summary_json",
        type=Path,
        default=None,
        help="Optional split repeat summary JSON used to add train/validation prevalence targets by seed.",
    )
    parser.add_argument(
        "--area_cap",
        type=float,
        default=0.10,
        help="Conservative full-scene predicted-positive-rate cap used as an operating policy.",
    )
    parser.add_argument(
        "--area_caps",
        type=str,
        default=None,
        help="Comma-separated conservative full-scene positive-rate caps, e.g. 0.05,0.10,0.15,0.20.",
    )
    parser.add_argument("--random_baseline_repeats", type=int, default=0)
    parser.add_argument("--random_baseline_seed", type=int, default=20260426)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_patches <= 0:
        raise ValueError("Baseline full-scene evaluation is sampled-only; pass --max_patches > 0")
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
    rows = []
    for run in runs:
        rows.append(run_one(args, run, cube, y_true, valid))
    _write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
