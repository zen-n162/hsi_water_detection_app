from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


DEFAULT_CONFIDENCE = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/labels/confidence_ali.tif"
)
DEFAULT_VALID_MASK = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/masks/final_valid_mask_clean_ali_aoi90.tif"
)


def _read_raster(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def _downsample_max(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    if h == 0 or w == 0:
        return arr
    cropped = arr[:h, :w]
    return cropped.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def _downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_thresholds(summary: dict[str, Any], mode: str) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for run in summary["runs"]:
        seed = str(run["seed"])
        thresholds[seed] = {}
        for policy, metrics in run[mode]["thresholds"].items():
            thresholds[seed][policy] = float(metrics["threshold"])
    return thresholds


def _policy_targets(summary: dict[str, Any], mode: str) -> dict[str, dict[str, float | None]]:
    targets: dict[str, dict[str, float | None]] = {}
    for run in summary["runs"]:
        seed = str(run["seed"])
        targets[seed] = {}
        for policy, metrics in run[mode]["thresholds"].items():
            value = metrics.get("target_positive_rate")
            targets[seed][policy] = None if value is None else float(value)
    return targets


def _array_from_npz(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of {names} found in {data.files}")


def _stable_seed(*parts: str, base_seed: int = 20260426) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**32)


def _exact_area_cap_prediction(scores: np.ndarray, target_rate: float, *, seed: int) -> np.ndarray:
    n = int(scores.size)
    k = int(round(float(np.clip(target_rate, 0.0, 1.0)) * n))
    k = max(0, min(k, n))
    pred = np.zeros(n, dtype=bool)
    if k <= 0:
        return pred
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(n)
    order = np.lexsort((tie_breaker, -np.asarray(scores, dtype=np.float64)))
    pred[order[:k]] = True
    return pred


def _tier_counts(confidence_values: np.ndarray) -> dict[str, int]:
    return {
        "tier_0_dry": int(np.sum(confidence_values == 0)),
        "tier_1_strong": int(np.sum(confidence_values == 1)),
        "tier_2_medium": int(np.sum(confidence_values == 2)),
        "tier_3_weak": int(np.sum(confidence_values == 3)),
    }


def _render_overlay(
    *,
    confidence: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    pred: np.ndarray,
    output_png: Path,
    title: str,
    downsample: int,
) -> None:
    base = np.full(confidence.shape, 4, dtype=np.uint8)
    base[(valid) & (confidence == 0)] = 0
    base[(valid) & (confidence == 1)] = 1
    base[(valid) & (confidence == 2)] = 2
    base[(valid) & (confidence == 3)] = 3

    sampled_mask = np.zeros(confidence.shape, dtype=np.uint8)
    pred_mask = np.zeros(confidence.shape, dtype=np.uint8)
    sampled_mask[rows, cols] = 1
    pred_mask[rows[pred], cols[pred]] = 1

    base_ds = _downsample_nearest(base, downsample)
    sampled_ds = _downsample_max(sampled_mask, downsample)
    pred_ds = _downsample_max(pred_mask, downsample)

    cmap = ListedColormap(["#1f2933", "#0f766e", "#38bdf8", "#facc15", "#cbd5e1"])
    plt.figure(figsize=(9, 8))
    plt.imshow(base_ds, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    sampled_overlay = np.ma.masked_where(sampled_ds == 0, sampled_ds)
    pred_overlay = np.ma.masked_where(pred_ds == 0, pred_ds)
    plt.imshow(sampled_overlay, cmap=ListedColormap(["#ffffff"]), alpha=0.12, interpolation="nearest")
    plt.imshow(pred_overlay, cmap=ListedColormap(["#ff2d55"]), alpha=0.75, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    confidence = _read_raster(args.confidence_tif)
    valid = _read_raster(args.valid_mask_tif) == 1
    summary = _load_summary(args.summary_json)
    thresholds = _policy_thresholds(summary, args.mode)
    targets = _policy_targets(summary, args.mode)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    rows_out: list[dict[str, Any]] = []
    for seed, seed_thresholds in sorted(thresholds.items(), key=lambda item: int(item[0])):
        npz_path = args.policy_dir / f"seed{seed}" / (
            "sampled_scores_calibrated.npz" if args.mode == "calibrated" else "sampled_scores.npz"
        )
        data = np.load(npz_path)
        scores = (
            _array_from_npz(data, ("calibrated_scores", "scores"))
            if args.mode == "calibrated"
            else _array_from_npz(data, ("raw_scores", "scores"))
        )
        labels = (
            _array_from_npz(data, ("calibrated_labels", "labels"))
            if args.mode == "calibrated"
            else _array_from_npz(data, ("raw_labels", "labels"))
        )
        rows = data["rows"].astype(np.int64)
        cols = data["cols"].astype(np.int64)
        confidence_values = confidence[rows, cols]
        for policy in policies:
            if policy not in seed_thresholds:
                continue
            threshold = seed_thresholds[policy]
            target_rate = targets.get(seed, {}).get(policy)
            if args.selection_mode == "exact_area_cap" and target_rate is not None:
                pred = _exact_area_cap_prediction(
                    scores,
                    target_rate,
                    seed=_stable_seed(seed, args.mode, policy),
                )
            else:
                pred = scores >= threshold
            pred_confidence = confidence_values[pred]
            counts = _tier_counts(pred_confidence)
            selected = int(pred.sum())
            wet_selected = int(labels[pred].sum()) if selected else 0
            precision = float(wet_selected / selected) if selected else 0.0
            recall = float(wet_selected / labels.sum()) if labels.sum() else 0.0
            rows_out.append(
                {
                    "seed": seed,
                    "mode": args.mode,
                    "selection_mode": args.selection_mode,
                    "policy": policy,
                    "threshold": threshold,
                    "target_positive_rate": target_rate,
                    "sampled_pixels": int(labels.size),
                    "sampled_wet_pixels": int(labels.sum()),
                    "predicted_positive_pixels": selected,
                    "predicted_positive_rate": float(pred.mean()),
                    "precision_on_sample": precision,
                    "recall_on_sample": recall,
                    **counts,
                }
            )
            output_png = args.output_dir / f"seed{seed}" / f"{args.mode}_{policy}_overlay.png"
            _render_overlay(
                confidence=confidence,
                valid=valid,
                rows=rows,
                cols=cols,
                pred=pred,
                output_png=output_png,
                title=f"seed{seed} {args.mode} {policy}",
                downsample=args.downsample,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fullscene_policy_overlay_summary.csv"
    json_path = args.output_dir / "fullscene_policy_overlay_summary.json"
    if rows_out:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows_out)
    json_path.write_text(json.dumps({"rows": rows_out}, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render sampled full-scene threshold policy overlays")
    parser.add_argument("--policy_dir", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--mode", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--selection_mode", choices=["threshold", "exact_area_cap"], default="threshold")
    parser.add_argument("--policies", type=str, default="area_cap_05,area_cap_10,area_cap_15,area_cap_20")
    parser.add_argument("--downsample", type=int, default=8)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
