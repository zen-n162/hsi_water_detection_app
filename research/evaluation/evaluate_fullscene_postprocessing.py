from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage


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


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds_by_seed(summary: dict[str, Any], *, mode: str, policy: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for run in summary.get("runs", []):
        seed = str(run["seed"])
        thresholds[seed] = float(run[mode]["thresholds"][policy]["threshold"])
    return thresholds


def _downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def _downsample_max(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    if h == 0 or w == 0:
        return arr
    cropped = arr[:h, :w]
    return cropped.reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def _remove_small_components(pred: np.ndarray, min_size: int) -> np.ndarray:
    labels, n_labels = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels <= 0:
        return pred.copy()
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def _remove_thin_isolated_components(
    pred: np.ndarray,
    *,
    max_aspect_ratio: float,
    max_area: int,
) -> np.ndarray:
    labels, n_labels = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels <= 0:
        return pred.copy()
    keep = np.ones(n_labels + 1, dtype=bool)
    keep[0] = False
    sizes = np.bincount(labels.ravel())
    objects = ndimage.find_objects(labels)
    for idx, obj in enumerate(objects, start=1):
        if obj is None:
            keep[idx] = False
            continue
        row_slice, col_slice = obj
        height = int(row_slice.stop - row_slice.start)
        width = int(col_slice.stop - col_slice.start)
        short = max(1, min(height, width))
        long = max(height, width)
        aspect_ratio = float(long / short)
        if int(sizes[idx]) <= int(max_area) and aspect_ratio >= float(max_aspect_ratio):
            keep[idx] = False
    return keep[labels]


def _local_percentile_filter(
    *,
    pred: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    window: int,
) -> np.ndarray:
    score_for_filter = np.where(valid, scores, 0.0).astype(np.float32, copy=False)
    local = ndimage.percentile_filter(
        score_for_filter,
        percentile=float(percentile),
        size=int(window),
        mode="nearest",
    )
    return pred & (scores >= local) & valid


def _erode_valid(pred: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(valid, structure=structure, iterations=int(radius), border_value=0)
    return pred & eroded


def _postprocess_variants(pred: np.ndarray, scores: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    min_component_256 = _remove_small_components(pred, 256)
    local_p90_w31 = _local_percentile_filter(
        pred=pred,
        scores=scores,
        valid=valid,
        percentile=90,
        window=31,
    )
    local_p95_w31 = _local_percentile_filter(
        pred=pred,
        scores=scores,
        valid=valid,
        percentile=95,
        window=31,
    )
    erode_r1 = _erode_valid(pred, valid, 1)
    variants = {
        "raw_area_cap20": pred,
        "valid_erode_r1": erode_r1,
        "valid_erode_r2": _erode_valid(pred, valid, 2),
        "min_component_64": _remove_small_components(pred, 64),
        "min_component_256": min_component_256,
        "min_component_1024": _remove_small_components(pred, 1024),
        "remove_thin_ar8_area4096": _remove_thin_isolated_components(
            pred,
            max_aspect_ratio=8.0,
            max_area=4096,
        ),
        "local_p90_w31": local_p90_w31,
        "local_p95_w31": local_p95_w31,
        "local_p90_w31_min_component_256": _remove_small_components(local_p90_w31, 256),
        "local_p95_w31_min_component_256": _remove_small_components(local_p95_w31, 256),
        "valid_erode_r1_min_component_256": _remove_small_components(erode_r1, 256),
        "min_component_256_remove_thin": _remove_thin_isolated_components(
            min_component_256,
            max_aspect_ratio=8.0,
            max_area=4096,
        ),
    }
    return variants


def _label_modes(confidence: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, np.ndarray | str]]:
    return {
        "strict_tier1_2": {
            "description": "tier1+tier2をwet、tier3は曖昧として評価から除外",
            "eval_valid": valid & np.isin(confidence, [0, 1, 2]),
            "wet": np.isin(confidence, [1, 2]),
        },
        "weak_inclusive_tier1_2_3": {
            "description": "tier1+tier2+tier3をwetとして評価",
            "eval_valid": valid & np.isin(confidence, [0, 1, 2, 3]),
            "wet": np.isin(confidence, [1, 2, 3]),
        },
    }


def _tier_counts(confidence_values: np.ndarray) -> dict[str, int]:
    return {
        "tier_0_dry_selected": int(np.sum(confidence_values == 0)),
        "tier_1_strong_selected": int(np.sum(confidence_values == 1)),
        "tier_2_medium_selected": int(np.sum(confidence_values == 2)),
        "tier_3_weak_selected": int(np.sum(confidence_values == 3)),
    }


def _metrics_for_prediction(
    *,
    seed: str,
    variant: str,
    label_mode: str,
    pred: np.ndarray,
    confidence: np.ndarray,
    wet: np.ndarray,
    eval_valid: np.ndarray,
    search_valid: np.ndarray,
) -> dict[str, Any]:
    pred_eval = pred & eval_valid
    wet_eval = wet & eval_valid
    tp = int(np.sum(pred_eval & wet_eval))
    fp = int(np.sum(pred_eval & ~wet_eval))
    fn = int(np.sum(~pred_eval & wet_eval))
    tn = int(np.sum(~pred_eval & ~wet_eval & eval_valid))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    selected_confidence = confidence[pred & search_valid]
    return {
        "seed": seed,
        "variant": variant,
        "label_mode": label_mode,
        "search_valid_pixels": int(search_valid.sum()),
        "eval_valid_pixels": int(eval_valid.sum()),
        "eval_wet_pixels": int(wet_eval.sum()),
        "selected_pixels_search": int(np.sum(pred & search_valid)),
        "selected_rate_search": float(np.mean(pred[search_valid])) if np.any(search_valid) else 0.0,
        "selected_pixels_eval": int(pred_eval.sum()),
        "selected_rate_eval": float(np.mean(pred_eval[eval_valid])) if np.any(eval_valid) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **_tier_counts(selected_confidence),
    }


def _render_overlay(
    *,
    confidence: np.ndarray,
    valid: np.ndarray,
    pred: np.ndarray,
    output_png: Path,
    title: str,
    downsample: int,
) -> None:
    base = np.full(confidence.shape, 4, dtype=np.uint8)
    base[valid & (confidence == 0)] = 0
    base[valid & (confidence == 1)] = 1
    base[valid & (confidence == 2)] = 2
    base[valid & (confidence == 3)] = 3

    base_ds = _downsample_nearest(base, downsample)
    pred_ds = _downsample_max((pred & valid).astype(np.uint8), downsample)

    cmap = ListedColormap(["#1f2933", "#0f766e", "#38bdf8", "#facc15", "#000000"])
    plt.figure(figsize=(9, 8))
    plt.imshow(base_ds, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    pred_overlay = np.ma.masked_where(pred_ds == 0, pred_ds)
    plt.imshow(pred_overlay, cmap=ListedColormap(["#ff2d55"]), alpha=0.78, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def _write_summary_md(rows: list[dict[str, Any]], output_md: Path) -> None:
    lines = [
        "# area_cap_20 full-scene post-processing diagnostics",
        "",
        "## Label modes",
        "",
        "- `strict_tier1_2`: tier1+tier2をwet、tier3は評価から除外。",
        "- `weak_inclusive_tier1_2_3`: tier1+tier2+tier3をwetとして評価。",
        "",
        "## Best precision by seed and label mode",
        "",
        "| seed | label_mode | raw precision | raw recall | raw F1 | best variant | precision | recall | F1 | selected_rate_search | tier3 selected |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["seed"]), str(row["label_mode"])), []).append(row)

    for (seed, label_mode), group in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        raw = next(row for row in group if row["variant"] == "raw_area_cap20")
        best = max(group, key=lambda row: (float(row["precision"]), float(row["f1"])))
        lines.append(
            "| {seed} | {label_mode} | {raw_precision:.4f} | {raw_recall:.4f} | {raw_f1:.4f} | "
            "`{variant}` | {precision:.4f} | {recall:.4f} | {f1:.4f} | {rate:.4f} | {tier3} |".format(
                seed=seed,
                label_mode=label_mode,
                raw_precision=float(raw["precision"]),
                raw_recall=float(raw["recall"]),
                raw_f1=float(raw["f1"]),
                variant=best["variant"],
                precision=float(best["precision"]),
                recall=float(best["recall"]),
                f1=float(best["f1"]),
                rate=float(best["selected_rate_search"]),
                tier3=int(best["tier_3_weak_selected"]),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ここでのpost-processingは、area_cap_20で選ばれた探索候補からピクセルを削る診断です。面積を再充填していないため、precision改善とrecall低下のトレードオフを見ます。",
            "- `selected_rate_search` は元の探索valid-mask上の選択率です。area_cap_20からどれだけ候補が削られたかを見る指標です。",
            "- overlayはconfidence labelを背景に、選択候補をmagentaで重ねています。探索範囲外はblackです。",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    confidence = _read_raster(args.confidence_tif)
    valid = _read_raster(args.valid_mask_tif) == 1
    summary = _load_summary(args.summary_json)
    thresholds = _thresholds_by_seed(summary, mode=args.mode, policy=args.policy)
    label_modes = _label_modes(confidence, valid)

    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        seeds = sorted(thresholds, key=int)

    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        threshold = thresholds[seed]
        map_path = args.map_dir / f"seed{seed}" / f"{args.mode}_probability_map.npy"
        scores = np.load(map_path).astype(np.float32, copy=False)
        if scores.shape != confidence.shape:
            raise ValueError(f"map/label shape mismatch for seed{seed}: {scores.shape} vs {confidence.shape}")
        pred = (scores >= threshold) & valid
        variants = _postprocess_variants(pred, scores, valid)
        for variant_name, variant_pred in variants.items():
            if args.render_overlays:
                _render_overlay(
                    confidence=confidence,
                    valid=valid,
                    pred=variant_pred,
                    output_png=args.output_dir / "overlays" / f"seed{seed}_{variant_name}.png",
                    title=f"seed{seed} {args.mode} {args.policy} {variant_name}",
                    downsample=args.downsample,
                )
            for label_mode_name, mode_obj in label_modes.items():
                all_rows.append(
                    _metrics_for_prediction(
                        seed=seed,
                        variant=variant_name,
                        label_mode=label_mode_name,
                        pred=variant_pred,
                        confidence=confidence,
                        wet=mode_obj["wet"],
                        eval_valid=mode_obj["eval_valid"],
                        search_valid=valid,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fullscene_postprocessing_summary.csv"
    json_path = args.output_dir / "fullscene_postprocessing_summary.json"
    md_path = args.output_dir / "fullscene_postprocessing_summary.md"
    if not all_rows:
        raise RuntimeError("No post-processing rows were produced")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps({"rows": all_rows}, indent=2), encoding="utf-8")
    _write_summary_md(all_rows, md_path)
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {json_path}")
    print(f"[INFO] wrote {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate post-processing policies on full-scene probability maps")
    parser.add_argument("--map_dir", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--mode", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--policy", type=str, default="area_cap_20")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--render_overlays", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
