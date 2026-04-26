from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.evaluation.evaluate_fullscene_postprocessing import (
    DEFAULT_CONFIDENCE,
    DEFAULT_VALID_MASK,
    _label_modes,
    _metrics_for_prediction,
    _read_raster,
    _remove_small_components,
    _render_overlay,
)


def _parse_area_caps(value: str) -> list[float]:
    caps: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        caps.append(float(np.clip(float(item), 0.0, 1.0)))
    return sorted(set(caps))


def _policy_name(area_cap: float) -> str:
    return f"area_cap_{int(round(area_cap * 100)):02d}"


def _positive_rate_threshold(scores: np.ndarray, target_rate: float) -> float:
    finite_scores = np.asarray(scores, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return 1.0
    target = float(np.clip(target_rate, 0.0, 1.0))
    if target <= 0.0:
        return float(np.nextafter(np.max(finite_scores), np.inf))
    if target >= 1.0:
        return float(np.min(finite_scores))

    sorted_scores = np.sort(finite_scores)[::-1]
    kth = max(1, int(np.ceil(target * sorted_scores.size))) - 1
    kth_score = float(sorted_scores[min(kth, sorted_scores.size - 1)])
    predicted_rate = float(np.mean(finite_scores >= kth_score))
    if predicted_rate <= target + 1e-12:
        return kth_score
    return float(np.nextafter(kth_score, np.inf))


def _row_with_policy(row: dict[str, Any], *, policy: str, threshold: float, area_cap: float) -> dict[str, Any]:
    out = {
        "policy": policy,
        "area_cap": area_cap,
        "threshold": threshold,
    }
    out.update(row)
    return out


def _write_summary_md(rows: list[dict[str, Any]], output_md: Path, min_component_size: int) -> None:
    lines = [
        "# Full-scene area-cap comparison from saved maps",
        "",
        "## Scope",
        "",
        "- Reuses saved full-scene probability maps.",
        "- Compares `area_cap_10`, `area_cap_15`, and `area_cap_20` without rerunning HyperSIGMA inference.",
        f"- Post-processing variant of interest: `min_component_{min_component_size}`.",
        "- Label modes: `strict_tier1_2` and `weak_inclusive_tier1_2_3`.",
        "",
        "## `min_component` comparison",
        "",
        "| seed | label_mode | policy | precision | recall | F1 | selected_rate_search | tier3 selected |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    target_variant = f"min_component_{min_component_size}"
    target_rows = [row for row in rows if row["variant"] == target_variant]
    for row in sorted(target_rows, key=lambda item: (int(item["seed"]), item["label_mode"], item["area_cap"])):
        lines.append(
            "| {seed} | {label_mode} | `{policy}` | {precision:.4f} | {recall:.4f} | {f1:.4f} | {rate:.4f} | {tier3} |".format(
                seed=row["seed"],
                label_mode=row["label_mode"],
                policy=row["policy"],
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                rate=float(row["selected_rate_search"]),
                tier3=int(row["tier_3_weak_selected"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best precision by seed and label mode",
            "",
            "| seed | label_mode | best policy | precision | recall | F1 | selected_rate_search |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in target_rows:
        grouped.setdefault((str(row["seed"]), str(row["label_mode"])), []).append(row)
    for (seed, label_mode), group in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        best = max(group, key=lambda row: (float(row["precision"]), float(row["f1"])))
        lines.append(
            "| {seed} | {label_mode} | `{policy}` | {precision:.4f} | {recall:.4f} | {f1:.4f} | {rate:.4f} |".format(
                seed=seed,
                label_mode=label_mode,
                policy=best["policy"],
                precision=float(best["precision"]),
                recall=float(best["recall"]),
                f1=float(best["f1"]),
                rate=float(best["selected_rate_search"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- If lower area caps lift strict precision substantially, keep the conservative cap as the current operating baseline.",
            "- If strict precision remains low even at `area_cap_10`, the bottleneck is not just over-selection; move to confidence-aware learning.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    confidence = _read_raster(args.confidence_tif)
    valid = _read_raster(args.valid_mask_tif) == 1
    label_modes = _label_modes(confidence, valid)
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]
    area_caps = _parse_area_caps(args.area_caps)
    target_variant = f"min_component_{args.min_component_size}"

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        map_path = args.map_dir / f"seed{seed}" / f"{args.mode}_probability_map.npy"
        scores = np.load(map_path).astype(np.float32, copy=False)
        if scores.shape != confidence.shape:
            raise ValueError(f"map/label shape mismatch for seed{seed}: {scores.shape} vs {confidence.shape}")
        valid_scores = scores[valid]
        for area_cap in area_caps:
            policy = _policy_name(area_cap)
            threshold = _positive_rate_threshold(valid_scores, area_cap)
            raw_pred = (scores >= threshold) & valid
            variants = {
                "raw": raw_pred,
                target_variant: _remove_small_components(raw_pred, args.min_component_size),
            }
            for variant_name, pred in variants.items():
                if args.render_overlays:
                    _render_overlay(
                        confidence=confidence,
                        valid=valid,
                        pred=pred,
                        output_png=args.output_dir / "overlays" / f"seed{seed}_{policy}_{variant_name}.png",
                        title=f"seed{seed} {args.mode} {policy} {variant_name}",
                        downsample=args.downsample,
                    )
                for label_mode_name, mode_obj in label_modes.items():
                    row = _metrics_for_prediction(
                        seed=seed,
                        variant=variant_name,
                        label_mode=label_mode_name,
                        pred=pred,
                        confidence=confidence,
                        wet=mode_obj["wet"],
                        eval_valid=mode_obj["eval_valid"],
                        search_valid=valid,
                    )
                    rows.append(
                        _row_with_policy(
                            row,
                            policy=policy,
                            threshold=threshold,
                            area_cap=area_cap,
                        )
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fullscene_area_cap_from_maps_summary.csv"
    json_path = args.output_dir / "fullscene_area_cap_from_maps_summary.json"
    md_path = args.output_dir / "fullscene_area_cap_from_maps_summary.md"
    if not rows:
        raise RuntimeError("No area-cap rows were produced")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    _write_summary_md(rows, md_path, args.min_component_size)
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {json_path}")
    print(f"[INFO] wrote {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare full-scene area caps from saved probability maps")
    parser.add_argument("--map_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--mode", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--seeds", type=str, default="13,99")
    parser.add_argument("--area_caps", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--min_component_size", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--render_overlays", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
