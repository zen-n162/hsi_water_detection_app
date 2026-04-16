from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


DEFAULT_LABEL_RASTER = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/labels/confidence_ali.tif"
)
DEFAULT_INPUT_RASTER = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/hyperion/hyperion_stack_crop_f16.tif"
)
DEFAULT_OUTPUT_MANIFEST = Path(
    "annotations/manifests/wetness_manifest_from_confidence_ali.csv"
)

SPLIT_POLICIES = ("random_stratified", "spatial_block")

MANIFEST_HEADER = [
    "sample_id",
    "scene_id",
    "input_path",
    "sensor",
    "row_start",
    "row_stop",
    "col_start",
    "col_stop",
    "label",
    "confidence",
    "source",
    "split",
    "notes",
    "confidence_tier",
    "valid_ratio",
    "wet_ratio_valid",
    "valid_pixels",
    "dry_pixels",
    "wet_pixels",
    "ignore_pixels",
    "tier1_pixels",
    "tier2_pixels",
    "tier3_pixels",
    "block_row",
    "block_col",
    "block_id",
    "split_policy",
]


def _transform_tuple(transform: Any) -> tuple[float, float, float, float, float, float]:
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def _assert_same_grid(label_path: Path, input_path: Path) -> None:
    with rasterio.open(label_path) as label_ds, rasterio.open(input_path) as input_ds:
        problems: list[str] = []

        if (label_ds.height, label_ds.width) != (input_ds.height, input_ds.width):
            problems.append(
                f"shape mismatch label={(label_ds.height, label_ds.width)} "
                f"input={(input_ds.height, input_ds.width)}"
            )

        if str(label_ds.crs) != str(input_ds.crs):
            problems.append(
                f"crs mismatch label={label_ds.crs!s} input={input_ds.crs!s}"
            )

        label_transform = _transform_tuple(label_ds.transform)
        input_transform = _transform_tuple(input_ds.transform)
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(label_transform, input_transform)):
            problems.append(
                f"transform mismatch label={label_transform} input={input_transform}"
            )

        if problems:
            raise RuntimeError(
                "confidence_ali manifest generation requires label raster and input raster "
                "to share the same grid.\n" + "\n".join(problems)
            )


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _dominant_wet_tier(tier_counts: dict[int, int]) -> int:
    return max(tier_counts, key=lambda tier: (tier_counts[tier], tier))


def _build_block_fields(row_start: int, col_start: int, block_size: int) -> tuple[int, int, str]:
    block_row = row_start // block_size
    block_col = col_start // block_size
    block_id = f"block_r{block_row:03d}_c{block_col:03d}"
    return block_row, block_col, block_id


def _compute_patch_record(
    *,
    patch_arr: np.ndarray,
    row_start: int,
    col_start: int,
    patch_size: int,
    block_size: int,
    scene_id: str,
    input_path: Path,
    min_valid_ratio: float,
    min_wet_ratio: float,
    split_policy: str,
) -> dict[str, Any] | None:
    valid_mask = patch_arr != 255
    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        return None

    total_pixels = int(patch_arr.size)
    valid_ratio = valid_pixels / total_pixels
    if valid_ratio < min_valid_ratio:
        return None

    tier_counts = {tier: int((patch_arr == tier).sum()) for tier in (1, 2, 3)}
    wet_pixels = int(sum(tier_counts.values()))
    dry_pixels = int((patch_arr == 0).sum())
    ignore_pixels = int((patch_arr == 255).sum())
    wet_ratio_valid = wet_pixels / valid_pixels

    if wet_ratio_valid >= min_wet_ratio:
        label = "wet"
        weighted_tier_sum = sum(tier * count for tier, count in tier_counts.items())
        mean_tier = weighted_tier_sum / max(wet_pixels, 1)
        confidence = max(0.0, min(1.0, valid_ratio * (mean_tier / 3.0)))
        confidence_tier = _dominant_wet_tier(tier_counts)
        notes = (
            "Generated from confidence_ali.tif using label=wet when "
            f"wet_ratio_valid >= {min_wet_ratio:.3f}"
        )
    else:
        label = "dry"
        confidence = max(0.0, min(1.0, valid_ratio * (1.0 - wet_ratio_valid)))
        confidence_tier = 0
        notes = (
            "Generated from confidence_ali.tif using label=dry when "
            f"wet_ratio_valid < {min_wet_ratio:.3f}"
        )

    block_row, block_col, block_id = _build_block_fields(
        row_start=row_start,
        col_start=col_start,
        block_size=block_size,
    )

    return {
        "scene_id": scene_id,
        "input_path": str(input_path),
        "sensor": "hyperion",
        "row_start": row_start,
        "row_stop": row_start + patch_size,
        "col_start": col_start,
        "col_stop": col_start + patch_size,
        "label": label,
        "confidence": _format_float(confidence),
        "source": "confidence_ali.tif",
        "split": "",
        "notes": notes,
        "confidence_tier": str(confidence_tier),
        "valid_ratio": _format_float(valid_ratio),
        "wet_ratio_valid": _format_float(wet_ratio_valid),
        "valid_pixels": str(valid_pixels),
        "dry_pixels": str(dry_pixels),
        "wet_pixels": str(wet_pixels),
        "ignore_pixels": str(ignore_pixels),
        "tier1_pixels": str(tier_counts[1]),
        "tier2_pixels": str(tier_counts[2]),
        "tier3_pixels": str(tier_counts[3]),
        "block_row": str(block_row),
        "block_col": str(block_col),
        "block_id": block_id,
        "split_policy": split_policy,
    }


def _build_records(
    *,
    label_path: Path,
    input_path: Path,
    patch_size: int,
    stride: int,
    block_size: int,
    min_valid_ratio: float,
    min_wet_ratio: float,
    split_policy: str,
) -> list[dict[str, Any]]:
    with rasterio.open(label_path) as src:
        arr = src.read(1)

    scene_id = input_path.stem
    records: list[dict[str, Any]] = []

    for row_start in range(0, arr.shape[0] - patch_size + 1, stride):
        for col_start in range(0, arr.shape[1] - patch_size + 1, stride):
            patch_arr = arr[row_start:row_start + patch_size, col_start:col_start + patch_size]
            record = _compute_patch_record(
                patch_arr=patch_arr,
                row_start=row_start,
                col_start=col_start,
                patch_size=patch_size,
                block_size=block_size,
                scene_id=scene_id,
                input_path=input_path,
                min_valid_ratio=min_valid_ratio,
                min_wet_ratio=min_wet_ratio,
                split_policy=split_policy,
            )
            if record is not None:
                records.append(record)

    for index, record in enumerate(records, start=1):
        record["sample_id"] = f"confali_{index:05d}"

    return records


def _allocate_split_counts(n_items: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    n_test = int(round(n_items * test_ratio))
    n_val = int(round(n_items * val_ratio))

    if n_items >= 3:
        if test_ratio > 0.0 and n_test == 0:
            n_test = 1
        if val_ratio > 0.0 and n_val == 0:
            n_val = 1

    if n_test + n_val >= n_items:
        overflow = n_test + n_val - (n_items - 1)
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1

    n_train = n_items - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(
            f"split allocation failed: n_items={n_items}, n_train={n_train}, "
            f"n_val={n_val}, n_test={n_test}"
        )

    return n_train, n_val, n_test


def _assign_splits_random_stratified(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(record)

    rng = random.Random(seed)
    for label_records in by_label.values():
        rng.shuffle(label_records)
        n_train, n_val, n_test = _allocate_split_counts(
            len(label_records),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        for index, record in enumerate(label_records):
            if index < n_train:
                record["split"] = "train"
            elif index < n_train + n_val:
                record["split"] = "val"
            else:
                record["split"] = "test"


def _block_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_block: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "wet": 0, "dry": 0})
    for record in records:
        block_id = str(record["block_id"])
        by_block[block_id]["total"] += 1
        by_block[block_id][str(record["label"])] += 1
    return dict(by_block)


def _objective_for_assignment(
    *,
    split_stats: dict[str, dict[str, int]],
    target_stats: dict[str, dict[str, float]],
) -> float:
    score = 0.0
    for split_name, stats in split_stats.items():
        targets = target_stats[split_name]
        score += 3.0 * (float(stats["total"]) - float(targets["total"])) ** 2
        score += 2.0 * (float(stats["wet"]) - float(targets["wet"])) ** 2
        score += 1.0 * (float(stats["dry"]) - float(targets["dry"])) ** 2
        if stats["total"] == 0:
            score += 1e6
        if targets["wet"] > 0 and stats["wet"] == 0:
            score += 5e4
    return score


def _assign_blocks_spatially(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    assignment_restarts: int,
) -> None:
    split_ratio = {
        "train": 1.0 - val_ratio - test_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    block_stats = _block_counts(records)
    if len(block_stats) < 3:
        raise RuntimeError(
            f"spatial_block split needs at least 3 populated blocks, got {len(block_stats)}"
        )

    total_records = len(records)
    total_wet = sum(1 for record in records if str(record["label"]) == "wet")
    total_dry = sum(1 for record in records if str(record["label"]) == "dry")

    target_stats = {
        split_name: {
            "total": total_records * ratio,
            "wet": total_wet * ratio,
            "dry": total_dry * ratio,
        }
        for split_name, ratio in split_ratio.items()
    }

    block_ids = list(block_stats.keys())
    base_rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    for restart in range(max(assignment_restarts, 1)):
        rng = random.Random(base_rng.randint(0, 10**9) + restart)
        ordered_blocks = sorted(
            block_ids,
            key=lambda block_id: (
                -block_stats[block_id]["total"],
                -abs(block_stats[block_id]["wet"] - block_stats[block_id]["dry"]),
                rng.random(),
            ),
        )

        split_stats = {
            "train": {"total": 0, "wet": 0, "dry": 0},
            "val": {"total": 0, "wet": 0, "dry": 0},
            "test": {"total": 0, "wet": 0, "dry": 0},
        }
        assignment: dict[str, str] = {}
        split_order = ["train", "val", "test"]

        # Seed each split with one block so block-level split does not collapse.
        seeded_blocks = ordered_blocks[:3]
        for split_name, block_id in zip(split_order, seeded_blocks):
            assignment[block_id] = split_name
            for key in ("total", "wet", "dry"):
                split_stats[split_name][key] += block_stats[block_id][key]

        for block_id in ordered_blocks[3:]:
            best_local_split = None
            best_local_score = float("inf")
            for split_name in split_order:
                candidate_stats = {
                    name: stats.copy()
                    for name, stats in split_stats.items()
                }
                for key in ("total", "wet", "dry"):
                    candidate_stats[split_name][key] += block_stats[block_id][key]

                candidate_score = _objective_for_assignment(
                    split_stats=candidate_stats,
                    target_stats=target_stats,
                )
                if candidate_score < best_local_score:
                    best_local_score = candidate_score
                    best_local_split = split_name

            assert best_local_split is not None
            assignment[block_id] = best_local_split
            for key in ("total", "wet", "dry"):
                split_stats[best_local_split][key] += block_stats[block_id][key]

        final_score = _objective_for_assignment(
            split_stats=split_stats,
            target_stats=target_stats,
        )
        if final_score < best_score:
            best_score = final_score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Failed to compute spatial block split assignment")

    for record in records:
        record["split"] = best_assignment[str(record["block_id"])]


def _assign_splits(
    records: list[dict[str, Any]],
    *,
    split_policy: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    assignment_restarts: int,
) -> None:
    if split_policy == "random_stratified":
        _assign_splits_random_stratified(
            records,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        return

    if split_policy == "spatial_block":
        _assign_blocks_spatially(
            records,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            assignment_restarts=assignment_restarts,
        )
        return

    raise ValueError(f"Unsupported split_policy: {split_policy}")


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    label_path: Path,
    input_path: Path,
    patch_size: int,
    stride: int,
    block_size: int,
    min_valid_ratio: float,
    min_wet_ratio: float,
    split_policy: str,
    seed: int,
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    label_counts = Counter(record["label"] for record in records)
    confidence_tier_counts = Counter(record["confidence_tier"] for record in records)
    split_label_counts = Counter(
        f"{record['split']}::{record['label']}" for record in records
    )
    block_counts = _block_counts(records)
    split_block_counts = Counter(
        f"{record['split']}::{record['block_id']}" for record in records
    )
    blocks_per_split = Counter(
        record["split"] for record in {
            (str(record["split"]), str(record["block_id"])): record
            for record in records
        }.values()
    )

    return {
        "label_raster": str(label_path),
        "input_raster": str(input_path),
        "patch_size": patch_size,
        "stride": stride,
        "block_size": block_size,
        "min_valid_ratio": min_valid_ratio,
        "min_wet_ratio": min_wet_ratio,
        "split_policy": split_policy,
        "split_seed": seed,
        "n_records": len(records),
        "n_blocks": len(block_counts),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "confidence_tier_counts": dict(confidence_tier_counts),
        "split_label_counts": dict(split_label_counts),
        "blocks_per_split": dict(blocks_per_split),
        "block_patch_counts": block_counts,
        "split_block_counts": dict(split_block_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a patch-level wetness manifest from confidence_ali.tif. "
            "Supports the original random stratified split and the v3 spatial block split."
        )
    )
    parser.add_argument("--label_raster", type=Path, default=DEFAULT_LABEL_RASTER)
    parser.add_argument("--input_raster", type=Path, default=DEFAULT_INPUT_RASTER)
    parser.add_argument("--output_manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output_summary_json", type=Path, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--split_policy", type=str, default="random_stratified", choices=SPLIT_POLICIES)
    parser.add_argument("--assignment_restarts", type=int, default=200)
    parser.add_argument("--min_valid_ratio", type=float, default=0.90)
    parser.add_argument("--min_wet_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.label_raster.exists():
        raise FileNotFoundError(f"label_raster not found: {args.label_raster}")
    if not args.input_raster.exists():
        raise FileNotFoundError(f"input_raster not found: {args.input_raster}")

    if args.patch_size <= 0 or args.stride <= 0 or args.block_size <= 0:
        raise ValueError("patch_size, stride, and block_size must be positive")
    if not (0.0 <= args.min_valid_ratio <= 1.0):
        raise ValueError("min_valid_ratio must be in [0, 1]")
    if not (0.0 <= args.min_wet_ratio <= 1.0):
        raise ValueError("min_wet_ratio must be in [0, 1]")
    if args.val_ratio < 0.0 or args.test_ratio < 0.0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    _assert_same_grid(args.label_raster, args.input_raster)

    records = _build_records(
        label_path=args.label_raster,
        input_path=args.input_raster,
        patch_size=args.patch_size,
        stride=args.stride,
        block_size=args.block_size,
        min_valid_ratio=args.min_valid_ratio,
        min_wet_ratio=args.min_wet_ratio,
        split_policy=args.split_policy,
    )
    if not records:
        raise RuntimeError("No patches survived the current manifest generation rules")

    _assign_splits(
        records,
        split_policy=args.split_policy,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        assignment_restarts=args.assignment_restarts,
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(records)

    summary = _summarize_records(
        records,
        label_path=args.label_raster,
        input_path=args.input_raster,
        patch_size=args.patch_size,
        stride=args.stride,
        block_size=args.block_size,
        min_valid_ratio=args.min_valid_ratio,
        min_wet_ratio=args.min_wet_ratio,
        split_policy=args.split_policy,
        seed=args.seed,
    )

    summary_path = args.output_summary_json or args.output_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] manifest saved to {args.output_manifest}")
    print(f"[INFO] summary saved to {summary_path}")
    print(f"[INFO] split_policy={summary['split_policy']}")
    print(f"[INFO] n_records={summary['n_records']} n_blocks={summary['n_blocks']}")
    print(f"[INFO] split_counts={summary['split_counts']}")
    print(f"[INFO] label_counts={summary['label_counts']}")
    print(f"[INFO] blocks_per_split={summary['blocks_per_split']}")


if __name__ == "__main__":
    main()
