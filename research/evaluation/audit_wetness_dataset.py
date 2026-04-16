from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from research.training.dataloaders import WetnessPatchDataset


DEFAULT_INPUT_DIR = Path("datasets/processed/wetness_pretrain_v2")
DEFAULT_MANIFEST_PATH = Path("annotations/manifests/wetness_manifest_from_confidence_ali.csv")
DEFAULT_OUTPUT_JSON = Path("experiments/eval_v2/dataset_runtime_audit.json")


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _label_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        label_name = str(record["label_raw"])
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def _confidence_tier_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        tier = str(int(float(item["confidence_tier"])))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _unique_block_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    per_split: dict[str, set[str]] = {"all": set(), "train": set(), "val": set(), "test": set()}
    for idx in range(len(dataset)):
        item = dataset[idx]
        block_id = str(item["block_id"])
        split = str(item["split"])
        per_split["all"].add(block_id)
        if split in per_split:
            per_split[split].add(block_id)
    return {k: len(v) for k, v in per_split.items()}


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _spatial_leakage_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    audits: dict[str, Any] = {}

    for i, left in enumerate(rows):
        left_split = str(left["split"])
        left_box = (
            int(left["row_start"]),
            int(left["row_stop"]),
            int(left["col_start"]),
            int(left["col_stop"]),
        )
        for right in rows[i + 1:]:
            right_split = str(right["split"])
            if left_split == right_split:
                continue

            right_box = (
                int(right["row_start"]),
                int(right["row_stop"]),
                int(right["col_start"]),
                int(right["col_stop"]),
            )
            row_gap = _interval_gap(left_box[0], left_box[1], right_box[0], right_box[1])
            col_gap = _interval_gap(left_box[2], left_box[3], right_box[2], right_box[3])
            euclidean_gap = math.sqrt(float(row_gap ** 2 + col_gap ** 2))

            pair_key = "::".join(sorted([left_split, right_split]))
            pair = audits.setdefault(
                pair_key,
                {
                    "min_row_gap": None,
                    "min_col_gap": None,
                    "min_euclidean_gap": None,
                    "touching_or_overlapping_pairs": 0,
                    "closest_examples": [],
                },
            )

            pair["min_row_gap"] = row_gap if pair["min_row_gap"] is None else min(pair["min_row_gap"], row_gap)
            pair["min_col_gap"] = col_gap if pair["min_col_gap"] is None else min(pair["min_col_gap"], col_gap)
            pair["min_euclidean_gap"] = (
                euclidean_gap
                if pair["min_euclidean_gap"] is None
                else min(float(pair["min_euclidean_gap"]), euclidean_gap)
            )
            if row_gap == 0 and col_gap == 0:
                pair["touching_or_overlapping_pairs"] += 1

            candidate = {
                "left_sample_id": left["sample_id"],
                "left_split": left_split,
                "left_block_id": left.get("block_id", ""),
                "right_sample_id": right["sample_id"],
                "right_split": right_split,
                "right_block_id": right.get("block_id", ""),
                "row_gap": row_gap,
                "col_gap": col_gap,
                "euclidean_gap": euclidean_gap,
            }
            pair["closest_examples"].append(candidate)
            pair["closest_examples"] = sorted(
                pair["closest_examples"],
                key=lambda x: (float(x["euclidean_gap"]), int(x["row_gap"]), int(x["col_gap"])),
            )[:5]

    return audits


def _summary_leakage_counts(audit_obj: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(value.get("touching_or_overlapping_pairs", 0))
        for key, value in audit_obj.items()
    }


def _compare_audits(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(current.keys()) | set(previous.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        curr = int(current.get(key, {}).get("touching_or_overlapping_pairs", 0))
        prev = int(previous.get(key, {}).get("touching_or_overlapping_pairs", 0))
        rows.append(
            {
                "pair": key,
                "current_touching_pairs": curr,
                "previous_touching_pairs": prev,
                "delta_touching_pairs": curr - prev,
            }
        )
    return {
        "pairs": rows,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime validation and simple spatial leakage audit for a wetness patch dataset")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--compare_to_audit_json", type=Path, default=None)
    args = parser.parse_args()

    full_ds = WetnessPatchDataset(args.input_dir, split=None, allow_uncertain=False)
    train_ds = WetnessPatchDataset(args.input_dir, split="train", allow_uncertain=False)
    val_ds = WetnessPatchDataset(args.input_dir, split="val", allow_uncertain=False)
    test_ds = WetnessPatchDataset(args.input_dir, split="test", allow_uncertain=False)

    if len(full_ds) == 0:
        raise RuntimeError(f"No patches found in {args.input_dir}")

    sample = train_ds[0] if len(train_ds) > 0 else full_ds[0]
    rows = _load_manifest_rows(args.manifest_path)
    leakage_audit = _spatial_leakage_audit(rows)

    result = {
        "manifest_path": str(args.manifest_path.resolve()) if args.manifest_path.exists() else str(args.manifest_path),
        "input_dir": str(args.input_dir.resolve()) if args.input_dir.exists() else str(args.input_dir),
        "split_policy": str(sample.get("split_policy", "")),
        "dataset_counts": {
            "all": len(full_ds),
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "unique_block_counts": _unique_block_counts(full_ds),
        "label_counts": {
            "all": _label_counts(full_ds),
            "train": _label_counts(train_ds),
            "val": _label_counts(val_ds),
            "test": _label_counts(test_ds),
        },
        "confidence_tier_counts": {
            "all": _confidence_tier_counts(full_ds),
            "train": _confidence_tier_counts(train_ds),
            "val": _confidence_tier_counts(val_ds),
            "test": _confidence_tier_counts(test_ds),
        },
        "sample_schema_check": {
            "keys": sorted(sample.keys()),
            "cube_shape": list(sample["cube"].shape),
            "label": float(sample["label"].item()),
            "confidence": float(sample["confidence"].item()),
            "split": sample["split"],
            "file": sample["file"],
            "path": sample["path"],
            "block_id": sample.get("block_id", ""),
            "split_policy": sample.get("split_policy", ""),
            "confidence_tier": float(sample["confidence_tier"]),
            "valid_ratio": float(sample["valid_ratio"]),
            "wet_ratio_valid": float(sample["wet_ratio_valid"]),
        },
        "spatial_leakage_audit": leakage_audit,
        "touching_pair_counts": _summary_leakage_counts(leakage_audit),
    }

    if args.compare_to_audit_json is not None and args.compare_to_audit_json.exists():
        previous = json.loads(args.compare_to_audit_json.read_text(encoding="utf-8"))
        result["compared_to"] = str(args.compare_to_audit_json.resolve())
        result["leakage_comparison"] = _compare_audits(
            current=leakage_audit,
            previous=previous.get("spatial_leakage_audit", {}),
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_ready(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] dataset counts: {result['dataset_counts']}")
    print(f"[INFO] unique block counts: {result['unique_block_counts']}")
    print(f"[INFO] sample keys: {result['sample_schema_check']['keys']}")
    print(f"[INFO] touching pair counts: {result['touching_pair_counts']}")
    print(f"[INFO] audit saved to {args.output_json}")


if __name__ == "__main__":
    main()
