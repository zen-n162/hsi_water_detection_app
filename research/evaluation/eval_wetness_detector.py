from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hsi_water_detection_app.models.hyper_sigma import load_model
from research.evaluation.metric_utils import compute_binary_metrics, format_histogram, safe_float, sigmoid_from_logits
from research.training.dataloaders import WetnessPatchDataset


DEFAULT_INPUT_DIR = Path("datasets/processed/wetness_pretrain_v2")
DEFAULT_MANIFEST_PATH = Path("annotations/manifests/wetness_manifest_from_confidence_ali.csv")


def _resolve_temperature(
    temperature: float | None,
    temperature_json: Path | None,
) -> tuple[float, str | None]:
    if temperature_json is not None:
        obj = json.loads(temperature_json.read_text(encoding="utf-8"))
        value = safe_float(
            obj.get("best_temperature", obj.get("temperature", 1.0)),
            default=1.0,
        )
        return value, str(temperature_json.resolve())
    if temperature is not None:
        return float(temperature), None
    return 1.0, None


def _split_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        split = str(record["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts


def _label_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        label_name = str(record["label_raw"])
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _extract_logit_mean(result: Dict[str, Any]) -> float:
    if "logit_mean" in result:
        return safe_float(result["logit_mean"], default=0.0)
    if "logit_map" in result:
        return float(np.asarray(result["logit_map"], dtype=np.float32).mean())
    raise KeyError("infer_patch() result does not contain 'logit_mean' or 'logit_map'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned HyperSIGMA wetness detector with optional temperature scaling")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_type", type=str, default="ss", choices=["ss", "sa"])
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--spat_checkpoint", type=str, default=None)
    parser.add_argument("--spec_checkpoint", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature_json", type=Path, default=None)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    full_ds = WetnessPatchDataset(args.input_dir, split=None, allow_uncertain=False)
    target_split = None if args.split == "all" else args.split
    ds = WetnessPatchDataset(args.input_dir, split=target_split, allow_uncertain=False)

    if len(ds) == 0:
        raise RuntimeError(
            f"No patches found for split={args.split} in {args.input_dir}. "
            f"discovered split counts={_split_counts(full_ds)}"
        )

    temperature_value, temperature_source = _resolve_temperature(
        args.temperature,
        args.temperature_json,
    )

    sample = ds[0]
    in_channels = int(sample["cube"].shape[0])

    model = load_model(
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        in_channels=in_channels,
        patch_size=args.patch_size,
        model_type=args.model_type,
        spat_checkpoint=args.spat_checkpoint,
        spec_checkpoint=args.spec_checkpoint,
    )

    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
        print("[INFO] set eval mode via model.model.eval()")
    elif hasattr(model, "eval"):
        model.eval()
        print("[INFO] set eval mode via model.eval()")

    y_true: List[int] = []
    y_score: List[float] = []
    y_logit: List[float] = []
    sample_rows: List[Dict[str, Any]] = []

    for idx in range(len(ds)):
        item = ds[idx]
        cube = item["cube"].cpu().numpy().astype(np.float32)
        label = int(float(item["label"].item()) >= 0.5)
        result = model.infer_patch(cube, return_attn=False, temperature=1.0)
        logit_mean = _extract_logit_mean(result)
        score = float(sigmoid_from_logits(np.asarray([logit_mean], dtype=np.float64), temperature=temperature_value)[0])
        pred = int(score >= args.threshold)

        y_true.append(label)
        y_score.append(score)
        y_logit.append(logit_mean)
        sample_rows.append(
            {
                "file": item["file"],
                "path": item["path"],
                "sample_id": item["sample_id"],
                "split": item["split"],
                "label_name": item["label_name"],
                "confidence": float(item["confidence"].item()),
                "confidence_tier": float(item["confidence_tier"]),
                "valid_ratio": float(item["valid_ratio"]),
                "wet_ratio_valid": float(item["wet_ratio_valid"]),
                "block_id": item.get("block_id", ""),
                "split_policy": item.get("split_policy", ""),
                "y_true": int(label),
                "y_logit_mean": float(logit_mean),
                "y_score": float(score),
                "y_pred": pred,
            }
        )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    metrics = compute_binary_metrics(
        y_true=y_true_np,
        y_score=y_score_np,
        threshold=args.threshold,
        n_bins=args.ece_bins,
    )
    _ = metrics.pop("y_pred")

    n_wet = int(sum(1 for x in y_true if x == 1))
    n_dry = int(sum(1 for x in y_true if x == 0))

    out_obj = {
        "model_name": "hypersigma",
        "split": args.split,
        "threshold": float(args.threshold),
        "n_samples": len(ds),
        "n_wet": n_wet,
        "n_dry": n_dry,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": ["dry(0)", "wet(1)"],
        "score_distribution": metrics["score_distribution"],
        "score_histogram": format_histogram(y_score_np, bins=10),
        "brier_score": metrics["brier_score"],
        "ece": metrics["ece"],
        "samples": sample_rows,
        "dataset_info": {
            "manifest_path": str(args.manifest_path.resolve()) if args.manifest_path.exists() else str(args.manifest_path),
            "input_dir": str(args.input_dir.resolve()) if args.input_dir.exists() else str(args.input_dir),
            "requested_split": args.split,
            "discovered_split_counts": _split_counts(full_ds),
            "selected_label_counts": _label_counts(ds),
        },
        "calibration_info": {
            "mode": "temperature_scaled" if temperature_value != 1.0 or temperature_source is not None else "uncalibrated",
            "temperature": float(temperature_value),
            "temperature_json": temperature_source,
        },
        "model_info": {
            "model_type": args.model_type,
            "checkpoint_mode": "fine_tuned" if args.model_checkpoint else "pretrained_backbone_only",
            "model_checkpoint": args.model_checkpoint,
            "spat_checkpoint": args.spat_checkpoint,
            "spec_checkpoint": args.spec_checkpoint,
            "device": args.device,
            "patch_size": args.patch_size,
            "in_channels": in_channels,
            "load_info": getattr(model, "load_info", {}),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_ready(out_obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    thr_disp = f"{args.threshold:.2f}"
    print(f"[INFO] split={args.split}")
    print(f"[INFO] n_samples={len(ds)} wet={n_wet} dry={n_dry}")
    print(f"[INFO] calibration_mode={out_obj['calibration_info']['mode']} temperature={out_obj['calibration_info']['temperature']:.6f}")
    print(f"[INFO] ROC-AUC={out_obj['roc_auc']:.6f}" if out_obj["roc_auc"] is not None else "[INFO] ROC-AUC=None")
    print(f"[INFO] PR-AUC={out_obj['pr_auc']:.6f}" if out_obj["pr_auc"] is not None else "[INFO] PR-AUC=None")
    print(f"[INFO] Precision@{thr_disp}={out_obj['precision']:.6f}")
    print(f"[INFO] Recall@{thr_disp}={out_obj['recall']:.6f}")
    print(f"[INFO] F1@{thr_disp}={out_obj['f1']:.6f}")
    print(f"[INFO] Brier={out_obj['brier_score']:.6f} ECE={out_obj['ece']:.6f}")
    print(
        f"[INFO] score range=({out_obj['score_distribution']['min']:.6f}, "
        f"{out_obj['score_distribution']['max']:.6f})"
    )
    print(f"[INFO] metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
