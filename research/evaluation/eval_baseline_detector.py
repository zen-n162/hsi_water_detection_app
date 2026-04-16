from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.evaluation.metric_utils import compute_binary_metrics
from research.training.dataloaders import WetnessPatchDataset


DEFAULT_INPUT_DIR = Path("datasets/processed/wetness_pretrain_v2")
DEFAULT_MANIFEST_PATH = Path("annotations/manifests/wetness_manifest_from_confidence_ali.csv")


class BaselineNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.head(feat).squeeze(1)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the baseline wetness detector on the v2 dataset")
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    full_ds = WetnessPatchDataset(args.input_dir, split=None, expected_hw=(64, 64), allow_uncertain=False)
    target_split = None if args.split == "all" else args.split
    ds = WetnessPatchDataset(args.input_dir, split=target_split, expected_hw=(64, 64), allow_uncertain=False)

    if len(ds) == 0:
        raise RuntimeError(f"No dataset patches found for split={args.split} in {args.input_dir}")

    sample = ds[0]
    in_channels = int(sample["cube"].shape[0])

    model = BaselineNet(in_channels=in_channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(args.device)
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []
    y_logit: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx in range(len(ds)):
            item = ds[idx]
            label = int(float(item["label"].item()) >= 0.5)
            cube = item["cube"].unsqueeze(0).to(args.device)
            logit = float(model(cube).item())
            score = float(torch.sigmoid(torch.tensor(logit)).item())
            pred = int(score >= args.threshold)

            y_true.append(label)
            y_score.append(score)
            y_logit.append(logit)
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
                    "y_logit": float(logit),
                    "y_true": int(label),
                    "y_score": float(score),
                    "y_pred": int(pred),
                }
            )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    metrics_core = compute_binary_metrics(
        y_true=y_true_np,
        y_score=y_score_np,
        threshold=args.threshold,
    )

    metrics = {
        "model_name": "baseline",
        "split": args.split,
        "threshold": args.threshold,
        "n_samples": int(len(y_true_np)),
        "n_wet": int((y_true_np == 1).sum()),
        "n_dry": int((y_true_np == 0).sum()),
        "roc_auc": metrics_core["roc_auc"],
        "pr_auc": metrics_core["pr_auc"],
        "precision": metrics_core["precision"],
        "recall": metrics_core["recall"],
        "f1": metrics_core["f1"],
        "confusion_matrix": metrics_core["confusion_matrix"],
        "confusion_matrix_labels": ["dry(0)", "wet(1)"],
        "brier_score": metrics_core["brier_score"],
        "ece": metrics_core["ece"],
        "dataset_info": {
            "manifest_path": str(args.manifest_path.resolve()) if args.manifest_path.exists() else str(args.manifest_path),
            "input_dir": str(args.input_dir.resolve()) if args.input_dir.exists() else str(args.input_dir),
            "requested_split": args.split,
            "discovered_split_counts": _split_counts(full_ds),
            "selected_label_counts": _label_counts(ds),
        },
        "model_info": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": args.device,
            "in_channels": in_channels,
        },
        "score_distribution": metrics_core["score_distribution"],
        "samples": sample_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] split={metrics['split']}")
    print(f"[INFO] n_samples={metrics['n_samples']} wet={metrics['n_wet']} dry={metrics['n_dry']}")
    print(f"[INFO] ROC-AUC={metrics['roc_auc']}")
    print(f"[INFO] PR-AUC={metrics['pr_auc']}")
    print(f"[INFO] Precision@{args.threshold:.2f}={metrics['precision']}")
    print(f"[INFO] Recall@{args.threshold:.2f}={metrics['recall']}")
    print(f"[INFO] F1@{args.threshold:.2f}={metrics['f1']}")
    print(f"[INFO] score range=({metrics['score_distribution']['min']:.6f}, {metrics['score_distribution']['max']:.6f})")
    print(f"[INFO] metrics saved to {out_path}")


if __name__ == "__main__":
    main()
