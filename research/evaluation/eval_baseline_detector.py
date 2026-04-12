from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from research.training.dataloaders import WetnessPatchDataset


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


def _label_to_int(label) -> int:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, np.ndarray):
        label = label.item()
    if isinstance(label, torch.Tensor):
        label = label.item()
    if isinstance(label, str):
        label = label.strip().lower()
        if label == "wet":
            return 1
        if label == "dry":
            return 0
    return int(float(label) >= 0.5)


def _resolve_split(npz_path: Path) -> Optional[str]:
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    split = meta.get("split")
    return None if split is None else str(split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    ds = WetnessPatchDataset(args.input_dir, expected_hw=(64, 64))
    if len(ds) == 0:
        raise RuntimeError(f"No dataset patches found in {args.input_dir}")

    selected_files: List[Path] = []
    split_counts = {}

    for p in ds.files:
        split = _resolve_split(p)
        split_counts[split] = split_counts.get(split, 0) + 1
        if args.split == "all" or split == args.split:
            selected_files.append(p)

    print("[INFO] discovered split counts:", split_counts)

    if not selected_files:
        raise RuntimeError(f"No patches found for split={args.split}")

    first = np.load(selected_files[0], allow_pickle=True)
    in_channels = int(first["cube"].shape[0])

    model = BaselineNet(in_channels=in_channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(args.device)
    model.eval()

    y_true = []
    y_score = []
    sample_rows = []

    with torch.no_grad():
        for path in selected_files:
            d = np.load(path, allow_pickle=True)
            cube = d["cube"].astype(np.float32)
            label = _label_to_int(d["label"])

            x = torch.from_numpy(cube).unsqueeze(0).to(args.device)
            score = torch.sigmoid(model(x)).item()
            pred = int(score >= args.threshold)

            y_true.append(label)
            y_score.append(score)
            sample_rows.append(
                {
                    "file": path.name,
                    "y_true": int(label),
                    "y_score": float(score),
                    "y_pred": int(pred),
                }
            )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= args.threshold).astype(np.int64)

    metrics = {
        "model_name": "baseline",
        "split": args.split,
        "threshold": args.threshold,
        "n_samples": int(len(y_true)),
        "n_wet": int((y_true == 1).sum()),
        "n_dry": int((y_true == 0).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["dry(0)", "wet(1)"],
        "samples": sample_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] split={metrics['split']}")
    print(f"[INFO] n_samples={metrics['n_samples']} wet={metrics['n_wet']} dry={metrics['n_dry']}")
    print(f"[INFO] ROC-AUC={metrics['roc_auc']}")
    print(f"[INFO] PR-AUC={metrics['pr_auc']}")
    print(f"[INFO] Precision@{args.threshold:.2f}={metrics['precision']}")
    print(f"[INFO] Recall@{args.threshold:.2f}={metrics['recall']}")
    print(f"[INFO] F1@{args.threshold:.2f}={metrics['f1']}")
    print("[INFO] Confusion Matrix [[TN, FP], [FN, TP]] =")
    print(np.array(metrics["confusion_matrix"]))
    print(f"[INFO] metrics saved to {out_path}")


if __name__ == "__main__":
    main()
