from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from research.training.dataloaders import WetnessPatchDataset
from hsi_water_detection_app.models.hyper_sigma import load_model


def extract_logits_from_output(out: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        score_map = out[0]
    else:
        score_map = out

    if score_map.ndim != 3:
        raise ValueError(f"Expected score_map with shape (B,H,W), got {tuple(score_map.shape)}")

    logits = score_map.mean(dim=(1, 2))
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wetness detector")
    parser.add_argument("--input_dir", type=Path, default=Path("datasets/processed/wetness_pretrain"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_type", type=str, default="ss", choices=["ss", "sa"])
    parser.add_argument("--spat_checkpoint", type=str, required=True)
    parser.add_argument("--spec_checkpoint", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    args = parser.parse_args()

    ds = WetnessPatchDataset(args.input_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No evaluation patches found in {args.input_dir}")

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    sample = ds[0]
    in_channels = int(sample["cube"].shape[0])

    wrapper = load_model(
        device=args.device,
        in_channels=in_channels,
        patch_size=args.patch_size,
        model_type=args.model_type,
        spat_checkpoint=args.spat_checkpoint,
        spec_checkpoint=args.spec_checkpoint,
    )

    model = wrapper.model
    model.eval()

    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in dl:
            x = batch["cube"].to(args.device)
            y = batch["label"].cpu().numpy()
            ts = x.mean(dim=(2, 3))

            # return_attn=False 経路の不具合回避
            if args.model_type == "ss":
                out = model(x, ts, return_attn=True)
            else:
                out = model(x, ts)

            logits = extract_logits_from_output(out)
            prob = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y.tolist())
            y_score.extend(prob.tolist())

    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    mask = (y_true == 0.0) | (y_true == 1.0)
    if mask.sum() >= 2:
        auc = roc_auc_score(y_true[mask], y_score[mask])
        ap = average_precision_score(y_true[mask], y_score[mask])
        print(f"[INFO] ROC-AUC={auc:.6f}")
        print(f"[INFO] PR-AUC={ap:.6f}")
    else:
        print("[WARN] not enough hard labels for ROC-AUC / PR-AUC")
        print(f"[INFO] y_true={y_true.tolist()}")
        print(f"[INFO] y_score={y_score.tolist()}")


if __name__ == "__main__":
    main()
