from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from research.training.dataloaders import WetnessPatchDataset
from research.training.losses import confidence_weighted_bce


DEFAULT_INPUT_DIR = Path("datasets/processed/wetness_pretrain_v2")
DEFAULT_MANIFEST_PATH = Path("annotations/manifests/wetness_manifest_from_confidence_ali.csv")
DEFAULT_SAVE_DIR = Path("experiments/runs_v2")


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _label_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        label_name = str(record["label_raw"])
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline wetness detector on the v2 patch dataset")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run_name", type=str, default="baseline_v2")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    train_ds = WetnessPatchDataset(
        args.input_dir,
        split=args.train_split,
        expected_hw=(64, 64),
        allow_uncertain=False,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No training patches found for split={args.train_split} in {args.input_dir}")

    sample = train_ds[0]
    in_channels = int(sample["cube"].shape[0])

    run_dir = args.save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "manifest_path": str(args.manifest_path.resolve()) if args.manifest_path.exists() else str(args.manifest_path),
        "input_dir": str(args.input_dir.resolve()) if args.input_dir.exists() else str(args.input_dir),
        "train_split": args.train_split,
        "n_train_samples": len(train_ds),
        "train_label_counts": _label_counts(train_ds),
    }

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = BaselineNet(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_ckpt_path = run_dir / "model_best.pt"
    last_ckpt_path = run_dir / "model_last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in dl:
            x = batch["cube"].to(device)
            y = batch["label"].to(device)
            c = batch["confidence"].to(device)

            logits = model(x)
            loss = confidence_weighted_bce(logits, y, c)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
            }
        )
        print(f"[INFO] epoch={epoch} train_loss={train_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "dataset_info": dataset_info,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "history": history,
        }
        torch.save(ckpt, last_ckpt_path)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(ckpt, best_ckpt_path)

    _save_json(
        run_dir / "train_history.json",
        {
            "run_name": args.run_name,
            "device": device,
            "dataset_info": dataset_info,
            "history": history,
            "best_train_loss": best_train_loss,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
        },
    )
    _save_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "device": device,
            "dataset_info": dataset_info,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
        },
    )

    print(f"[INFO] saved best baseline checkpoint to {best_ckpt_path}")
    print(f"[INFO] saved last baseline checkpoint to {last_ckpt_path}")
    print(f"[INFO] run directory: {run_dir}")


if __name__ == "__main__":
    main()
