from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from research.training.dataloaders import WetnessPatchDataset
from research.training.losses import confidence_weighted_bce


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline wetness detector")
    parser.add_argument("--input_dir", type=Path, default=Path("datasets/processed/wetness_pretrain"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    ds = WetnessPatchDataset(args.input_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No training patches found in {args.input_dir}")

    sample = ds[0]
    in_channels = int(sample["cube"].shape[0])

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BaselineNet(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            x = batch["cube"].to(device)
            y = batch["label"].to(device)
            c = batch["confidence"].to(device)

            logits = model(x)
            loss = confidence_weighted_bce(logits, y, c)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[INFO] epoch={epoch+1} loss={float(loss):.6f}")

    out = Path("experiments/checkpoints/baseline_wetness.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"[INFO] saved baseline checkpoint to {out}")


if __name__ == "__main__":
    main()
