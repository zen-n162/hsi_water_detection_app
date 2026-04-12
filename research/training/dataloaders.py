from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "dry": 0.0,
    "wet": 1.0,
    "uncertain": 0.5,
}


class WetnessPatchDataset(Dataset):
    def __init__(self, input_dir: str | Path, expected_hw: tuple[int, int] = (64, 64)) -> None:
        self.input_dir = Path(input_dir)
        self.expected_hw = expected_hw
        self.files = []

        for path in sorted(self.input_dir.glob("*.npz")):
            try:
                data = np.load(path, allow_pickle=True)
                cube = data["cube"]
                if cube.ndim != 3:
                    continue
                if cube.shape[1:] != expected_hw:
                    print(f"[WARN] skip dataset file due to shape mismatch: {path} -> {cube.shape}")
                    continue
                self.files.append(path)
            except Exception as e:
                print(f"[WARN] skip unreadable dataset file: {path} ({e!r})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        data = np.load(path, allow_pickle=True)

        cube = data["cube"].astype(np.float32)
        label_raw = str(data["label"])
        confidence = float(data["confidence"])

        y = np.float32(LABEL_MAP.get(label_raw, 0.5))
        return {
            "cube": torch.from_numpy(cube),
            "label": torch.tensor(y, dtype=torch.float32),
            "confidence": torch.tensor(confidence, dtype=torch.float32),
            "path": str(path),
        }
