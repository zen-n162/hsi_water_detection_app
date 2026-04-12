from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_wavelength_sidecar(path: str | Path) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None

    suffix = p.suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(p), dtype=np.float32).reshape(-1)

    if suffix in {".csv", ".txt"}:
        values = []
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            parts = [x.strip() for x in line.replace(",", " ").split()]
            for token in parts:
                try:
                    values.append(float(token))
                except ValueError:
                    continue
        if not values:
            return None
        return np.asarray(values, dtype=np.float32)

    return None
