from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_model(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Dummy model loader.

    Later, replace with actual HyperSIGMA checkpoint loading.
    """
    ckpt = Path(checkpoint_path)

    if not ckpt.exists():
        print(f"[WARN] model checkpoint does not exist yet: {ckpt}")
        return {
            "checkpoint": str(ckpt),
            "device": device,
            "status": "dummy_model",
        }

    print(f"[INFO] load_model called with: {ckpt}, device={device}")
    return {
        "checkpoint": str(ckpt),
        "device": device,
        "status": "loaded_placeholder",
    }
