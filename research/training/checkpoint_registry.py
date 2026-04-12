from __future__ import annotations

from pathlib import Path


REGISTRY = {
    "dummy": {
        "type": "legacy_single",
        "path": "dummy.pth",
    },
    "wetness-pretrain-v1": {
        "type": "dual",
        "spat_checkpoint": "experiments/checkpoints/wetness-pretrain-v1_spat.pth",
        "spec_checkpoint": "experiments/checkpoints/wetness-pretrain-v1_spec.pth",
    },
    "wetness-finetune-v1": {
        "type": "dual",
        "spat_checkpoint": "experiments/checkpoints/wetness-finetune-v1_spat.pth",
        "spec_checkpoint": "experiments/checkpoints/wetness-finetune-v1_spec.pth",
    },
}


def resolve_checkpoint(name: str) -> dict:
    if name not in REGISTRY:
        raise KeyError(f"Unknown checkpoint name: {name}")
    return REGISTRY[name]
