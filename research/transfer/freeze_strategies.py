from __future__ import annotations

import torch.nn as nn


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True
