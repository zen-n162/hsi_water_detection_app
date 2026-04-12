from __future__ import annotations

import torch
import torch.nn.functional as F


def confidence_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = loss * confidence
    return weighted.mean()
