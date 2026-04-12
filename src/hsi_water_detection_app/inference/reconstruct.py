from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def reconstruct_from_patches(
    patch_outputs: List[Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]],
    patch_size: int,
    stride: int,
) -> Optional[np.ndarray]:
    """
    Reconstruct a full probability map from patch outputs.

    image_shape should be (height, width).
    """
    if image_shape is None:
        print("[INFO] reconstruct_from_patches skipped because image_shape is None")
        return None

    if not patch_outputs:
        print("[INFO] reconstruct_from_patches skipped because patch_outputs is empty")
        return np.zeros(image_shape, dtype=np.float32)

    height, width = image_shape
    accum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for item in patch_outputs:
        top = item["top"]
        left = item["left"]
        prob = item["prob_map"]
        h, w = prob.shape

        accum[top : top + h, left : left + w] += prob
        count[top : top + h, left : left + w] += 1.0

    count[count == 0] = 1.0
    out = accum / count
    print("[INFO] reconstruct_from_patches completed")
    return out
