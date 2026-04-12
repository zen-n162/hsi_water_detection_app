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


def reconstruct_spatial_attention_from_patches(
    patch_outputs: List[Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]],
) -> Optional[np.ndarray]:
    """
    Reconstruct full-size spatial attention map from patch_outputs[*]['spatial_attn'].
    """
    if image_shape is None:
        print("[INFO] reconstruct_spatial_attention_from_patches skipped because image_shape is None")
        return None

    if not patch_outputs:
        print("[INFO] reconstruct_spatial_attention_from_patches skipped because patch_outputs is empty")
        return np.zeros(image_shape, dtype=np.float32)

    height, width = image_shape
    accum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for item in patch_outputs:
        top = item["top"]
        left = item["left"]
        attn = item["spatial_attn"]
        h, w = attn.shape

        accum[top : top + h, left : left + w] += attn
        count[top : top + h, left : left + w] += 1.0

    count[count == 0] = 1.0
    out = accum / count

    amin = np.nanmin(out)
    amax = np.nanmax(out)
    if np.isclose(amin, amax):
        out = np.zeros_like(out, dtype=np.float32)
    else:
        out = (out - amin) / (amax - amin)

    print("[INFO] reconstruct_spatial_attention_from_patches completed")
    return out
