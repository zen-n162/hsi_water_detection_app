from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PatchInfo = Dict[str, Any]
PatchPrediction = Dict[str, Any]


def generate_patches(
    cube: Optional[np.ndarray],
    patch_size: int,
    stride: int,
) -> List[PatchInfo]:
    """
    Slice cube into patches.

    Returns a list of dicts:
    {
        "patch": np.ndarray,
        "top": int,
        "left": int,
        "height": int,
        "width": int,
    }
    """
    if cube is None:
        print("[INFO] generate_patches skipped because cube is None")
        return []

    if cube.ndim != 3:
        raise ValueError("cube must have shape (bands, height, width)")

    _, height, width = cube.shape
    patches: List[PatchInfo] = []

    for top in range(0, max(height - patch_size + 1, 1), stride):
        for left in range(0, max(width - patch_size + 1, 1), stride):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)

            patch = cube[:, top:bottom, left:right]
            patches.append(
                {
                    "patch": patch,
                    "top": top,
                    "left": left,
                    "height": patch.shape[1],
                    "width": patch.shape[2],
                }
            )

    print(f"[INFO] generate_patches created {len(patches)} patches")
    return patches


def infer_patches(
    model: Any,
    patches: List[PatchInfo],
    device: str = "cpu",
) -> List[PatchPrediction]:
    """
    Dummy patch inference.

    For now:
    - returns zero probability maps
    - returns zero spatial attention
    - returns zero spectral attention
    """
    if not patches:
        print("[INFO] infer_patches skipped because no patches were generated")
        return []

    outputs: List[PatchPrediction] = []

    for item in patches:
        h = item["height"]
        w = item["width"]
        patch = item["patch"]
        bands = patch.shape[0]

        prob_map = np.zeros((h, w), dtype=np.float32)
        spatial_attn = np.zeros((h, w), dtype=np.float32)
        spectral_attn = np.zeros((bands,), dtype=np.float32)

        outputs.append(
            {
                "top": item["top"],
                "left": item["left"],
                "height": h,
                "width": w,
                "prob_map": prob_map,
                "spatial_attn": spatial_attn,
                "spectral_attn": spectral_attn,
            }
        )

    print(f"[INFO] infer_patches finished on {len(outputs)} patches (device={device})")
    return outputs
