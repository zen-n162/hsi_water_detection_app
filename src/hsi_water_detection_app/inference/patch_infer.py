from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        "patch": np.ndarray,  # (C,H,W)
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
    target_signature: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> List[PatchPrediction]:
    """
    Real patch inference path when model provides .infer_patch(...).
    Falls back to dummy outputs if model is a placeholder dict.

    Returns per patch:
    {
        "top": int,
        "left": int,
        "height": int,
        "width": int,
        "prob_map": np.ndarray (H,W),
        "spatial_attn": np.ndarray (H,W),
        "spectral_attn": np.ndarray (K,),
    }
    """
    if not patches:
        print("[INFO] infer_patches skipped because no patches were generated")
        return []

    outputs: List[PatchPrediction] = []

    # Dummy fallback
    if isinstance(model, dict) and model.get("status") == "dummy_model":
        for item in patches:
            h = item["height"]
            w = item["width"]
            patch = item["patch"]
            bands = patch.shape[0]

            outputs.append(
                {
                    "top": item["top"],
                    "left": item["left"],
                    "height": h,
                    "width": w,
                    "prob_map": np.zeros((h, w), dtype=np.float32),
                    "spatial_attn": np.zeros((h, w), dtype=np.float32),
                    "spectral_attn": np.zeros((bands,), dtype=np.float32),
                }
            )
        print(f"[INFO] infer_patches finished on {len(outputs)} patches (dummy fallback, device={device})")
        return outputs

    # Real path
    if not hasattr(model, "infer_patch"):
        raise AttributeError("Model object does not have infer_patch(...)")

    for item in patches:
        patch = item["patch"]  # (C,H,W)
        result = model.infer_patch(
            patch,
            target_signature=target_signature,
            return_attn=True,
            temperature=temperature,
        )

        outputs.append(
            {
                "top": item["top"],
                "left": item["left"],
                "height": item["height"],
                "width": item["width"],
                "prob_map": result["prob_map"],
                "spatial_attn": result["spatial_attn"],
                "spectral_attn": result["spectral_attn"],
            }
        )

    print(f"[INFO] infer_patches finished on {len(outputs)} patches (real model, device={device})")
    return outputs
