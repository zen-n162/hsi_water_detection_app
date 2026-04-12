from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@contextmanager
def _temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class HyperSigmaWrapper:
    def __init__(
        self,
        module: torch.nn.Module,
        device: str,
        patch_size: int,
        in_channels: int,
        model_type: str = "ss",
    ):
        self.module = module
        self.device = device
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.model_type = model_type

    def _build_ts(self, patch_np: np.ndarray, target_signature: Optional[np.ndarray]) -> torch.Tensor:
        if target_signature is None:
            ts_np = patch_np.mean(axis=(1, 2), keepdims=False)[None, :]
        else:
            ts_np = np.asarray(target_signature, dtype=np.float32)
            if ts_np.ndim == 1:
                ts_np = ts_np[None, :]
            if ts_np.shape[1] != patch_np.shape[0]:
                raise ValueError(
                    f"target_signature channels {ts_np.shape[1]} != patch channels {patch_np.shape[0]}"
                )

        ts = torch.from_numpy(ts_np.astype(np.float32)).to(self.device)
        return ts

    @staticmethod
    def _normalize_map01(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        amin = np.nanmin(arr)
        amax = np.nanmax(arr)
        if np.isclose(amin, amax):
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - amin) / (amax - amin)

    def _spatial_attn_to_map(self, spat_attn: Any, height: int, width: int) -> np.ndarray:
        if spat_attn is None:
            return np.zeros((height, width), dtype=np.float32)

        tensors = list(spat_attn) if isinstance(spat_attn, (list, tuple)) else [spat_attn]
        chosen = None
        for t in reversed(tensors):
            if torch.is_tensor(t) and t.ndim == 4 and t.shape[-1] == t.shape[-2]:
                if t.shape[0] < 8 and t.shape[1] > 8:
                    t = t.permute(1, 0, 2, 3).contiguous()
                chosen = t
                break

        if chosen is None:
            return np.zeros((height, width), dtype=np.float32)

        attn = chosen.mean(dim=1)
        attn = torch.clamp(attn.float(), min=0.0)

        N = attn.shape[-1]
        expected = height * width

        if N == expected + 1:
            vec = attn[0, 0, 1:]
        elif N == expected:
            vec = attn[0].mean(dim=0)
        else:
            return np.zeros((height, width), dtype=np.float32)

        arr = vec.detach().cpu().numpy().reshape(height, width)
        return self._normalize_map01(arr)

    def _spectral_attn_to_vector(self, spec_attn: Any, bands: int) -> np.ndarray:
        if spec_attn is None:
            return np.zeros((bands,), dtype=np.float32)

        tensors = list(spec_attn) if isinstance(spec_attn, (list, tuple)) else [spec_attn]
        chosen = None
        for t in reversed(tensors):
            if torch.is_tensor(t) and t.ndim == 4 and t.shape[-1] == t.shape[-2]:
                if t.shape[0] < 8 and t.shape[1] > 8:
                    t = t.permute(1, 0, 2, 3).contiguous()
                chosen = t
                break

        if chosen is None:
            return np.zeros((bands,), dtype=np.float32)

        attn = chosen.mean(dim=1)
        vec = attn[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return self._normalize_map01(vec)

    def infer_patch(
        self,
        patch_np: np.ndarray,
        target_signature: Optional[np.ndarray] = None,
        return_attn: bool = True,
    ) -> Dict[str, Any]:
        if patch_np.ndim != 3:
            raise ValueError("patch_np must have shape (C, H, W)")

        bands, height, width = patch_np.shape
        x = torch.from_numpy(patch_np[None, ...].astype(np.float32)).to(self.device)
        ts = self._build_ts(patch_np, target_signature)

        with torch.no_grad():
            out = self.module(x, ts, return_attn=return_attn)

        if return_attn:
            score_map, spat_attn, spec_attn = out
        else:
            score_map = out
            spat_attn = None
            spec_attn = None

        score_map_np = score_map.squeeze(0).detach().cpu().numpy().astype(np.float32)
        score_map_np = np.clip(score_map_np, 0.0, 1.0)

        spatial_map = self._spatial_attn_to_map(spat_attn, height=height, width=width)
        spectral_vec = self._spectral_attn_to_vector(spec_attn, bands=bands)

        return {
            "prob_map": score_map_np,
            "spatial_attn": spatial_map,
            "spectral_attn": spectral_vec,
            "raw": {"spat_attn": spat_attn, "spec_attn": spec_attn},
        }


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    *,
    in_channels: Optional[int] = None,
    patch_size: int = 64,
    hyper_sigma_root: Optional[str] = None,
    model_type: str = "ss",
) -> Any:
    ckpt = Path(checkpoint_path)

    root = Path(
        hyper_sigma_root
        or os.environ.get(
            "HYPERSIGMA_ROOT",
            "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection",
        )
    ).expanduser().resolve()

    if in_channels is None:
        if ckpt.exists():
            print("[WARN] in_channels is None; trying full-model load only")
        else:
            print("[WARN] in_channels is None and no full-model checkpoint exists; returning dummy model")
            return {
                "checkpoint": str(ckpt),
                "device": device,
                "status": "dummy_model",
            }

    if not root.exists():
        raise FileNotFoundError(f"HYPERSIGMA_ROOT does not exist: {root}")

    sys.path.insert(0, str(root))
    from Target_Detection.models.models import SSHTDFramework, SpatialHTDFramework

    if ckpt.exists():
        try:
            with _temporary_cwd(root):
                full_model = torch.load(ckpt, map_location=device, weights_only=False)
            if isinstance(full_model, torch.nn.Module):
                full_model = full_model.to(device)
                full_model.eval()
                print(f"[INFO] loaded full serialized HTD model from: {ckpt}")
                return HyperSigmaWrapper(
                    module=full_model,
                    device=device,
                    patch_size=patch_size,
                    in_channels=in_channels or -1,
                    model_type=model_type,
                )
        except Exception as e:
            print(f"[WARN] full-model torch.load failed for {ckpt}: {e}")

    if in_channels is None:
        raise ValueError("in_channels must be provided when building SSHTDFramework/SpatialHTDFramework")

    framework_cls = SSHTDFramework if model_type == "ss" else SpatialHTDFramework

    with _temporary_cwd(root):
        model = framework_cls(args=None, img_size=patch_size, in_channels=in_channels).to(device)

    model.eval()
    print(
        f"[INFO] built HyperSIGMA model_type={model_type}, "
        f"patch_size={patch_size}, in_channels={in_channels}, device={device}"
    )

    return HyperSigmaWrapper(
        module=model,
        device=device,
        patch_size=patch_size,
        in_channels=in_channels,
        model_type=model_type,
    )
