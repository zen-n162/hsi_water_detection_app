from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch


# ============================================================
# Path / import helpers
# ============================================================

def _candidate_hypersigma_roots() -> List[Path]:
    env_root = os.environ.get("HYPERSIGMA_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend([
        Path("/home/zennakamura/MasterResearch/HyperSIGMA"),
        Path.home() / "MasterResearch" / "HyperSIGMA",
    ])
    return candidates


def _resolve_hypersigma_root() -> Path:
    for root in _candidate_hypersigma_roots():
        td = root / "HyperspectralDetection" / "Target_Detection"
        if td.exists():
            return root
    raise FileNotFoundError(
        "Could not resolve HyperSIGMA root. "
        "Set HYPERSIGMA_ROOT or place repository under ~/MasterResearch/HyperSIGMA"
    )


_HYPERSIGMA_ROOT = _resolve_hypersigma_root()
_HYPERSPECTRAL_DETECTION_ROOT = _HYPERSIGMA_ROOT / "HyperspectralDetection"
_TARGET_DETECTION_ROOT = _HYPERSPECTRAL_DETECTION_ROOT / "Target_Detection"

for p in [
    str(_HYPERSIGMA_ROOT),
    str(_HYPERSPECTRAL_DETECTION_ROOT),
    str(_TARGET_DETECTION_ROOT),
]:
    if p not in sys.path:
        sys.path.append(p)

from Target_Detection.models.models import SSHTDFramework, SpatialHTDFramework  # noqa: E402


# ============================================================
# Checkpoint utilities
# ============================================================

def _default_spat_checkpoint() -> Path:
    env_path = os.environ.get("HYPERSIGMA_SPAT_CHECKPOINT")
    if env_path:
        return Path(env_path)
    return _HYPERSPECTRAL_DETECTION_ROOT / "spat-vit-b-checkpoint-1599.pth"


def _default_spec_checkpoint() -> Path:
    env_path = os.environ.get("HYPERSIGMA_SPEC_CHECKPOINT")
    if env_path:
        return Path(env_path)
    return _HYPERSPECTRAL_DETECTION_ROOT / "spec-vit-b-checkpoint-1599.pth"


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format; could not extract state_dict.")


def _load_checkpoint_file(path: str | Path) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return _extract_state_dict(obj)


def _filter_state_dict_for_model(
    model_state: Dict[str, torch.Tensor],
    loaded_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    """
    Returns:
      filtered_state,
      missing_keys_after_filter,
      skipped_shape_keys,
      skipped_unknown_keys
    """
    filtered: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    skipped_unknown: List[str] = []

    for k, v in loaded_state.items():
        if k not in model_state:
            skipped_unknown.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered[k] = v

    missing_after_filter = [k for k in model_state.keys() if k not in filtered]
    return filtered, missing_after_filter, skipped_shape, skipped_unknown


def _smart_load_submodule(
    submodule: torch.nn.Module,
    checkpoint_path: str | Path,
    name: str,
) -> Dict[str, Any]:
    print(f"[INFO] loading {name} checkpoint: {checkpoint_path}")
    loaded_state = _load_checkpoint_file(checkpoint_path)
    model_state = submodule.state_dict()

    filtered, missing_after_filter, skipped_shape, skipped_unknown = _filter_state_dict_for_model(
        model_state=model_state,
        loaded_state=loaded_state,
    )

    msg = submodule.load_state_dict(filtered, strict=False)

    info = {
        "checkpoint_path": str(checkpoint_path),
        "loaded_keys": len(filtered),
        "missing_keys_count": len(missing_after_filter),
        "skipped_shape_count": len(skipped_shape),
        "skipped_unknown_count": len(skipped_unknown),
        "missing_keys_sample": missing_after_filter[:20],
        "skipped_shape_sample": skipped_shape[:20],
        "skipped_unknown_sample": skipped_unknown[:20],
        "load_state_dict_msg": str(msg),
    }

    print(
        f"[INFO] {name} checkpoint loaded. "
        f"matched={info['loaded_keys']} "
        f"missing={info['missing_keys_count']} "
        f"shape_skipped={info['skipped_shape_count']} "
        f"unknown_skipped={info['skipped_unknown_count']}"
    )
    return info


# ============================================================
# Attention reduction helpers
# ============================================================

def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amin, amax):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_spatial_attention_map(
    spat_attn: Any,
    patch_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Best-effort reducer:
    - accepts list/tuple of attention tensors
    - tries to convert to HxW map
    """
    ph, pw = patch_hw

    if spat_attn is None:
        return np.zeros((ph, pw), dtype=np.float32)

    attn_list = spat_attn if isinstance(spat_attn, (list, tuple)) else [spat_attn]

    candidates = []
    for a in attn_list:
        if not torch.is_tensor(a):
            continue

        # possible shape [B,H,N,N]
        if a.ndim == 4:
            a_np = a.detach().float().mean(dim=1).cpu().numpy()  # [B,N,N]
            candidates.append(a_np)

    if not candidates:
        return np.zeros((ph, pw), dtype=np.float32)

    A = candidates[-1][0]  # last layer, batch 0
    N = A.shape[-1]

    # If CLS exists -> use CLS-to-token
    if N > 1:
        vec = A[0, 1:] if A.shape[0] == N else A[0, 1:]
    else:
        vec = A.reshape(-1)

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)

    # Try square token grid
    g = int(np.sqrt(vec.size))
    if g * g == vec.size:
        grid = vec.reshape(g, g)
        # simple nearest upsample via repeat
        ry = max(ph // g, 1)
        rx = max(pw // g, 1)
        up = np.repeat(np.repeat(grid, ry, axis=0), rx, axis=1)
        up = up[:ph, :pw]
        return _normalize01(up)

    return np.zeros((ph, pw), dtype=np.float32)


def _extract_spectral_attention_vector(
    spec_attn: Any,
    num_tokens_fallback: int = 100,
) -> np.ndarray:
    """
    Best-effort reducer for spectral attention.
    Returns 1D vector.
    """
    if spec_attn is None:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    attn_list = spec_attn if isinstance(spec_attn, (list, tuple)) else [spec_attn]

    candidates = []
    for a in attn_list:
        if not torch.is_tensor(a):
            continue
        if a.ndim == 4:
            a_np = a.detach().float().mean(dim=1).cpu().numpy()  # [B,N,N]
            candidates.append(a_np)

    if not candidates:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    A = candidates[-1][0]
    N = A.shape[-1]

    if N > 1:
        vec = A[0, 1:] if A.shape[0] == N else A.mean(axis=0)
    else:
        vec = A.reshape(-1)

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return _normalize01(vec)


# ============================================================
# Wrapper
# ============================================================

class HyperSigmaWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        model_type: str,
        spat_checkpoint: Optional[str] = None,
        spec_checkpoint: Optional[str] = None,
        load_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.model_type = model_type
        self.spat_checkpoint = spat_checkpoint
        self.spec_checkpoint = spec_checkpoint
        self.load_info = load_info or {}

        self.model.to(self.device)
        self.model.eval()

    def _build_target_signature(self, patch_chw: np.ndarray) -> torch.Tensor:
        """
        patch_chw: (C,H,W)
        returns: (1,C)
        """
        ts = patch_chw.mean(axis=(1, 2), keepdims=False).astype(np.float32)
        ts = torch.from_numpy(ts).unsqueeze(0).to(self.device)
        return ts

    @torch.no_grad()
    def infer_patch(self, patch_chw: np.ndarray) -> Dict[str, np.ndarray]:
        """
        patch_chw: (C,H,W)
        returns:
          {
            "prob_map": (H,W),
            "spatial_attn": (H,W),
            "spectral_attn": (T,)
          }
        """
        if patch_chw.ndim != 3:
            raise ValueError(f"Expected patch shape (C,H,W), got {patch_chw.shape}")

        c, h, w = patch_chw.shape
        x = torch.from_numpy(patch_chw.astype(np.float32)).unsqueeze(0).to(self.device)
        ts = self._build_target_signature(patch_chw)

        if self.model_type == "ss":
            out = self.model(x, ts, return_attn=True)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError("SS model did not return (output, spat_attn, spec_attn)")
            pred, spat_attn, spec_attn = out[0], out[1], out[2]

            prob_map = torch.sigmoid(pred).squeeze(0).detach().cpu().numpy().astype(np.float32)
            spatial_map = _extract_spatial_attention_map(spat_attn, patch_hw=(h, w))
            spectral_vec = _extract_spectral_attention_vector(spec_attn)

        else:
            pred = self.model(x, ts)
            prob_map = torch.sigmoid(pred).squeeze(0).detach().cpu().numpy().astype(np.float32)
            spatial_map = np.zeros((h, w), dtype=np.float32)
            spectral_vec = np.zeros((100,), dtype=np.float32)

        return {
            "prob_map": prob_map,
            "spatial_attn": spatial_map,
            "spectral_attn": spectral_vec,
        }


# ============================================================
# Public loader
# ============================================================

def load_model(
    model_checkpoint: Optional[str] = None,
    *,
    device: str = "cpu",
    in_channels: Optional[int] = None,
    patch_size: int = 64,
    model_type: str = "ss",
    spat_checkpoint: Optional[str] = None,
    spec_checkpoint: Optional[str] = None,
) -> HyperSigmaWrapper:
    """
    Recommended usage for research:
      load_model(
          device="cuda",
          in_channels=198,
          patch_size=64,
          model_type="ss",
          spat_checkpoint="/path/to/spat-vit-b-checkpoint-1599.pth",
          spec_checkpoint="/path/to/spec-vit-b-checkpoint-1599.pth",
      )

    Backward compatibility:
      - model_checkpoint is accepted but not sufficient for SS mode unless
        your downstream code explicitly uses a single custom checkpoint.
      - If spat/spec are omitted, environment/default paths are used.
    """
    if in_channels is None:
        raise ValueError("in_channels must be provided to build HyperSIGMA model")

    if model_type not in {"ss", "sa"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    args = argparse.Namespace()
    load_info: Dict[str, Any] = {
        "hypersigma_root": str(_HYPERSIGMA_ROOT),
        "model_type": model_type,
        "patch_size": patch_size,
        "in_channels": in_channels,
    }

    if model_type == "ss":
        model = SSHTDFramework(args=args, img_size=patch_size, in_channels=in_channels)

        spat_ckpt = Path(spat_checkpoint) if spat_checkpoint else _default_spat_checkpoint()
        spec_ckpt = Path(spec_checkpoint) if spec_checkpoint else _default_spec_checkpoint()

        load_info["spat"] = _smart_load_submodule(model.spat_encoder, spat_ckpt, "spat_encoder")
        load_info["spec"] = _smart_load_submodule(model.spec_encoder, spec_ckpt, "spec_encoder")

        wrapper = HyperSigmaWrapper(
            model=model,
            device=device,
            model_type=model_type,
            spat_checkpoint=str(spat_ckpt),
            spec_checkpoint=str(spec_ckpt),
            load_info=load_info,
        )
        print(
            f"[INFO] built HyperSIGMA model_type={model_type}, "
            f"patch_size={patch_size}, in_channels={in_channels}, device={device}"
        )
        return wrapper

    # Spatial-only mode
    model = SpatialHTDFramework(args=args, img_size=patch_size, in_channels=in_channels)

    spat_ckpt = (
        Path(spat_checkpoint)
        if spat_checkpoint
        else Path(model_checkpoint) if model_checkpoint
        else _default_spat_checkpoint()
    )
    load_info["spat"] = _smart_load_submodule(model.encoder, spat_ckpt, "encoder")

    wrapper = HyperSigmaWrapper(
        model=model,
        device=device,
        model_type=model_type,
        spat_checkpoint=str(spat_ckpt),
        spec_checkpoint=None,
        load_info=load_info,
    )
    print(
        f"[INFO] built HyperSIGMA model_type={model_type}, "
        f"patch_size={patch_size}, in_channels={in_channels}, device={device}"
    )
    return wrapper
