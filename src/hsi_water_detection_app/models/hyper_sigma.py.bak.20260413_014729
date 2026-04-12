from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch


DEBUG = os.environ.get("HSI_DEBUG", "1") == "1"


def dprint(*args):
    if DEBUG:
        print("[DEBUG]", *args)


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

dprint("Resolved HyperSIGMA root:", _HYPERSIGMA_ROOT)
dprint("Resolved HyperspectralDetection root:", _HYPERSPECTRAL_DETECTION_ROOT)
dprint("Resolved Target_Detection root:", _TARGET_DETECTION_ROOT)


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


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


def _shape_of(x: Any) -> str:
    if torch.is_tensor(x):
        return f"torch{tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, np.ndarray):
        return f"np{tuple(x.shape)} dtype={x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[len={len(x)}]"
    if isinstance(x, dict):
        return f"dict[len={len(x)}]"
    return str(type(x))


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    dprint("Checkpoint object type:", type(obj))
    if isinstance(obj, dict):
        dprint("Top-level checkpoint keys sample:", list(obj.keys())[:20])

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            dprint("Using checkpoint['state_dict']")
            return obj["state_dict"]

        if "model" in obj and isinstance(obj["model"], dict):
            dprint("Using checkpoint['model']")
            return obj["model"]

        if all(isinstance(k, str) for k in obj.keys()):
            dprint("Using checkpoint as raw state_dict")
            return obj

    raise ValueError("Unsupported checkpoint format; could not extract state_dict.")


def _load_checkpoint_file(path: str | Path) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dprint("Loading checkpoint file:", ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    dprint("Extracted state_dict keys sample:", list(state.keys())[:20])
    dprint("Extracted state_dict size:", len(state))
    return state


def _filter_state_dict_for_model(
    model_state: Dict[str, torch.Tensor],
    loaded_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    filtered: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    skipped_unknown: List[str] = []

    for k, v in loaded_state.items():
        if k not in model_state:
            skipped_unknown.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append(
                f"{k}: ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"
            )
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

    dprint(f"{name} model state_dict size:", len(model_state))
    dprint(f"{name} model keys sample:", list(model_state.keys())[:20])

    filtered, missing_after_filter, skipped_shape, skipped_unknown = _filter_state_dict_for_model(
        model_state=model_state,
        loaded_state=loaded_state,
    )

    dprint(f"{name} matched keys sample:", list(filtered.keys())[:20])
    dprint(f"{name} missing keys sample:", missing_after_filter[:20])
    dprint(f"{name} skipped shape sample:", skipped_shape[:20])
    dprint(f"{name} skipped unknown sample:", skipped_unknown[:20])

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


def _extract_spatial_attention_map(
    spat_attn: Any,
    patch_hw: Tuple[int, int],
) -> np.ndarray:
    ph, pw = patch_hw
    dprint("spat_attn container:", _shape_of(spat_attn))

    if spat_attn is None:
        return np.zeros((ph, pw), dtype=np.float32)

    attn_list = spat_attn if isinstance(spat_attn, (list, tuple)) else [spat_attn]

    candidates = []
    for idx, a in enumerate(attn_list):
        dprint(f"spat_attn[{idx}] shape:", _shape_of(a))
        if not torch.is_tensor(a):
            continue
        if a.ndim == 4:
            a_np = a.detach().float().mean(dim=1).cpu().numpy()
            candidates.append(a_np)

    if not candidates:
        return np.zeros((ph, pw), dtype=np.float32)

    A = candidates[-1][0]
    dprint("Selected spatial attention matrix shape:", A.shape)

    if A.ndim != 2:
        return np.zeros((ph, pw), dtype=np.float32)

    N = A.shape[-1]
    if N > 1:
        vec = A[0, 1:]
    else:
        vec = A.reshape(-1)

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    g = int(np.sqrt(vec.size))

    if g * g == vec.size and g > 0:
        grid = vec.reshape(g, g)
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
    dprint("spec_attn container:", _shape_of(spec_attn))

    if spec_attn is None:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    attn_list = spec_attn if isinstance(spec_attn, (list, tuple)) else [spec_attn]

    candidates = []
    for idx, a in enumerate(attn_list):
        dprint(f"spec_attn[{idx}] shape:", _shape_of(a))
        if not torch.is_tensor(a):
            continue
        if a.ndim == 4:
            a_np = a.detach().float().mean(dim=1).cpu().numpy()
            candidates.append(a_np)

    if not candidates:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    A = candidates[-1][0]
    if A.ndim != 2:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    N = A.shape[-1]
    if N > 1:
        vec = A[0, 1:]
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

        dprint("Wrapper initialized with device:", self.device)
        dprint("Wrapper model_type:", self.model_type)

    def _build_target_signature(self, patch_chw: np.ndarray) -> torch.Tensor:
        ts = patch_chw.mean(axis=(1, 2), keepdims=False).astype(np.float32)
        ts = torch.from_numpy(ts).unsqueeze(0).to(self.device)
        dprint("Target signature tensor shape:", _shape_of(ts))
        return ts

    @torch.no_grad()
    def infer_patch(self, patch_chw: np.ndarray) -> Dict[str, np.ndarray]:
        if patch_chw.ndim != 3:
            raise ValueError(f"Expected patch shape (C,H,W), got {patch_chw.shape}")

        c, h, w = patch_chw.shape
        dprint("infer_patch input patch shape:", patch_chw.shape)

        x = torch.from_numpy(patch_chw.astype(np.float32)).unsqueeze(0).to(self.device)
        dprint("Model input x shape:", _shape_of(x))
        ts = self._build_target_signature(patch_chw)

        try:
            if self.model_type == "ss":
                dprint("Calling SS model forward(return_attn=True)")
                out = self.model(x, ts, return_attn=True)

                dprint("SS model raw output type:", type(out))
                if isinstance(out, (tuple, list)):
                    for i, item in enumerate(out):
                        dprint(f"SS output[{i}] ->", _shape_of(item))
                else:
                    raise RuntimeError("SS model output is not tuple/list")

                if len(out) < 3:
                    raise RuntimeError("SS model did not return (output, spat_attn, spec_attn)")

                pred, spat_attn, spec_attn = out[0], out[1], out[2]

                pred_sigmoid = torch.sigmoid(pred)
                pred_np = pred_sigmoid.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np[0]
                elif pred_np.ndim == 3 and pred_np.shape[-1] == 1:
                    pred_np = pred_np[..., 0]

                if pred_np.ndim != 2:
                    raise RuntimeError(f"Expected prob_map to be 2D after squeeze, got {pred_np.shape}")

                prob_map = pred_np
                spatial_map = _extract_spatial_attention_map(spat_attn, patch_hw=(h, w))
                spectral_vec = _extract_spectral_attention_vector(spec_attn)

            else:
                dprint("Calling SA model forward()")
                pred = self.model(x, ts)
                pred_sigmoid = torch.sigmoid(pred)
                pred_np = pred_sigmoid.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np[0]
                elif pred_np.ndim == 3 and pred_np.shape[-1] == 1:
                    pred_np = pred_np[..., 0]

                if pred_np.ndim != 2:
                    raise RuntimeError(f"Expected prob_map to be 2D after squeeze, got {pred_np.shape}")

                prob_map = pred_np
                spatial_map = np.zeros((h, w), dtype=np.float32)
                spectral_vec = np.zeros((100,), dtype=np.float32)

            dprint("Final prob_map shape:", prob_map.shape)
            dprint("Final spatial_attn shape:", spatial_map.shape)
            dprint("Final spectral_attn shape:", spectral_vec.shape)

            return {
                "prob_map": prob_map,
                "spatial_attn": spatial_map,
                "spectral_attn": spectral_vec,
            }

        except Exception as e:
            print("[ERROR] infer_patch failed:", repr(e))
            traceback.print_exc()
            raise


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
    if in_channels is None:
        raise ValueError("in_channels must be provided to build HyperSIGMA model")

    if model_type not in {"ss", "sa"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(
        f"[INFO] load_model called with "
        f"model_checkpoint={model_checkpoint}, "
        f"spat_checkpoint={spat_checkpoint}, "
        f"spec_checkpoint={spec_checkpoint}, "
        f"model_type={model_type}, "
        f"patch_size={patch_size}, "
        f"in_channels={in_channels}, "
        f"device={device}"
    )

    args = argparse.Namespace()
    load_info: Dict[str, Any] = {
        "hypersigma_root": str(_HYPERSIGMA_ROOT),
        "model_type": model_type,
        "patch_size": patch_size,
        "in_channels": in_channels,
    }

    try:
        if model_type == "ss":
            dprint("Building SSHTDFramework with temporary cwd:", _HYPERSPECTRAL_DETECTION_ROOT)
            with _pushd(_HYPERSPECTRAL_DETECTION_ROOT):
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

        dprint("Building SpatialHTDFramework with temporary cwd:", _HYPERSPECTRAL_DETECTION_ROOT)
        with _pushd(_HYPERSPECTRAL_DETECTION_ROOT):
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

    except Exception as e:
        print("[ERROR] load_model failed:", repr(e))
        traceback.print_exc()
        raise
