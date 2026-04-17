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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HYPERSIGMA_ROOT: Path | None = None
_HYPERSPECTRAL_DETECTION_ROOT: Path | None = None
_TARGET_DETECTION_ROOT: Path | None = None
SSHTDFramework = None
SpatialHTDFramework = None


def dprint(*args):
    if DEBUG:
        print("[DEBUG]", *args)


def _resolve_env_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _candidate_hypersigma_roots() -> List[Path]:
    env_root = os.environ.get("HYPERSIGMA_ROOT")
    runtime_root = os.environ.get("HSI_RUNTIME_ROOT")
    candidates: List[Path] = []
    if env_root:
        candidates.append(_resolve_env_path(env_root))
    if runtime_root:
        candidates.append(_resolve_env_path(runtime_root) / "upstream" / "HyperSIGMA")
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


def _ensure_hypersigma_imports() -> tuple[Path, Path, Path, Any, Any]:
    global _HYPERSIGMA_ROOT
    global _HYPERSPECTRAL_DETECTION_ROOT
    global _TARGET_DETECTION_ROOT
    global SSHTDFramework
    global SpatialHTDFramework

    if (
        _HYPERSIGMA_ROOT is not None
        and _HYPERSPECTRAL_DETECTION_ROOT is not None
        and _TARGET_DETECTION_ROOT is not None
        and SSHTDFramework is not None
        and SpatialHTDFramework is not None
    ):
        return (
            _HYPERSIGMA_ROOT,
            _HYPERSPECTRAL_DETECTION_ROOT,
            _TARGET_DETECTION_ROOT,
            SSHTDFramework,
            SpatialHTDFramework,
        )

    hypersigma_root = _resolve_hypersigma_root()
    hyperspectral_detection_root = hypersigma_root / "HyperspectralDetection"
    target_detection_root = hyperspectral_detection_root / "Target_Detection"

    for p in [
        str(hypersigma_root),
        str(hyperspectral_detection_root),
        str(target_detection_root),
    ]:
        if p not in sys.path:
            sys.path.append(p)

    from Target_Detection.models.models import (  # noqa: E402
        SSHTDFramework as ImportedSSHTDFramework,
        SpatialHTDFramework as ImportedSpatialHTDFramework,
    )

    _HYPERSIGMA_ROOT = hypersigma_root
    _HYPERSPECTRAL_DETECTION_ROOT = hyperspectral_detection_root
    _TARGET_DETECTION_ROOT = target_detection_root
    SSHTDFramework = ImportedSSHTDFramework
    SpatialHTDFramework = ImportedSpatialHTDFramework

    dprint("Resolved HyperSIGMA root:", _HYPERSIGMA_ROOT)
    dprint("Resolved HyperspectralDetection root:", _HYPERSPECTRAL_DETECTION_ROOT)
    dprint("Resolved Target_Detection root:", _TARGET_DETECTION_ROOT)

    return (
        _HYPERSIGMA_ROOT,
        _HYPERSPECTRAL_DETECTION_ROOT,
        _TARGET_DETECTION_ROOT,
        SSHTDFramework,
        SpatialHTDFramework,
    )


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _default_spat_checkpoint() -> Path:
    env_path = os.environ.get("HYPERSIGMA_SPAT_CHECKPOINT")
    if env_path:
        return _resolve_env_path(env_path)
    _, hyperspectral_detection_root, _, _, _ = _ensure_hypersigma_imports()
    return hyperspectral_detection_root / "spat-vit-b-checkpoint-1599.pth"


def _default_spec_checkpoint() -> Path:
    env_path = os.environ.get("HYPERSIGMA_SPEC_CHECKPOINT")
    if env_path:
        return _resolve_env_path(env_path)
    _, hyperspectral_detection_root, _, _, _ = _ensure_hypersigma_imports()
    return hyperspectral_detection_root / "spec-vit-b-checkpoint-1599.pth"


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


def _smart_load_full_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    name: str = "fine_tuned_model",
) -> Dict[str, Any]:
    print(f"[INFO] loading {name} checkpoint: {checkpoint_path}")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dprint("Fine-tuned checkpoint object type:", type(obj))

    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        loaded_state = obj["model_state_dict"]
        dprint("Using checkpoint['model_state_dict']")
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        loaded_state = obj["state_dict"]
        dprint("Using checkpoint['state_dict']")
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        loaded_state = obj["model"]
        dprint("Using checkpoint['model']")
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        loaded_state = obj
        dprint("Using checkpoint as raw state_dict")
    else:
        raise ValueError(f"Unsupported fine-tuned checkpoint format: {ckpt_path}")

    model_state = model.state_dict()
    filtered, missing_after_filter, skipped_shape, skipped_unknown = _filter_state_dict_for_model(
        model_state=model_state,
        loaded_state=loaded_state,
    )

    msg = model.load_state_dict(filtered, strict=False)

    info = {
        "checkpoint_path": str(ckpt_path),
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
    if spat_attn is None:
        return np.zeros((ph, pw), dtype=np.float32)

    attn_list = spat_attn if isinstance(spat_attn, (list, tuple)) else [spat_attn]
    candidates = []

    for a in attn_list:
        if not torch.is_tensor(a):
            continue
        if a.ndim == 4:
            a_np = a.detach().float().mean(dim=1).cpu().numpy()
            candidates.append(a_np)

    if not candidates:
        return np.zeros((ph, pw), dtype=np.float32)

    A = candidates[-1][0]
    if A.ndim != 2:
        return np.zeros((ph, pw), dtype=np.float32)

    N = A.shape[-1]
    vec = A[0, 1:] if N > 1 else A.reshape(-1)
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
    if spec_attn is None:
        return np.zeros((num_tokens_fallback,), dtype=np.float32)

    attn_list = spec_attn if isinstance(spec_attn, (list, tuple)) else [spec_attn]
    candidates = []

    for a in attn_list:
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
    vec = A[0, 1:] if N > 1 else A.reshape(-1)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return _normalize01(vec)


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
        ts = patch_chw.mean(axis=(1, 2), keepdims=False).astype(np.float32)
        ts = torch.from_numpy(ts).unsqueeze(0).to(self.device)
        dprint("Auto-built target signature:", _shape_of(ts))
        return ts

    @torch.no_grad()
    def infer_patch(
        self,
        patch_chw: np.ndarray,
        target_signature: Optional[np.ndarray | torch.Tensor] = None,
        return_attn: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if patch_chw.ndim != 3:
            raise ValueError(f"Expected patch shape (C,H,W), got {patch_chw.shape}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        _, h, w = patch_chw.shape
        dprint("infer_patch input patch shape:", patch_chw.shape)

        x = torch.from_numpy(patch_chw.astype(np.float32)).unsqueeze(0).to(self.device)
        dprint("Model input x shape:", _shape_of(x))

        if target_signature is None:
            ts = self._build_target_signature(patch_chw)
        else:
            if isinstance(target_signature, np.ndarray):
                ts = torch.from_numpy(target_signature.astype(np.float32))
            elif torch.is_tensor(target_signature):
                ts = target_signature.detach().float()
            else:
                raise TypeError(f"Unsupported target_signature type: {type(target_signature)}")

            if ts.ndim == 1:
                ts = ts.unsqueeze(0)
            if ts.ndim == 3 and ts.shape[-1] == 1:
                ts = ts.squeeze(-1)
            ts = ts.to(self.device)
            dprint("Provided target signature:", _shape_of(ts))

        try:
            if self.model_type == "ss":
                dprint(f"Calling SS model forward(return_attn={return_attn})")
                out = self.model(x, ts, return_attn=return_attn)

                if isinstance(out, (tuple, list)):
                    for i, item in enumerate(out):
                        dprint(f"SS output[{i}] ->", _shape_of(item))
                    if len(out) == 0:
                        raise RuntimeError("SS model returned empty tuple/list")
                    pred = out[0]
                    spat_attn = out[1] if len(out) > 1 else None
                    spec_attn = out[2] if len(out) > 2 else None
                else:
                    dprint("SS output ->", _shape_of(out))
                    pred = out
                    spat_attn = None
                    spec_attn = None

                pred_sigmoid = torch.sigmoid(pred / float(temperature))
                pred_np = pred_sigmoid.squeeze(0).detach().cpu().numpy().astype(np.float32)
                logit_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np[0]
                elif pred_np.ndim == 3 and pred_np.shape[-1] == 1:
                    pred_np = pred_np[..., 0]

                if logit_np.ndim == 3 and logit_np.shape[0] == 1:
                    logit_np = logit_np[0]
                elif logit_np.ndim == 3 and logit_np.shape[-1] == 1:
                    logit_np = logit_np[..., 0]

                if pred_np.ndim != 2:
                    raise RuntimeError(f"Expected prob_map to be 2D after squeeze, got {pred_np.shape}")
                if logit_np.ndim != 2:
                    raise RuntimeError(f"Expected logit_map to be 2D after squeeze, got {logit_np.shape}")

                prob_map = pred_np
                logit_map = logit_np
                spatial_map = _extract_spatial_attention_map(spat_attn, patch_hw=(h, w))
                spectral_vec = _extract_spectral_attention_vector(spec_attn)

            else:
                dprint("Calling SA model forward()")
                pred = self.model(x, ts)
                pred_sigmoid = torch.sigmoid(pred / float(temperature))
                pred_np = pred_sigmoid.squeeze(0).detach().cpu().numpy().astype(np.float32)
                logit_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np[0]
                elif pred_np.ndim == 3 and pred_np.shape[-1] == 1:
                    pred_np = pred_np[..., 0]

                if logit_np.ndim == 3 and logit_np.shape[0] == 1:
                    logit_np = logit_np[0]
                elif logit_np.ndim == 3 and logit_np.shape[-1] == 1:
                    logit_np = logit_np[..., 0]

                if pred_np.ndim != 2:
                    raise RuntimeError(f"Expected prob_map to be 2D after squeeze, got {pred_np.shape}")
                if logit_np.ndim != 2:
                    raise RuntimeError(f"Expected logit_map to be 2D after squeeze, got {logit_np.shape}")

                prob_map = pred_np
                logit_map = logit_np
                spatial_map = np.zeros((h, w), dtype=np.float32)
                spectral_vec = np.zeros((100,), dtype=np.float32)

            dprint("Final prob_map shape:", prob_map.shape)
            dprint("Final spatial_attn shape:", spatial_map.shape)
            dprint("Final spectral_attn shape:", spectral_vec.shape)

            return {
                "prob_map": prob_map,
                "probability_map": prob_map,
                "logit_map": logit_map,
                "logit_mean": np.asarray(float(np.mean(logit_map)), dtype=np.float32),
                "spatial_attn": spatial_map,
                "spectral_attn": spectral_vec,
            }

        except Exception as e:
            print("[ERROR] infer_patch failed:", repr(e))
            traceback.print_exc()
            raise


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

    hypersigma_root, hyperspectral_detection_root, _, ss_framework_cls, spatial_framework_cls = (
        _ensure_hypersigma_imports()
    )
    args = argparse.Namespace()
    load_info: Dict[str, Any] = {
        "hypersigma_root": str(hypersigma_root),
        "model_type": model_type,
        "patch_size": patch_size,
        "in_channels": in_channels,
    }

    try:
        if model_type == "ss":
            dprint("Building SSHTDFramework with temporary cwd:", hyperspectral_detection_root)
            with _pushd(hyperspectral_detection_root):
                model = ss_framework_cls(args=args, img_size=patch_size, in_channels=in_channels)

            # 1) まず pretrained backbone を読む
            spat_ckpt = Path(spat_checkpoint) if spat_checkpoint else _default_spat_checkpoint()
            spec_ckpt = Path(spec_checkpoint) if spec_checkpoint else _default_spec_checkpoint()

            load_info["spat"] = _smart_load_submodule(model.spat_encoder, spat_ckpt, "spat_encoder")
            load_info["spec"] = _smart_load_submodule(model.spec_encoder, spec_ckpt, "spec_encoder")

            # 2) その後で fine-tuned checkpoint があれば model 全体へ上書き
            if model_checkpoint is not None:
                load_info["fine_tuned"] = _smart_load_full_model(
                    model=model,
                    checkpoint_path=model_checkpoint,
                    name="fine_tuned_model",
                )

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

        dprint("Building SpatialHTDFramework with temporary cwd:", hyperspectral_detection_root)
        with _pushd(hyperspectral_detection_root):
            model = spatial_framework_cls(args=args, img_size=patch_size, in_channels=in_channels)

        spat_ckpt = (
            Path(spat_checkpoint)
            if spat_checkpoint
            else Path(model_checkpoint) if model_checkpoint and Path(model_checkpoint).suffix == ".pth"
            else _default_spat_checkpoint()
        )
        load_info["spat"] = _smart_load_submodule(model.encoder, spat_ckpt, "encoder")

        if model_checkpoint is not None:
            load_info["fine_tuned"] = _smart_load_full_model(
                model=model,
                checkpoint_path=model_checkpoint,
                name="fine_tuned_model",
            )

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
