from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from hsi_water_detection_app.models.hyper_sigma import load_model
from research.training.dataloaders import WetnessPatchDataset


DEFAULT_INPUT_DIR = Path("datasets/processed/wetness_pretrain_v2")
DEFAULT_MANIFEST_PATH = Path("annotations/manifests/wetness_manifest_from_confidence_ali.csv")
DEFAULT_SAVE_DIR = Path("experiments/runs_v2")


def confidence_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    weight = confidence.clamp(min=0.0, max=1.0)
    denom = torch.clamp(weight.sum(), min=1e-8)
    return (per_sample * weight).sum() / denom


def plain_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
        pos_weight=pos_weight,
    )


def patch_logits_to_sample_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.mean(dim=(1, 2))
    if logits.ndim == 4:
        return logits.mean(dim=(1, 2, 3))
    if logits.ndim == 2:
        return logits.mean(dim=1)
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def build_target_signature(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W)
    return x.mean(dim=(2, 3))


def set_requires_grad(module: Optional[torch.nn.Module], flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def maybe_get_blocks(module: Optional[torch.nn.Module]) -> list[torch.nn.Module]:
    if module is None:
        return []
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        return []
    try:
        return list(blocks)
    except Exception:
        return []


def freeze_all_encoders(model: torch.nn.Module) -> None:
    if hasattr(model, "spat_encoder"):
        set_requires_grad(model.spat_encoder, False)
    if hasattr(model, "spec_encoder"):
        set_requires_grad(model.spec_encoder, False)
    if hasattr(model, "encoder"):
        set_requires_grad(model.encoder, False)


def unfreeze_last_n_blocks(module: Optional[torch.nn.Module], n: int) -> None:
    if module is None or n <= 0:
        return
    blocks = maybe_get_blocks(module)
    if not blocks:
        set_requires_grad(module, True)
        return
    for blk in blocks[-n:]:
        set_requires_grad(blk, True)


def collect_head_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("spat_encoder.") or name.startswith("spec_encoder.") or name.startswith("encoder."):
            continue
        params.append(p)
    return params


def collect_encoder_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("spat_encoder.") or name.startswith("spec_encoder.") or name.startswith("encoder."):
            params.append(p)
    return params


def apply_freeze_strategy(
    model: torch.nn.Module,
    strategy: str,
    unfreeze_spat_last_n: int,
    unfreeze_spec_last_n: int,
) -> None:
    for p in model.parameters():
        p.requires_grad = True

    if strategy == "full":
        return

    if strategy == "head_only":
        freeze_all_encoders(model)
        return

    if strategy == "spec_last":
        freeze_all_encoders(model)
        if hasattr(model, "spec_encoder"):
            unfreeze_last_n_blocks(model.spec_encoder, unfreeze_spec_last_n)
        return

    if strategy == "spat_last":
        freeze_all_encoders(model)
        if hasattr(model, "spat_encoder"):
            unfreeze_last_n_blocks(model.spat_encoder, unfreeze_spat_last_n)
        elif hasattr(model, "encoder"):
            unfreeze_last_n_blocks(model.encoder, unfreeze_spat_last_n)
        return

    if strategy == "dual_last":
        freeze_all_encoders(model)
        if hasattr(model, "spat_encoder"):
            unfreeze_last_n_blocks(model.spat_encoder, unfreeze_spat_last_n)
        elif hasattr(model, "encoder"):
            unfreeze_last_n_blocks(model.encoder, unfreeze_spat_last_n)
        if hasattr(model, "spec_encoder"):
            unfreeze_last_n_blocks(model.spec_encoder, unfreeze_spec_last_n)
        return

    if strategy == "progressive":
        freeze_all_encoders(model)
        return

    raise ValueError(f"Unsupported freeze_strategy: {strategy}")


def build_optimizer(
    model: torch.nn.Module,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = collect_head_params(model)
    encoder_params = collect_encoder_params(model)

    param_groups: list[dict[str, Any]] = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if encoder_params and encoder_lr > 0:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check freeze strategy.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def compute_auto_pos_weight(dataset: WetnessPatchDataset, device: str) -> torch.Tensor:
    positives = sum(1 for r in dataset.records if str(r["label_raw"]) == "wet")
    negatives = sum(1 for r in dataset.records if str(r["label_raw"]) == "dry")
    if positives == 0:
        value = 1.0
    else:
        value = max(float(negatives) / float(positives), 1.0)
    print(f"[INFO] auto pos_weight={value:.6f} (neg={negatives}, pos={positives})")
    return torch.tensor([value], dtype=torch.float32, device=device)


def _label_counts(dataset: WetnessPatchDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        label_name = str(record["label_raw"])
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def _score_distribution(y_score: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(y_score)),
        "max": float(np.max(y_score)),
        "mean": float(np.mean(y_score)),
        "std": float(np.std(y_score)),
        "q05": float(np.quantile(y_score, 0.05)),
        "q50": float(np.quantile(y_score, 0.50)),
        "q95": float(np.quantile(y_score, 0.95)),
        "frac_le_0_01": float(np.mean(y_score <= 0.01)),
        "frac_ge_0_99": float(np.mean(y_score >= 0.99)),
        "frac_le_0_05": float(np.mean(y_score <= 0.05)),
        "frac_ge_0_95": float(np.mean(y_score >= 0.95)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


@torch.no_grad()
def evaluate_split(
    wrapper: Any,
    dataset: WetnessPatchDataset,
    batch_size: int,
    device: str,
    threshold: float,
) -> Dict[str, Any]:
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for split={dataset.split}")

    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true: list[float] = []
    y_score: list[float] = []
    samples: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["cube"].to(device)
        ts = build_target_signature(x)

        out = model(x, ts, return_attn=False)
        if isinstance(out, (tuple, list)):
            out = out[0]

        logits = patch_logits_to_sample_logits(out)
        probs = torch.sigmoid(logits)

        y_true.extend(batch["label"].cpu().numpy().astype(float).tolist())
        batch_scores = probs.detach().cpu().numpy().astype(float).tolist()
        y_score.extend(batch_scores)

        for i, score in enumerate(batch_scores):
            label_value = float(batch["label"][i].cpu().item())
            pred_value = int(score >= threshold)
            samples.append(
                {
                    "file": batch["file"][i],
                    "path": batch["path"][i],
                    "sample_id": batch["sample_id"][i],
                    "split": batch["split"][i],
                    "label_name": batch["label_name"][i],
                    "confidence": float(batch["confidence"][i].cpu().item()),
                    "confidence_tier": float(batch["confidence_tier"][i]),
                    "valid_ratio": float(batch["valid_ratio"][i]),
                    "wet_ratio_valid": float(batch["wet_ratio_valid"][i]),
                    "y_true": int(label_value >= 0.5),
                    "y_score": float(score),
                    "y_pred": pred_value,
                }
            )

    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_score_np = np.asarray(y_score, dtype=np.float32)
    y_pred_np = (y_score_np >= threshold).astype(np.int64)

    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    tn = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    fn = int(((y_true_np == 1) & (y_pred_np == 0)).sum())

    roc_auc = float(roc_auc_score(y_true_np, y_score_np)) if len(np.unique(y_true_np)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true_np, y_score_np)) if len(np.unique(y_true_np)) > 1 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))

    return {
        "split": dataset.split,
        "n_samples": int(len(y_true_np)),
        "wet": int((y_true_np == 1).sum()),
        "dry": int((y_true_np == 0).sum()),
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "score_distribution": _score_distribution(y_score_np),
        "samples": samples,
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HyperSIGMA wetness pretrain/fine-tuning")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_type", type=str, choices=["ss", "sa"], default="ss")
    parser.add_argument("--spat_checkpoint", type=str, required=True)
    parser.add_argument("--spec_checkpoint", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=64)

    parser.add_argument("--run_name", type=str, default="debug_run")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        choices=["full", "head_only", "spec_last", "spat_last", "dual_last", "progressive"],
        default="head_only",
    )
    parser.add_argument("--unfreeze_spat_last_n", type=int, default=0)
    parser.add_argument("--unfreeze_spec_last_n", type=int, default=0)

    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["bce", "pos_weight_bce", "confidence_bce", "hybrid_bce"],
        default="bce",
    )
    parser.add_argument("--pos_weight", type=str, default="auto")
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to cpu.")
        device = "cpu"

    run_dir = args.save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WetnessPatchDataset(args.input_dir, split=args.train_split, allow_uncertain=False)
    val_ds = WetnessPatchDataset(args.input_dir, split=args.val_split, allow_uncertain=False)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training patches found for split={args.train_split} in {args.input_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation patches found for split={args.val_split} in {args.input_dir}")

    sample = train_ds[0]
    in_channels = int(sample["cube"].shape[0])
    dataset_info = {
        "manifest_path": str(args.manifest_path.resolve()) if args.manifest_path.exists() else str(args.manifest_path),
        "input_dir": str(args.input_dir.resolve()) if args.input_dir.exists() else str(args.input_dir),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_split": args.eval_split,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_label_counts": _label_counts(train_ds),
        "val_label_counts": _label_counts(val_ds),
    }

    wrapper = load_model(
        model_checkpoint=None,
        device=device,
        in_channels=in_channels,
        patch_size=args.patch_size,
        model_type=args.model_type,
        spat_checkpoint=args.spat_checkpoint,
        spec_checkpoint=args.spec_checkpoint,
    )
    model = wrapper.model if hasattr(wrapper, "model") else wrapper

    apply_freeze_strategy(
        model=model,
        strategy=args.freeze_strategy,
        unfreeze_spat_last_n=args.unfreeze_spat_last_n,
        unfreeze_spec_last_n=args.unfreeze_spec_last_n,
    )
    optimizer = build_optimizer(
        model=model,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )

    if args.pos_weight == "auto":
        pos_weight_tensor = compute_auto_pos_weight(train_ds, device)
    else:
        pos_weight_tensor = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)

    save_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "device": device,
            "dataset_info": dataset_info,
            "load_info": getattr(wrapper, "load_info", {}),
        },
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_ckpt_path = run_dir / "model_best.pt"
    last_ckpt_path = run_dir / "model_last.pt"

    for epoch in range(1, args.epochs + 1):
        if args.freeze_strategy == "progressive":
            if epoch == 1:
                apply_freeze_strategy(model, "head_only", 0, 0)
                optimizer = build_optimizer(model, args.head_lr, 0.0, args.weight_decay)
            elif epoch == max(2, args.epochs // 2):
                apply_freeze_strategy(
                    model,
                    "dual_last",
                    args.unfreeze_spat_last_n,
                    args.unfreeze_spec_last_n,
                )
                optimizer = build_optimizer(model, args.head_lr, args.encoder_lr, args.weight_decay)

        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            x = batch["cube"].to(device)
            y = batch["label"].to(device)
            c = batch["confidence"].to(device)

            ts = build_target_signature(x)
            out = model(x, ts, return_attn=False)
            if isinstance(out, (tuple, list)):
                out = out[0]

            logits = patch_logits_to_sample_logits(out)

            if args.loss_type == "bce":
                loss = plain_bce(logits, y, pos_weight=None)
            elif args.loss_type == "pos_weight_bce":
                loss = plain_bce(logits, y, pos_weight=pos_weight_tensor)
            elif args.loss_type == "confidence_bce":
                loss = confidence_weighted_bce(logits, y, c, pos_weight=None)
            elif args.loss_type == "hybrid_bce":
                loss = confidence_weighted_bce(logits, y, c, pos_weight=pos_weight_tensor)
            else:
                raise ValueError(f"Unsupported loss_type: {args.loss_type}")

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_split(
            wrapper=wrapper,
            dataset=val_ds,
            batch_size=args.batch_size,
            device=device,
            threshold=0.5,
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_metrics["dataset_info"] = {
            **dataset_info,
            "requested_split": args.val_split,
        }
        val_metrics["model_info"] = {
            "model_type": args.model_type,
            "model_checkpoint": None,
            "spat_checkpoint": args.spat_checkpoint,
            "spec_checkpoint": args.spec_checkpoint,
            "device": device,
            "patch_size": args.patch_size,
            "in_channels": in_channels,
        }
        score = (
            (0.0 if np.isnan(val_metrics["roc_auc"]) else val_metrics["roc_auc"]) +
            (0.0 if np.isnan(val_metrics["pr_auc"]) else val_metrics["pr_auc"])
        ) / 2.0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "score": score,
        }
        history.append(row)

        print(
            f"[INFO] epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_roc_auc={val_metrics['roc_auc']:.6f} "
            f"val_pr_auc={val_metrics['pr_auc']:.6f} "
            f"val_f1={val_metrics['f1']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "dataset_info": dataset_info,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "load_info": getattr(wrapper, "load_info", {}),
        }
        torch.save(ckpt, last_ckpt_path)

        if score > best_score:
            best_score = score
            torch.save(ckpt, best_ckpt_path)
            save_json(run_dir / "best_val_metrics.json", val_metrics)

    save_json(
        run_dir / "train_history.json",
        {
            "history": history,
            "dataset_info": dataset_info,
            "device": device,
            "best_score": best_score,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
        },
    )

    print(f"[INFO] saved best checkpoint to {best_ckpt_path}")
    print(f"[INFO] saved last checkpoint to {last_ckpt_path}")
    print(f"[INFO] run directory: {run_dir}")


if __name__ == "__main__":
    main()
