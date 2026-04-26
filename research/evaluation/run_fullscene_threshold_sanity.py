from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
import numpy as np

from hsi_water_detection_app.config import HYPERION_BAD_BANDS_0BASED
from hsi_water_detection_app.data.loader import load_hsi_cube
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands
from hsi_water_detection_app.inference.patch_infer import generate_patches
from hsi_water_detection_app.inference.reconstruct import reconstruct_from_patches
from hsi_water_detection_app.models.hyper_sigma import load_model
from research.evaluation.metric_utils import compute_binary_metrics, score_distribution


DEFAULT_INPUT = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/hyperion/hyperion_stack_crop_f16.tif"
)
DEFAULT_CONFIDENCE = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/labels/confidence_ali.tif"
)
DEFAULT_VALID_MASK = Path(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/"
    "Hyperion_WaterLabel_20111222/data/processed/masks/final_valid_mask_clean_ali_aoi90.tif"
)


def _read_raster(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def _load_temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj.get("best_temperature", obj.get("temperature", 1.0)))


def _load_threshold(path: Path | None) -> float:
    if path is None:
        return 0.5
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj.get("best_threshold", obj.get("threshold", 0.5)))


def _positive_rate_threshold(scores: np.ndarray, target_rate: float) -> float:
    """Return a conservative threshold whose positive rate should not exceed target_rate."""
    finite_scores = np.asarray(scores, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return 1.0
    target = float(np.clip(target_rate, 0.0, 1.0))
    if target <= 0.0:
        return float(np.nextafter(np.max(finite_scores), np.inf))
    if target >= 1.0:
        return float(np.min(finite_scores))

    sorted_scores = np.sort(finite_scores)[::-1]
    kth = max(1, int(np.ceil(target * sorted_scores.size))) - 1
    kth_score = float(sorted_scores[min(kth, sorted_scores.size - 1)])
    predicted_rate = float(np.mean(finite_scores >= kth_score))
    if predicted_rate <= target + 1e-12:
        return kth_score
    return float(np.nextafter(kth_score, np.inf))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _stable_seed(*parts: Any, base_seed: int) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**32)


def _binary_metrics_from_prediction(labels: np.ndarray, pred_positive: np.ndarray) -> dict[str, float | list[list[int]]]:
    labels_bool = np.asarray(labels).astype(bool)
    pred_bool = np.asarray(pred_positive).astype(bool)
    tp = int(np.sum(pred_bool & labels_bool))
    fp = int(np.sum(pred_bool & ~labels_bool))
    fn = int(np.sum(~pred_bool & labels_bool))
    tn = int(np.sum(~pred_bool & ~labels_bool))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predicted_positive_rate": float(np.mean(pred_bool)) if pred_bool.size else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _random_same_area_baseline(
    *,
    labels: np.ndarray,
    predicted_positive_rate: float,
    repeats: int,
    seed: int,
) -> dict[str, Any] | None:
    labels = np.asarray(labels).astype(np.int64)
    n = int(labels.size)
    if n <= 0 or repeats <= 0:
        return None

    k = int(round(float(np.clip(predicted_positive_rate, 0.0, 1.0)) * n))
    k = max(0, min(k, n))
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, float | list[list[int]]]] = []
    for _ in range(int(repeats)):
        pred = np.zeros(n, dtype=bool)
        if k > 0:
            pred[rng.choice(n, size=k, replace=False)] = True
        metrics.append(_binary_metrics_from_prediction(labels, pred))

    def _series(name: str) -> np.ndarray:
        return np.asarray([float(item[name]) for item in metrics], dtype=np.float64)

    f1_values = _series("f1")
    precision_values = _series("precision")
    recall_values = _series("recall")
    positive_rate_values = _series("predicted_positive_rate")
    return {
        "kind": "random_same_area",
        "repeats": int(repeats),
        "seed": int(seed),
        "selected_pixels": k,
        "predicted_positive_rate_mean": float(positive_rate_values.mean()),
        "f1_mean": float(f1_values.mean()),
        "f1_std": float(f1_values.std()),
        "precision_mean": float(precision_values.mean()),
        "precision_std": float(precision_values.std()),
        "recall_mean": float(recall_values.mean()),
        "recall_std": float(recall_values.std()),
    }


def _run_prevalence(run: dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = run.get(name)
    if value is not None:
        return float(value)
    if name == "train_prevalence" and run.get("train_wet") is not None and run.get("train_dry") is not None:
        denom = float(run["train_wet"] + run["train_dry"])
        return None if denom <= 0 else float(run["train_wet"] / denom)
    if name == "validation_prevalence" and run.get("val_wet") is not None and run.get("val_dry") is not None:
        denom = float(run["val_wet"] + run["val_dry"])
        return None if denom <= 0 else float(run["val_wet"] / denom)
    return default


def _build_thresholds(
    *,
    scores: np.ndarray,
    val_f1_threshold: float,
    run: dict[str, Any],
    area_caps: list[float],
) -> dict[str, dict[str, float | None | str]]:
    thresholds: dict[str, dict[str, float | None | str]] = {
        "fixed_0_5": {
            "threshold": 0.5,
            "target_positive_rate": None,
            "policy_type": "fixed",
        },
        "val_f1_tuned": {
            "threshold": float(val_f1_threshold),
            "target_positive_rate": None,
            "policy_type": "patch_validation_f1",
        },
    }

    train_prev = _run_prevalence(run, "train_prevalence")
    if train_prev is not None:
        thresholds["prior_aware_train_prevalence"] = {
            "threshold": _positive_rate_threshold(scores, train_prev),
            "target_positive_rate": float(train_prev),
            "policy_type": "fullscene_positive_rate_target",
        }

    val_prev = _run_prevalence(run, "validation_prevalence")
    if val_prev is not None:
        thresholds["validation_prevalence_target"] = {
            "threshold": _positive_rate_threshold(scores, val_prev),
            "target_positive_rate": float(val_prev),
            "policy_type": "fullscene_positive_rate_target",
        }

    for area_cap in area_caps:
        cap = float(np.clip(area_cap, 0.0, 1.0))
        cap_name = f"area_cap_{int(round(cap * 100)):02d}"
        thresholds[cap_name] = {
            "threshold": _positive_rate_threshold(scores, cap),
            "target_positive_rate": cap,
            "policy_type": "fullscene_positive_rate_cap",
        }
    return thresholds


def _infer_probability_map(
    *,
    cube: np.ndarray,
    model: Any,
    patch_size: int,
    stride: int,
    temperature: float,
) -> np.ndarray:
    patches = generate_patches(cube, patch_size=patch_size, stride=stride)
    patch_outputs: list[dict[str, Any]] = []
    for item in patches:
        result = model.infer_patch(
            item["patch"],
            return_attn=False,
            temperature=temperature,
        )
        patch_outputs.append(
            {
                "top": item["top"],
                "left": item["left"],
                "height": item["height"],
                "width": item["width"],
                "prob_map": result["prob_map"],
            }
        )

    out = reconstruct_from_patches(
        patch_outputs,
        image_shape=(cube.shape[1], cube.shape[2]),
        patch_size=patch_size,
        stride=stride,
    )
    if out is None:
        raise RuntimeError("full-scene probability reconstruction returned None")
    return out.astype(np.float32)


def _select_patch_subset(patches: list[dict[str, Any]], max_patches: int | None) -> list[dict[str, Any]]:
    if max_patches is None or max_patches <= 0 or len(patches) <= max_patches:
        return patches
    indices = np.linspace(0, len(patches) - 1, num=max_patches, dtype=int)
    return [patches[int(i)] for i in indices]


def _infer_sampled_scores(
    *,
    cube: np.ndarray,
    model: Any,
    patch_size: int,
    stride: int,
    temperature: float,
    y_true: np.ndarray,
    valid: np.ndarray,
    max_patches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    patches = generate_patches(cube, patch_size=patch_size, stride=stride)
    sampled = _select_patch_subset(patches, max_patches)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    valid_pixels = 0
    for idx, item in enumerate(sampled):
        result = model.infer_patch(
            item["patch"],
            return_attn=False,
            temperature=temperature,
        )
        top = int(item["top"])
        left = int(item["left"])
        h = int(item["height"])
        w = int(item["width"])
        patch_valid = valid[top : top + h, left : left + w]
        if not np.any(patch_valid):
            continue
        patch_labels = y_true[top : top + h, left : left + w][patch_valid]
        patch_scores = result["prob_map"][patch_valid]
        local_rows, local_cols = np.nonzero(patch_valid)
        labels.append(patch_labels.astype(np.int64))
        scores.append(patch_scores.astype(np.float64))
        rows.append((local_rows + top).astype(np.int32))
        cols.append((local_cols + left).astype(np.int32))
        valid_pixels += int(patch_valid.sum())
        if (idx + 1) % 25 == 0:
            print(f"[INFO] sampled inference progress {idx + 1}/{len(sampled)} patches")
    if not scores:
        raise RuntimeError("No valid pixels found in sampled full-scene patches")
    meta = {
        "total_generated_patches": len(patches),
        "sampled_patches": len(sampled),
        "sampled_valid_pixels": valid_pixels,
    }
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(rows), np.concatenate(cols), meta


def _map_scores(prob_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return prob_map[valid].astype(np.float64)


def _summarize_map(
    *,
    name: str,
    prob_map: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    thresholds: dict[str, dict[str, float | None | str]],
    random_baseline_repeats: int,
    random_baseline_seed_base: int,
    run_seed: str,
) -> dict[str, Any]:
    scores = prob_map[valid].astype(np.float64)
    labels = y_true[valid].astype(np.int64)
    out: dict[str, Any] = {
        "name": name,
        "valid_pixels": int(valid.sum()),
        "wet_pixels": int(labels.sum()),
        "dry_pixels": int((labels == 0).sum()),
        "wet_prevalence": float(labels.mean()) if labels.size else None,
        "score_distribution": score_distribution(scores),
        "thresholds": {},
    }
    for threshold_name, threshold_obj in thresholds.items():
        threshold = float(threshold_obj["threshold"])
        metrics = compute_binary_metrics(
            y_true=labels,
            y_score=scores,
            threshold=threshold,
        )
        predicted_positive_rate = float(np.mean(scores >= threshold))
        random_baseline = _random_same_area_baseline(
            labels=labels,
            predicted_positive_rate=predicted_positive_rate,
            repeats=random_baseline_repeats,
            seed=_stable_seed(run_seed, name, threshold_name, threshold, base_seed=random_baseline_seed_base),
        )
        out["thresholds"][threshold_name] = {
            "threshold": threshold,
            "target_positive_rate": threshold_obj.get("target_positive_rate"),
            "policy_type": threshold_obj.get("policy_type"),
            "predicted_positive_rate": predicted_positive_rate,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "random_same_area_baseline": random_baseline,
            "f1_minus_random_mean": (
                None if random_baseline is None else float(metrics["f1"] - random_baseline["f1_mean"])
            ),
        }
    return out


def _summarize_scores(
    *,
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, dict[str, float | None | str]],
    sample_meta: dict[str, Any],
    random_baseline_repeats: int,
    random_baseline_seed_base: int,
    run_seed: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "valid_pixels": int(labels.size),
        "wet_pixels": int(labels.sum()),
        "dry_pixels": int((labels == 0).sum()),
        "wet_prevalence": float(labels.mean()) if labels.size else None,
        "score_distribution": score_distribution(scores),
        "sample_meta": sample_meta,
        "thresholds": {},
    }
    for threshold_name, threshold_obj in thresholds.items():
        threshold = float(threshold_obj["threshold"])
        metrics = compute_binary_metrics(
            y_true=labels,
            y_score=scores,
            threshold=threshold,
        )
        predicted_positive_rate = float(np.mean(scores >= threshold))
        random_baseline = _random_same_area_baseline(
            labels=labels,
            predicted_positive_rate=predicted_positive_rate,
            repeats=random_baseline_repeats,
            seed=_stable_seed(run_seed, name, threshold_name, threshold, base_seed=random_baseline_seed_base),
        )
        out["thresholds"][threshold_name] = {
            "threshold": threshold,
            "target_positive_rate": threshold_obj.get("target_positive_rate"),
            "policy_type": threshold_obj.get("policy_type"),
            "predicted_positive_rate": predicted_positive_rate,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "random_same_area_baseline": random_baseline,
            "f1_minus_random_mean": (
                None if random_baseline is None else float(metrics["f1"] - random_baseline["f1_mean"])
            ),
        }
    return out


def _save_figures(output_dir: Path, seed: str, raw_map: np.ndarray, calibrated_map: np.ndarray, valid: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [("raw_probability", raw_map), ("calibrated_probability", calibrated_map)]:
        plt.figure(figsize=(6, 5))
        masked = np.where(valid, arr, np.nan)
        plt.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(label="wet probability")
        plt.title(f"{seed} {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{seed}_{name}.png", dpi=160)
        plt.close()


def run_one(args: argparse.Namespace, run: dict[str, Any], cube: np.ndarray, y_true: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    seed = str(run["seed"])
    checkpoint = Path(run["model_checkpoint"])
    threshold_path = Path(run["threshold_json"]) if run.get("threshold_json") else None
    temperature_path = Path(run["temperature_json"]) if run.get("temperature_json") else None
    threshold = _load_threshold(threshold_path)
    temperature = _load_temperature(temperature_path)

    model = load_model(
        model_checkpoint=str(checkpoint),
        device=args.device,
        in_channels=cube.shape[0],
        patch_size=args.patch_size,
        model_type=args.model_type,
    )

    seed_dir = args.output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    if args.max_patches and args.max_patches > 0:
        raw_scores, raw_labels, raw_rows, raw_cols, raw_meta = _infer_sampled_scores(
            cube=cube,
            model=model,
            patch_size=args.patch_size,
            stride=args.stride,
            temperature=1.0,
            y_true=y_true,
            valid=valid,
            max_patches=args.max_patches,
        )
        calibrated_scores, calibrated_labels, calibrated_rows, calibrated_cols, calibrated_meta = _infer_sampled_scores(
            cube=cube,
            model=model,
            patch_size=args.patch_size,
            stride=args.stride,
            temperature=temperature,
            y_true=y_true,
            valid=valid,
            max_patches=args.max_patches,
        )
        raw_thresholds = _build_thresholds(
            scores=raw_scores,
            val_f1_threshold=threshold,
            run=run,
            area_caps=args.area_caps,
        )
        calibrated_thresholds = _build_thresholds(
            scores=calibrated_scores,
            val_f1_threshold=threshold,
            run=run,
            area_caps=args.area_caps,
        )
        np.savez_compressed(
            seed_dir / "sampled_scores.npz",
            raw_scores=raw_scores,
            raw_labels=raw_labels,
            rows=raw_rows,
            cols=raw_cols,
        )
        np.savez_compressed(
            seed_dir / "sampled_scores_calibrated.npz",
            calibrated_scores=calibrated_scores,
            calibrated_labels=calibrated_labels,
            rows=calibrated_rows,
            cols=calibrated_cols,
        )
        out = {
            "seed": seed,
            "mode": "sampled_fullscene_coverage",
            "model_checkpoint": str(checkpoint),
            "temperature_json": None if temperature_path is None else str(temperature_path),
            "best_val_threshold_json": None if threshold_path is None else str(threshold_path),
            "temperature": temperature,
            "val_f1_tuned_threshold": threshold,
            "train_prevalence": _run_prevalence(run, "train_prevalence"),
            "validation_prevalence": _run_prevalence(run, "validation_prevalence"),
            "area_caps": args.area_caps,
            "raw": _summarize_scores(
                name="raw",
                scores=raw_scores,
                labels=raw_labels,
                thresholds=raw_thresholds,
                sample_meta=raw_meta,
                random_baseline_repeats=args.random_baseline_repeats,
                random_baseline_seed_base=args.random_baseline_seed,
                run_seed=seed,
            ),
            "calibrated": _summarize_scores(
                name="calibrated",
                scores=calibrated_scores,
                labels=calibrated_labels,
                thresholds=calibrated_thresholds,
                sample_meta=calibrated_meta,
                random_baseline_repeats=args.random_baseline_repeats,
                random_baseline_seed_base=args.random_baseline_seed,
                run_seed=seed,
            ),
        }
        (seed_dir / "fullscene_threshold_sanity.json").write_text(
            json.dumps(out, indent=2),
            encoding="utf-8",
        )
        return out

    raw_map = _infer_probability_map(
        cube=cube,
        model=model,
        patch_size=args.patch_size,
        stride=args.stride,
        temperature=1.0,
    )
    calibrated_map = _infer_probability_map(
        cube=cube,
        model=model,
        patch_size=args.patch_size,
        stride=args.stride,
        temperature=temperature,
    )

    np.save(seed_dir / "raw_probability_map.npy", raw_map)
    np.save(seed_dir / "calibrated_probability_map.npy", calibrated_map)
    _save_figures(seed_dir, f"seed{seed}", raw_map, calibrated_map, valid)
    raw_thresholds = _build_thresholds(
        scores=_map_scores(raw_map, valid),
        val_f1_threshold=threshold,
        run=run,
        area_caps=args.area_caps,
    )
    calibrated_thresholds = _build_thresholds(
        scores=_map_scores(calibrated_map, valid),
        val_f1_threshold=threshold,
        run=run,
        area_caps=args.area_caps,
    )

    out = {
        "seed": seed,
        "mode": "fullscene_exhaustive",
        "model_checkpoint": str(checkpoint),
        "temperature_json": None if temperature_path is None else str(temperature_path),
        "best_val_threshold_json": None if threshold_path is None else str(threshold_path),
        "temperature": temperature,
        "val_f1_tuned_threshold": threshold,
        "train_prevalence": _run_prevalence(run, "train_prevalence"),
        "validation_prevalence": _run_prevalence(run, "validation_prevalence"),
        "area_caps": args.area_caps,
        "raw": _summarize_map(
            name="raw",
            prob_map=raw_map,
            y_true=y_true,
            valid=valid,
            thresholds=raw_thresholds,
            random_baseline_repeats=args.random_baseline_repeats,
            random_baseline_seed_base=args.random_baseline_seed,
            run_seed=seed,
        ),
        "calibrated": _summarize_map(
            name="calibrated",
            prob_map=calibrated_map,
            y_true=y_true,
            valid=valid,
            thresholds=calibrated_thresholds,
            random_baseline_repeats=args.random_baseline_repeats,
            random_baseline_seed_base=args.random_baseline_seed,
            run_seed=seed,
        ),
    }
    (seed_dir / "fullscene_threshold_sanity.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out


def _load_runs(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if "runs" in obj:
        return list(obj["runs"])
    raise ValueError(f"Unsupported runs JSON schema: {path}")


def _load_split_repeat_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows", obj if isinstance(obj, list) else [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if seed is not None:
            out[str(seed)] = row
    return out


def _merge_repeat_metadata(runs: list[dict[str, Any]], repeat_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for run in runs:
        seed = str(run["seed"])
        repeat = repeat_rows.get(seed, {})
        enriched = dict(run)
        for src, dst in [
            ("train_wet", "train_wet"),
            ("train_dry", "train_dry"),
            ("val_wet", "val_wet"),
            ("val_dry", "val_dry"),
            ("test_wet", "test_wet"),
            ("test_dry", "test_dry"),
        ]:
            if dst not in enriched and src in repeat:
                enriched[dst] = repeat[src]
        merged.append(enriched)
    return merged


def _parse_area_caps(value: str | None, fallback: float) -> list[float]:
    if value is None or value.strip() == "":
        return [float(fallback)]
    caps = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        caps.append(float(np.clip(float(item), 0.0, 1.0)))
    return sorted(set(caps))


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fullscene_threshold_sanity_summary.json"
    csv_path = output_dir / "fullscene_threshold_sanity_summary.csv"
    json_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        for mode in [name for name in ("raw", "calibrated") if name in row]:
            mode_obj = row[mode]
            dist = mode_obj["score_distribution"]
            for policy, metrics in mode_obj["thresholds"].items():
                random_baseline = metrics.get("random_same_area_baseline") or {}
                flat_rows.append(
                    {
                        "model": row.get("model_name", "hypersigma"),
                        "seed": row["seed"],
                        "mode": mode,
                        "policy": policy,
                        "policy_type": metrics.get("policy_type"),
                        "temperature": row["temperature"],
                        "threshold": metrics["threshold"],
                        "target_positive_rate": metrics.get("target_positive_rate"),
                        "predicted_positive_rate": metrics["predicted_positive_rate"],
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "random_f1_mean": random_baseline.get("f1_mean"),
                        "random_f1_std": random_baseline.get("f1_std"),
                        "random_precision_mean": random_baseline.get("precision_mean"),
                        "random_recall_mean": random_baseline.get("recall_mean"),
                        "f1_minus_random_mean": metrics.get("f1_minus_random_mean"),
                        "score_min": dist["min"],
                        "score_max": dist["max"],
                        "score_mean": dist["mean"],
                        "score_std": dist["std"],
                        "frac_ge_0_99": dist["frac_ge_0_99"],
                        "frac_le_0_01": dist["frac_le_0_01"],
                        "valid_pixels": mode_obj["valid_pixels"],
                        "wet_prevalence": mode_obj["wet_prevalence"],
                        "train_prevalence": row.get("train_prevalence"),
                        "validation_prevalence": row.get("validation_prevalence"),
                        "area_caps": ",".join(str(x) for x in row.get("area_caps", [])),
                    }
                )
    if not flat_rows:
        raise RuntimeError("No summary rows were produced")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[INFO] saved {json_path}")
    print(f"[INFO] saved {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-scene threshold sanity checks for selected HyperSIGMA runs")
    parser.add_argument("--runs_json", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--confidence_tif", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--valid_mask_tif", type=Path, default=DEFAULT_VALID_MASK)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--sensor", type=str, default="hyperion")
    parser.add_argument("--model_type", type=str, default="ss", choices=["ss", "sa"])
    parser.add_argument(
        "--max_patches",
        type=int,
        default=0,
        help="If >0, run a uniform full-scene patch sample instead of exhaustive reconstruction.",
    )
    parser.add_argument(
        "--split_repeat_summary_json",
        type=Path,
        default=None,
        help="Optional split repeat summary JSON used to add train/validation prevalence targets by seed.",
    )
    parser.add_argument(
        "--area_cap",
        type=float,
        default=0.10,
        help="Conservative full-scene predicted-positive-rate cap used as an operating policy.",
    )
    parser.add_argument(
        "--area_caps",
        type=str,
        default=None,
        help="Comma-separated conservative full-scene positive-rate caps, e.g. 0.05,0.10,0.15,0.20.",
    )
    parser.add_argument(
        "--random_baseline_repeats",
        type=int,
        default=0,
        help="If >0, compare each policy against this many random same-area predictions.",
    )
    parser.add_argument(
        "--random_baseline_seed",
        type=int,
        default=20260426,
        help="Base RNG seed for deterministic random same-area baselines.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.area_caps = _parse_area_caps(args.area_caps, args.area_cap)
    cube, _ = load_hsi_cube(str(args.input), allow_dummy=False, sensor=args.sensor)
    if args.sensor == "hyperion" and cube.shape[0] == 242:
        cube = remove_bad_bands(cube, bad_band_indices=HYPERION_BAD_BANDS_0BASED)
    cube = normalize_cube(cube)

    confidence = _read_raster(args.confidence_tif)
    valid_mask = _read_raster(args.valid_mask_tif)
    if confidence.shape != valid_mask.shape:
        raise ValueError(f"confidence/mask shape mismatch: {confidence.shape} vs {valid_mask.shape}")
    if confidence.shape != tuple(cube.shape[1:]):
        raise ValueError(f"label/cube shape mismatch: labels={confidence.shape}, cube={cube.shape}")

    valid = valid_mask == 1
    y_true = np.isin(confidence, [1, 2, 3]).astype(np.int64)

    runs = _merge_repeat_metadata(_load_runs(args.runs_json), _load_split_repeat_rows(args.split_repeat_summary_json))
    rows = []
    for run in runs:
        rows.append(run_one(args, run, cube, y_true, valid))
    _write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
