from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_LABEL_VALUES = {255}
MASK_IGNORE_VALUES = {255}


def _optional_import_rasterio():
    try:
        import rasterio  # type: ignore

        return rasterio
    except ImportError:
        return None


def _as_2d(arr: np.ndarray, *, path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr[0]
    raise ValueError(f"Expected 2D or 3D label/mask array, got shape={arr.shape} for {path}")


def _load_numpy_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    return _as_2d(arr, path=path)


def _read_raster_window(
    path: Path,
    *,
    row_start: int | None,
    row_stop: int | None,
    col_start: int | None,
    col_stop: int | None,
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
) -> np.ndarray:
    rasterio = _optional_import_rasterio()
    if rasterio is None:
        raise ImportError("rasterio is required to read raster label/mask files.")

    with rasterio.open(path) as src:
        if all(v is not None for v in (row_start, row_stop, col_start, col_stop)):
            from rasterio.windows import Window

            window = Window(
                col_off=int(col_start),
                row_off=int(row_start),
                width=int(col_stop) - int(col_start),
                height=int(row_stop) - int(row_start),
            )
            return src.read(1, window=window)

        if all(v is not None for v in (xmin, ymin, xmax, ymax)):
            from rasterio.windows import from_bounds

            window = from_bounds(
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax),
                transform=src.transform,
            )
            window = window.round_offsets().round_lengths()
            return src.read(1, window=window)

        return src.read(1)


def _load_eval_array(
    path_value: str,
    *,
    target_shape: tuple[int, int],
    row_start: int | None,
    row_stop: int | None,
    col_start: int | None,
    col_stop: int | None,
    xmin: float | None,
    ymin: float | None,
    xmax: float | None,
    ymax: float | None,
) -> np.ndarray:
    path = Path(path_value)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = _load_numpy_array(path)
        if arr.shape == target_shape:
            return arr
        raise ValueError(f"{path.name} shape {arr.shape} does not match probability map {target_shape}")

    rasterio = _optional_import_rasterio()
    if rasterio is not None:
        with rasterio.open(path) as src:
            if (src.height, src.width) == target_shape:
                return src.read(1)

    arr = _read_raster_window(
        path,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )
    arr = _as_2d(arr, path=path)
    if arr.shape == target_shape:
        return arr

    full_arr = None
    if rasterio is not None:
        with rasterio.open(path) as src:
            full_arr = src.read(1)
    if full_arr is not None and full_arr.shape == target_shape:
        return full_arr

    raise ValueError(f"{path.name} shape {arr.shape} does not match probability map {target_shape}")


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def mask_to_valid_pixels(mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(mask)
    valid &= mask > 0
    for ignore_value in MASK_IGNORE_VALUES:
        valid &= mask != ignore_value
    return valid


def load_mask_array(
    mask_path: str,
    *,
    target_shape: tuple[int, int],
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
) -> np.ndarray:
    return _load_eval_array(
        mask_path,
        target_shape=target_shape,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def load_label_array(
    label_path: str,
    *,
    target_shape: tuple[int, int],
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
) -> np.ndarray:
    return _load_eval_array(
        label_path,
        target_shape=target_shape,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def apply_probability_mask(
    *,
    probability_map_npy: Path,
    mask_path: str,
    output_json: Path | None = None,
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probability = np.load(probability_map_npy).astype(np.float32)
    if probability.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape={probability.shape}")

    mask = load_mask_array(
        mask_path,
        target_shape=probability.shape,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )
    valid = mask_to_valid_pixels(mask)
    masked_probability = np.where(valid, probability, 0.0).astype(np.float32)

    total_pixels = int(probability.size)
    valid_pixels = int(np.sum(valid))
    masked_pixels = int(total_pixels - valid_pixels)
    summary: dict[str, Any] = {
        "applied": True,
        "mask_path": mask_path,
        "mask_rule": "mask > 0 and not 255 is included in inference output",
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "masked_pixels": masked_pixels,
        "valid_rate": _safe_div(valid_pixels, total_pixels),
        "masked_rate": _safe_div(masked_pixels, total_pixels),
        "masked_visualization_color": "black",
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return masked_probability, valid, summary


def _optional_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    if np.unique(y_true).size < 2:
        return {"roc_auc": None, "pr_auc": None}

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        return {"roc_auc": None, "pr_auc": None}


def compute_label_metrics(
    *,
    probability_map_npy: Path,
    threshold: float,
    label_path: str,
    mask_path: str | None,
    output_json: Path | None = None,
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
) -> dict[str, Any]:
    probability = np.load(probability_map_npy).astype(np.float64)
    if probability.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape={probability.shape}")

    label = load_label_array(
        label_path,
        target_shape=probability.shape,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )

    valid = np.isfinite(label)
    for ignore_value in IGNORE_LABEL_VALUES:
        valid &= label != ignore_value

    if mask_path:
        mask = load_mask_array(
            mask_path,
            target_shape=probability.shape,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        valid &= mask_to_valid_pixels(mask)

    y_score = probability[valid]
    y_label_raw = label[valid]
    y_true = (y_label_raw > 0).astype(np.int64)
    y_pred = (y_score >= float(threshold)).astype(np.int64)

    if y_true.size == 0:
        raise ValueError("No valid pixels remain after applying label ignore values and mask.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = int(y_true.size)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, total)
    iou = _safe_div(tp, tp + fp + fn)
    brier = float(np.mean((y_score - y_true) ** 2))

    auc_metrics = _optional_auc_metrics(y_true, y_score)
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "valid_pixels": total,
        "total_pixels": int(probability.size),
        "ignored_pixels": int(probability.size - total),
        "label_positive_pixels": int(np.sum(y_true == 1)),
        "label_negative_pixels": int(np.sum(y_true == 0)),
        "predicted_positive_pixels": int(np.sum(y_pred == 1)),
        "predicted_negative_pixels": int(np.sum(y_pred == 0)),
        "label_positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "iou": iou,
        "brier_score": brier,
        "roc_auc": auc_metrics["roc_auc"],
        "pr_auc": auc_metrics["pr_auc"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["dry(0)", "wet(1)"],
        "label_path": label_path,
        "mask_path": mask_path,
        "label_rule": "0=dry, values >0 and not 255=wet, 255/NaN=ignore",
        "mask_rule": "mask > 0 and not 255 is valid" if mask_path else None,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    return out
