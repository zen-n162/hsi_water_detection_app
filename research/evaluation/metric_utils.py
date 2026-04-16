from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_distribution(y_score: np.ndarray) -> dict[str, float]:
    y_score = np.asarray(y_score, dtype=np.float64)
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


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_score))
    ece = 0.0

    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)

        if not np.any(mask):
            continue

        conf = float(np.mean(y_score[mask]))
        acc = float(np.mean(y_true[mask]))
        frac = float(np.mean(mask))
        ece += frac * abs(acc - conf)

    return float(ece)


def sigmoid_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = logits / float(temperature)
    scaled = np.clip(scaled, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def compute_binary_metrics(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bins: int = 10,
) -> dict[str, Any]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    y_pred_np = (y_score_np >= threshold).astype(np.int64)

    n_pos = int((y_true_np == 1).sum())
    n_neg = int((y_true_np == 0).sum())

    roc_auc = None
    pr_auc = None
    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true_np, y_score_np))
        pr_auc = float(average_precision_score(y_true_np, y_score_np))

    precision = float(precision_score(y_true_np, y_pred_np, zero_division=0))
    recall = float(recall_score(y_true_np, y_pred_np, zero_division=0))
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).tolist()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "score_distribution": score_distribution(y_score_np),
        "brier_score": brier_score(y_true_np, y_score_np),
        "ece": expected_calibration_error(y_true_np, y_score_np, n_bins=n_bins),
        "y_pred": y_pred_np.tolist(),
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def format_histogram(values: np.ndarray, bins: int = 10) -> dict[str, Any]:
    hist, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, range=(0.0, 1.0))
    return {
        "counts": hist.astype(int).tolist(),
        "edges": [float(x) for x in edges.tolist()],
    }


def temperature_search_grid(
    *,
    logits: np.ndarray,
    y_true: np.ndarray,
    log_temp_min: float = -4.0,
    log_temp_max: float = 4.0,
    n_steps: int = 801,
) -> tuple[float, float]:
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    best_temp = 1.0
    best_nll = math.inf

    for log_temp in np.linspace(log_temp_min, log_temp_max, n_steps):
        temp = float(np.exp(log_temp))
        probs = sigmoid_from_logits(logits, temperature=temp)
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        nll = float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    return best_temp, best_nll
