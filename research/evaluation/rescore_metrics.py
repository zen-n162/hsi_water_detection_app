from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.evaluation.metric_utils import compute_binary_metrics, format_histogram, safe_float, sigmoid_from_logits


def _resolve_temperature(
    temperature: float | None,
    temperature_json: Path | None,
) -> tuple[float, str | None]:
    if temperature_json is not None:
        obj = json.loads(temperature_json.read_text(encoding="utf-8"))
        value = safe_float(
            obj.get("best_temperature", obj.get("temperature", 1.0)),
            default=1.0,
        )
        return value, str(temperature_json.resolve())
    if temperature is not None:
        return float(temperature), None
    return 1.0, None


def _extract_logits(samples: list[dict[str, Any]]) -> np.ndarray:
    logits: list[float] = []
    for sample in samples:
        if "y_logit_mean" in sample:
            logits.append(safe_float(sample["y_logit_mean"], default=0.0))
        elif "y_logit" in sample:
            logits.append(safe_float(sample["y_logit"], default=0.0))
        else:
            raise RuntimeError("Samples do not contain y_logit_mean/y_logit")
    return np.asarray(logits, dtype=np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics from stored logits with optional temperature scaling")
    parser.add_argument("--metrics_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature_json", type=Path, default=None)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--mode_label", type=str, default=None)
    args = parser.parse_args()

    obj = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {args.metrics_json}")

    temperature_value, temperature_source = _resolve_temperature(
        args.temperature,
        args.temperature_json,
    )
    logits = _extract_logits(samples)
    y_true = np.asarray([int(sample["y_true"]) for sample in samples], dtype=np.int64)
    y_score = sigmoid_from_logits(logits, temperature=temperature_value)
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=args.threshold,
        n_bins=args.ece_bins,
    )
    y_pred = metrics.pop("y_pred")

    new_samples: list[dict[str, Any]] = []
    for sample, logit, score, pred in zip(samples, logits.tolist(), y_score.tolist(), y_pred):
        row = dict(sample)
        if "y_logit_mean" not in row and "y_logit" not in row:
            row["y_logit_mean"] = float(logit)
        row["y_score"] = float(score)
        row["y_pred"] = int(pred)
        new_samples.append(row)

    out = {
        **obj,
        "threshold": float(args.threshold),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "score_distribution": metrics["score_distribution"],
        "score_histogram": format_histogram(y_score, bins=10),
        "brier_score": metrics["brier_score"],
        "ece": metrics["ece"],
        "samples": new_samples,
        "calibration_info": {
            "mode": args.mode_label or ("temperature_scaled" if temperature_value != 1.0 or temperature_source else "uncalibrated"),
            "temperature": float(temperature_value),
            "temperature_json": temperature_source,
            "source_metrics_json": str(args.metrics_json.resolve()),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_json_ready(out), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] temperature={temperature_value:.6f}")
    print(f"[INFO] threshold={args.threshold:.6f}")
    print(f"[INFO] f1={out['f1']:.6f} brier={out['brier_score']:.6f} ece={out['ece']:.6f}")
    print(f"[INFO] saved to {args.output_json}")


if __name__ == "__main__":
    main()
