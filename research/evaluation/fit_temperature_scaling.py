from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.evaluation.metric_utils import compute_binary_metrics, format_histogram, safe_float, sigmoid_from_logits, temperature_search_grid


def _load_logits_and_labels(metrics_json: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(metrics_json.read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {metrics_json}")

    logits: list[float] = []
    labels: list[int] = []
    for sample in samples:
        if "y_logit_mean" in sample:
            logits.append(safe_float(sample["y_logit_mean"], default=0.0))
        elif "y_logit" in sample:
            logits.append(safe_float(sample["y_logit"], default=0.0))
        else:
            raise RuntimeError(f"Sample does not contain y_logit_mean/y_logit in {metrics_json}")
        labels.append(int(sample["y_true"]))

    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling on a validation metrics JSON")
    parser.add_argument("--metrics_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--log_temp_min", type=float, default=-4.0)
    parser.add_argument("--log_temp_max", type=float, default=4.0)
    parser.add_argument("--grid_steps", type=int, default=801)
    args = parser.parse_args()

    logits, y_true = _load_logits_and_labels(args.metrics_json)

    temperature, best_nll = temperature_search_grid(
        logits=logits,
        y_true=y_true,
        log_temp_min=args.log_temp_min,
        log_temp_max=args.log_temp_max,
        n_steps=args.grid_steps,
    )

    uncal_scores = sigmoid_from_logits(logits, temperature=1.0)
    cal_scores = sigmoid_from_logits(logits, temperature=temperature)

    before = compute_binary_metrics(
        y_true=y_true,
        y_score=uncal_scores,
        threshold=args.threshold,
        n_bins=args.ece_bins,
    )
    _ = before.pop("y_pred")
    after = compute_binary_metrics(
        y_true=y_true,
        y_score=cal_scores,
        threshold=args.threshold,
        n_bins=args.ece_bins,
    )
    _ = after.pop("y_pred")

    out = {
        "source_metrics_json": str(args.metrics_json.resolve()),
        "best_temperature": float(temperature),
        "best_nll": float(best_nll),
        "fit_split": "val",
        "threshold_for_report": float(args.threshold),
        "n_samples": int(len(y_true)),
        "before": {
            **before,
            "score_histogram": format_histogram(uncal_scores, bins=10),
        },
        "after": {
            **after,
            "score_histogram": format_histogram(cal_scores, bins=10),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] best_temperature={out['best_temperature']:.6f}")
    print(f"[INFO] best_nll={out['best_nll']:.6f}")
    print(f"[INFO] before_f1={out['before']['f1']:.6f} after_f1={out['after']['f1']:.6f}")
    print(f"[INFO] before_brier={out['before']['brier_score']:.6f} after_brier={out['after']['brier_score']:.6f}")
    print(f"[INFO] before_ece={out['before']['ece']:.6f} after_ece={out['after']['ece']:.6f}")
    print(f"[INFO] saved to {args.output_json}")


if __name__ == "__main__":
    main()
