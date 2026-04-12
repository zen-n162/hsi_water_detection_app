from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_json", required=True, help="eval_*.json with samples[]")
    ap.add_argument("--criterion", default="f1", choices=["f1"])
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    obj = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {args.metrics_json}")

    y_true = np.asarray([int(s["y_true"]) for s in samples], dtype=np.int64)
    y_score = np.asarray([float(s["y_score"]) for s in samples], dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Threshold selection on val requires both classes in y_true")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # thresholds has length n, precision/recall have length n+1
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    denom = precision_t + recall_t
    f1 = np.where(denom > 0, 2.0 * precision_t * recall_t / denom, 0.0)

    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])

    out = {
        "source_metrics_json": args.metrics_json,
        "criterion": args.criterion,
        "best_threshold": best_threshold,
        "best_f1": float(f1[best_idx]),
        "precision_at_best": float(precision_t[best_idx]),
        "recall_at_best": float(recall_t[best_idx]),
        "n_thresholds": int(len(thresholds)),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] best_threshold={out['best_threshold']:.6f}")
    print(f"[INFO] best_f1={out['best_f1']:.6f}")
    print(f"[INFO] precision_at_best={out['precision_at_best']:.6f}")
    print(f"[INFO] recall_at_best={out['recall_at_best']:.6f}")
    print(f"[INFO] saved to {out_path}")


if __name__ == "__main__":
    main()
