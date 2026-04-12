from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--hypersigma_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    baseline = load_json(args.baseline_json)
    hypersigma = load_json(args.hypersigma_json)

    keys = [
        "model_name",
        "split",
        "threshold",
        "n_samples",
        "n_wet",
        "n_dry",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    ]

    rows = []
    for obj, name in [(baseline, "baseline"), (hypersigma, "hypersigma")]:
        row = {k: obj.get(k) for k in keys}
        row["model_name"] = name
        rows.append(row)

    summary = {
        "baseline": baseline,
        "hypersigma": hypersigma,
        "comparison_rows": rows,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] comparison json saved to {out_json}")
    print(f"[INFO] comparison csv saved to {out_csv}")


if __name__ == "__main__":
    main()
