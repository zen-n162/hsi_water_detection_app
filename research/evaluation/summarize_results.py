from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _sort_desc(v: Any) -> float:
    x = _to_float(v)
    if x != x:
        return float("-inf")
    return x


def _sort_asc(v: Any) -> float:
    x = _to_float(v)
    if x != x:
        return float("inf")
    return x


def _safe_read_comparison_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_run_name_from_file(path: Path) -> str:
    name = path.stem  # model_comparison_E02_head_only_posw_vs_baseline
    prefix = "model_comparison_"
    suffix = "_vs_baseline"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return name


def _pick_model_row(rows: list[dict[str, str]], model_name: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("model_name") == model_name:
            return row
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize multiple comparison CSVs into one ranking CSV")
    ap.add_argument(
        "--input_glob",
        default="experiments/eval_v3/model_comparison_*_vs_baseline.csv",
        help="Glob pattern for comparison CSV files",
    )
    ap.add_argument(
        "--output_csv",
        default="experiments/eval_v3/hypersigma_finetune_ranking.csv",
        help="Output ranking CSV path",
    )
    ap.add_argument(
        "--output_json",
        default="experiments/eval_v3/hypersigma_finetune_ranking.json",
        help="Output ranking JSON path",
    )
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    summary_rows: list[dict[str, Any]] = []

    for path in paths:
        rows = _safe_read_comparison_csv(path)

        baseline = _pick_model_row(rows, "baseline")
        hypersigma = _pick_model_row(rows, "hypersigma")

        if baseline is None or hypersigma is None:
            print(f"[WARN] skip malformed comparison csv: {path}")
            continue

        run_name = _extract_run_name_from_file(path)

        hs_f1 = _to_float(hypersigma.get("f1"))
        hs_pr_auc = _to_float(hypersigma.get("pr_auc"))
        hs_roc_auc = _to_float(hypersigma.get("roc_auc"))

        bl_f1 = _to_float(baseline.get("f1"))
        bl_pr_auc = _to_float(baseline.get("pr_auc"))
        bl_roc_auc = _to_float(baseline.get("roc_auc"))

        row = {
            "run_name": run_name,
            "comparison_csv": str(path),
            "baseline_threshold": baseline.get("threshold", ""),
            "baseline_roc_auc": bl_roc_auc,
            "baseline_pr_auc": bl_pr_auc,
            "baseline_brier_score": _to_float(baseline.get("brier_score")),
            "baseline_ece": _to_float(baseline.get("ece")),
            "baseline_precision": _to_float(baseline.get("precision")),
            "baseline_recall": _to_float(baseline.get("recall")),
            "baseline_f1": bl_f1,
            "baseline_confusion_matrix": baseline.get("confusion_matrix", ""),
            "hypersigma_threshold": hypersigma.get("threshold", ""),
            "hypersigma_roc_auc": hs_roc_auc,
            "hypersigma_pr_auc": hs_pr_auc,
            "hypersigma_brier_score": _to_float(hypersigma.get("brier_score")),
            "hypersigma_ece": _to_float(hypersigma.get("ece")),
            "hypersigma_calibration_mode": hypersigma.get("calibration_mode", ""),
            "hypersigma_precision": _to_float(hypersigma.get("precision")),
            "hypersigma_recall": _to_float(hypersigma.get("recall")),
            "hypersigma_f1": hs_f1,
            "hypersigma_confusion_matrix": hypersigma.get("confusion_matrix", ""),
            "delta_roc_auc": hs_roc_auc - bl_roc_auc,
            "delta_pr_auc": hs_pr_auc - bl_pr_auc,
            "delta_f1": hs_f1 - bl_f1,
            "delta_brier_score": _to_float(hypersigma.get("brier_score")) - _to_float(baseline.get("brier_score")),
            "delta_ece": _to_float(hypersigma.get("ece")) - _to_float(baseline.get("ece")),
            "winner_by_f1": "hypersigma" if hs_f1 > bl_f1 else ("baseline" if hs_f1 < bl_f1 else "tie"),
            "winner_by_pr_auc": "hypersigma" if hs_pr_auc > bl_pr_auc else ("baseline" if hs_pr_auc < bl_pr_auc else "tie"),
            "winner_by_roc_auc": "hypersigma" if hs_roc_auc > bl_roc_auc else ("baseline" if hs_roc_auc < bl_roc_auc else "tie"),
        }
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No valid comparison rows were collected")

    # ランキング基準:
    # 1) hypersigma_f1 降順
    # 2) hypersigma_pr_auc 降順
    # 3) hypersigma_roc_auc 降順
    # 4) hypersigma_brier_score 昇順
    # 5) hypersigma_ece 昇順
    # 6) run_name 昇順
    summary_rows.sort(
        key=lambda r: (
            -_sort_desc(r["hypersigma_f1"]),
            -_sort_desc(r["hypersigma_pr_auc"]),
            -_sort_desc(r["hypersigma_roc_auc"]),
            _sort_asc(r["hypersigma_brier_score"]),
            _sort_asc(r["hypersigma_ece"]),
            str(r["run_name"]),
        )
    )

    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "run_name",
        "comparison_csv",
        "baseline_threshold",
        "baseline_roc_auc",
        "baseline_pr_auc",
        "baseline_brier_score",
        "baseline_ece",
        "baseline_precision",
        "baseline_recall",
        "baseline_f1",
        "baseline_confusion_matrix",
        "hypersigma_threshold",
        "hypersigma_roc_auc",
        "hypersigma_pr_auc",
        "hypersigma_brier_score",
        "hypersigma_ece",
        "hypersigma_calibration_mode",
        "hypersigma_precision",
        "hypersigma_recall",
        "hypersigma_f1",
        "hypersigma_confusion_matrix",
        "delta_roc_auc",
        "delta_pr_auc",
        "delta_f1",
        "delta_brier_score",
        "delta_ece",
        "winner_by_f1",
        "winner_by_pr_auc",
        "winner_by_roc_auc",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    out_json.write_text(
        json.dumps(
            {
                "input_glob": args.input_glob,
                "n_runs": len(summary_rows),
                "ranking_criterion": [
                    "hypersigma_f1 desc",
                    "hypersigma_pr_auc desc",
                    "hypersigma_roc_auc desc",
                    "hypersigma_brier_score asc",
                    "hypersigma_ece asc",
                    "run_name asc",
                ],
                "rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] ranking csv saved to {out_csv}")
    print(f"[INFO] ranking json saved to {out_json}")
    print("[INFO] top runs:")
    for row in summary_rows[:10]:
        print(
            f"  rank={row['rank']} "
            f"run={row['run_name']} "
            f"hs_f1={row['hypersigma_f1']:.6f} "
            f"hs_pr_auc={row['hypersigma_pr_auc']:.6f} "
            f"hs_roc_auc={row['hypersigma_roc_auc']:.6f} "
            f"hs_brier={row['hypersigma_brier_score']:.6f} "
            f"mode={row['hypersigma_calibration_mode']} "
            f"winner_f1={row['winner_by_f1']}"
        )


if __name__ == "__main__":
    main()
