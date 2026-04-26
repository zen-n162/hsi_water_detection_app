from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["policy"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (mode, policy), items in sorted(grouped.items()):
        pos_rates = [float(item["predicted_positive_rate"]) for item in items]
        f1_values = [float(item["f1"]) for item in items]
        random_f1_values = [_to_float(item.get("random_f1_mean")) for item in items]
        random_f1_values = [v for v in random_f1_values if v is not None]
        f1_minus_random_values = [_to_float(item.get("f1_minus_random_mean")) for item in items]
        f1_minus_random_values = [v for v in f1_minus_random_values if v is not None]
        target_values = [_to_float(item.get("target_positive_rate")) for item in items]
        target_values = [v for v in target_values if v is not None]
        has_extreme = any(v <= 0.02 or v >= 0.98 for v in pos_rates)
        pos_range = max(pos_rates) - min(pos_rates)
        pass_stability = pos_range <= 0.10 and not has_extreme
        pass_area_cap = True
        if policy.startswith("area_cap_"):
            cap = max(target_values) if target_values else 0.10
            pass_area_cap = max(pos_rates) <= cap + 0.01
        summaries.append(
            {
                "mode": mode,
                "policy": policy,
                "seeds": [item["seed"] for item in items],
                "target_positive_rate_mean": mean(target_values) if target_values else None,
                "predicted_positive_rate_min": min(pos_rates),
                "predicted_positive_rate_max": max(pos_rates),
                "predicted_positive_rate_mean": mean(pos_rates),
                "predicted_positive_rate_std": pstdev(pos_rates),
                "predicted_positive_rate_range": pos_range,
                "f1_mean": mean(f1_values),
                "f1_std": pstdev(f1_values),
                "random_f1_mean": mean(random_f1_values) if random_f1_values else None,
                "f1_minus_random_mean": mean(f1_minus_random_values) if f1_minus_random_values else None,
                "has_extreme_positive_rate": has_extreme,
                "passes_positive_rate_stability": pass_stability,
                "passes_area_cap": pass_area_cap,
            }
        )
    return {"policy_summaries": summaries}


def _write_markdown(summary: dict[str, Any], output_md: Path) -> None:
    lines = [
        "# Full-scene Threshold Policy Gate",
        "",
        "This gate evaluates sampled full-scene predicted-positive-rate stability across selected seeds.",
        "",
        "| mode | policy | pred_pos_rate min | max | range | mean | F1 mean | random F1 | F1-random | extreme? | pass stability | pass cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["policy_summaries"]:
        lines.append(
            "| {mode} | {policy} | {mn:.4f} | {mx:.4f} | {rg:.4f} | {av:.4f} | {f1:.4f} | {rf1} | {delta} | {ext} | {stable} | {cap} |".format(
                mode=row["mode"],
                policy=row["policy"],
                mn=row["predicted_positive_rate_min"],
                mx=row["predicted_positive_rate_max"],
                rg=row["predicted_positive_rate_range"],
                av=row["predicted_positive_rate_mean"],
                f1=row["f1_mean"],
                rf1="-" if row["random_f1_mean"] is None else f"{row['random_f1_mean']:.4f}",
                delta="-" if row["f1_minus_random_mean"] is None else f"{row['f1_minus_random_mean']:.4f}",
                ext=str(row["has_extreme_positive_rate"]).lower(),
                stable=str(row["passes_positive_rate_stability"]).lower(),
                cap=str(row["passes_area_cap"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- `passes_positive_rate_stability` requires range <= 0.10 and no seed with positive rate <= 0.02 or >= 0.98.",
            "- `passes_area_cap` additionally checks area-cap policies do not exceed the requested cap by more than 0.01.",
            "- `random F1` is a same-area random baseline, so positive `F1-random` indicates localization better than selecting the same number of pixels randomly.",
            "- Patch-level F1 is reported for context only; this gate is about full-scene operating stability.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full-scene threshold policy stability gate")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = _summarize(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, args.output_md)
    print(f"[INFO] wrote {args.output_json}")
    print(f"[INFO] wrote {args.output_md}")


if __name__ == "__main__":
    main()
