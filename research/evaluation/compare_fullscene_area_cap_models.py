from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_rows(path: Path, *, mode: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("mode") == mode]
    if not rows:
        raise RuntimeError(f"No rows found for mode={mode} in {path}")
    return rows


def _summarize(rows: list[dict[str, str]], *, policies: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("policy") in policies:
            grouped[str(row["policy"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for policy in policies:
        items = grouped.get(policy, [])
        if not items:
            continue
        pos_rates = [float(item["predicted_positive_rate"]) for item in items]
        f1_values = [float(item["f1"]) for item in items]
        random_f1 = [_to_float(item.get("random_f1_mean")) for item in items]
        random_f1 = [v for v in random_f1 if v is not None]
        delta_random = [_to_float(item.get("f1_minus_random_mean")) for item in items]
        delta_random = [v for v in delta_random if v is not None]
        summary[policy] = {
            "seeds": ",".join(item["seed"] for item in items),
            "positive_rate_min": min(pos_rates),
            "positive_rate_max": max(pos_rates),
            "positive_rate_mean": mean(pos_rates),
            "f1_mean": mean(f1_values),
            "random_f1_mean": mean(random_f1) if random_f1 else None,
            "f1_minus_random_mean": mean(delta_random) if delta_random else None,
        }
    return summary


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Full-scene Area-cap Model Comparison",
        "",
        "This compares the current HyperSIGMA confmask candidate against the baseline model on the same sampled full-scene area-cap protocol.",
        "",
        "| policy | HyperSIGMA F1 | baseline F1 | F1 delta | HyperSIGMA F1-random | baseline F1-random | HyperSIGMA pos rate | baseline pos rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {hs_f1} | {bl_f1} | {delta} | {hs_rand} | {bl_rand} | {hs_pos} | {bl_pos} |".format(
                policy=row["policy"],
                hs_f1=_fmt(row["hypersigma_f1_mean"]),
                bl_f1=_fmt(row["baseline_f1_mean"]),
                delta=_fmt(row["hypersigma_minus_baseline_f1"]),
                hs_rand=_fmt(row["hypersigma_f1_minus_random_mean"]),
                bl_rand=_fmt(row["baseline_f1_minus_random_mean"]),
                hs_pos=_fmt(row["hypersigma_positive_rate_mean"]),
                bl_pos=_fmt(row["baseline_positive_rate_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- HyperSIGMA uses calibrated pixel-level scores from the sampled full-scene evaluator.",
            "- Baseline uses one scalar score per patch, repeated over valid pixels, so small area caps can collapse to zero positives when score ties are coarse.",
            "- Positive `F1-random` means the model-selected area beats same-area random valid-pixel selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare HyperSIGMA and baseline sampled full-scene area-cap summaries")
    parser.add_argument("--hypersigma_csv", type=Path, required=True)
    parser.add_argument("--baseline_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--hypersigma_mode", default="calibrated")
    parser.add_argument("--baseline_mode", default="raw")
    parser.add_argument("--policies", default="area_cap_05,area_cap_10,area_cap_15,area_cap_20")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    hypersigma = _summarize(_load_rows(args.hypersigma_csv, mode=args.hypersigma_mode), policies=policies)
    baseline = _summarize(_load_rows(args.baseline_csv, mode=args.baseline_mode), policies=policies)

    rows: list[dict[str, Any]] = []
    for policy in policies:
        hs = hypersigma.get(policy)
        bl = baseline.get(policy)
        if not hs or not bl:
            continue
        rows.append(
            {
                "policy": policy,
                "hypersigma_seeds": hs["seeds"],
                "baseline_seeds": bl["seeds"],
                "hypersigma_positive_rate_mean": hs["positive_rate_mean"],
                "baseline_positive_rate_mean": bl["positive_rate_mean"],
                "hypersigma_f1_mean": hs["f1_mean"],
                "baseline_f1_mean": bl["f1_mean"],
                "hypersigma_minus_baseline_f1": hs["f1_mean"] - bl["f1_mean"],
                "hypersigma_random_f1_mean": hs["random_f1_mean"],
                "baseline_random_f1_mean": bl["random_f1_mean"],
                "hypersigma_f1_minus_random_mean": hs["f1_minus_random_mean"],
                "baseline_f1_minus_random_mean": bl["f1_minus_random_mean"],
                "hypersigma_positive_rate_min": hs["positive_rate_min"],
                "hypersigma_positive_rate_max": hs["positive_rate_max"],
                "baseline_positive_rate_min": bl["positive_rate_min"],
                "baseline_positive_rate_max": bl["positive_rate_max"],
            }
        )

    if not rows:
        raise RuntimeError("No comparable policies found")
    _write_csv(rows, args.output_csv)
    _write_md(rows, args.output_md)
    print(f"[INFO] wrote {args.output_csv}")
    print(f"[INFO] wrote {args.output_md}")


if __name__ == "__main__":
    main()
