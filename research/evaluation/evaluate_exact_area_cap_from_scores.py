from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


def _stable_seed(*parts: Any, base_seed: int) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**32)


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _binary_metrics_from_prediction(labels: np.ndarray, pred_positive: np.ndarray) -> dict[str, Any]:
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
    metrics: list[dict[str, Any]] = []
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


def _array_from_npz(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of {names} found in {data.files}")


def _exact_prediction(scores: np.ndarray, target_rate: float, *, seed: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    k = int(round(float(np.clip(target_rate, 0.0, 1.0)) * n))
    k = max(0, min(k, n))
    pred = np.zeros(n, dtype=bool)
    if k <= 0:
        return pred
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(n)
    order = np.lexsort((tie_breaker, -scores))
    pred[order[:k]] = True
    return pred


def _parse_caps(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _load_scores(seed_dir: Path, mode: str) -> tuple[np.ndarray, np.ndarray]:
    npz_name = "sampled_scores_calibrated.npz" if mode == "calibrated" else "sampled_scores.npz"
    data = np.load(seed_dir / npz_name)
    if mode == "calibrated":
        scores = _array_from_npz(data, ("calibrated_scores", "scores"))
        labels = _array_from_npz(data, ("calibrated_labels", "labels"))
    else:
        scores = _array_from_npz(data, ("raw_scores", "scores"))
        labels = _array_from_npz(data, ("raw_labels", "labels"))
    return scores.astype(np.float64), labels.astype(np.int64)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    policies = sorted({row["policy"] for row in rows})
    for policy in policies:
        items = [row for row in rows if row["policy"] == policy]
        f1_values = [float(row["f1"]) for row in items]
        random_values = [float(row["random_f1_mean"]) for row in items]
        delta_values = [float(row["f1_minus_random_mean"]) for row in items]
        pos_rates = [float(row["predicted_positive_rate"]) for row in items]
        out.append(
            {
                "policy": policy,
                "seeds": ",".join(str(row["seed"]) for row in items),
                "predicted_positive_rate_mean": mean(pos_rates),
                "predicted_positive_rate_std": pstdev(pos_rates),
                "f1_mean": mean(f1_values),
                "f1_std": pstdev(f1_values),
                "random_f1_mean": mean(random_values),
                "f1_minus_random_mean": mean(delta_values),
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: list[dict[str, Any]], path: Path, *, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Exact area-cap ranking selects exactly the requested number of sampled valid pixels by score.",
        "Ties are broken by a deterministic random key, which is useful for patch-scalar models whose scores are repeated over many pixels.",
        "",
        "| policy | pos rate mean | F1 mean | random F1 | F1-random |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {policy} | {pos:.4f} | {f1:.4f} | {random:.4f} | {delta:.4f} |".format(
                policy=row["policy"],
                pos=row["predicted_positive_rate_mean"],
                f1=row["f1_mean"],
                random=row["random_f1_mean"],
                delta=row["f1_minus_random_mean"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exact top-k area-cap metrics from sampled score files")
    parser.add_argument("--policy_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["raw", "calibrated"], default="raw")
    parser.add_argument("--caps", default="0.15,0.20")
    parser.add_argument("--random_baseline_repeats", type=int, default=64)
    parser.add_argument("--random_baseline_seed", type=int, default=20260426)
    parser.add_argument("--title", default="Exact Area-cap Ranking Summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    caps = _parse_caps(args.caps)
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(args.policy_dir.glob("seed*"), key=lambda p: int(p.name.removeprefix("seed"))):
        seed = seed_dir.name.removeprefix("seed")
        scores, labels = _load_scores(seed_dir, args.mode)
        for cap in caps:
            policy = f"area_cap_{int(round(cap * 100)):02d}_exact_rank"
            pred = _exact_prediction(
                scores,
                cap,
                seed=_stable_seed(seed, args.mode, policy, base_seed=args.random_baseline_seed),
            )
            metrics = _binary_metrics_from_prediction(labels, pred)
            random_baseline = _random_same_area_baseline(
                labels=labels,
                predicted_positive_rate=float(pred.mean()),
                repeats=args.random_baseline_repeats,
                seed=_stable_seed(seed, args.mode, policy, "random", base_seed=args.random_baseline_seed),
            )
            random_f1 = 0.0 if random_baseline is None else float(random_baseline["f1_mean"])
            rows.append(
                {
                    "seed": seed,
                    "mode": args.mode,
                    "policy": policy,
                    "target_positive_rate": cap,
                    "predicted_positive_rate": float(pred.mean()),
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "random_f1_mean": random_f1,
                    "random_f1_std": None if random_baseline is None else random_baseline["f1_std"],
                    "f1_minus_random_mean": float(metrics["f1"] - random_f1),
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )

    if not rows:
        raise RuntimeError(f"No seed score files found under {args.policy_dir}")
    summary = _summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "exact_area_cap_rows.csv")
    _write_csv(summary, args.output_dir / "exact_area_cap_summary.csv")
    _write_markdown(summary, args.output_dir / "exact_area_cap_summary.md", title=args.title)
    (args.output_dir / "exact_area_cap_summary.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
