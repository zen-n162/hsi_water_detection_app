from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_ranking_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No rows found in ranking json: {path}")
    return rows


def _pick_best_run(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if metric not in {"hypersigma_f1", "hypersigma_pr_auc", "hypersigma_roc_auc"}:
        raise ValueError(f"Unsupported metric: {metric}")

    return sorted(
        rows,
        key=lambda r: (
            -_as_float(r.get(metric)),
            -_as_float(r.get("hypersigma_pr_auc")),
            -_as_float(r.get("hypersigma_roc_auc")),
            str(r.get("run_name", "")),
        ),
    )[0]


def _build_next_plan(best_run_name: str) -> list[dict[str, Any]]:
    """
    いまの流れに合わせて、best run を起点に次の4本を自動提案する。
    目的:
      1. head_lr を少し下げる
      2. encoder_lr を少し下げる
      3. epochs を伸ばす
      4. progressive を追加で試す
    """

    base = {
        "input_dir": "datasets/processed/wetness_pretrain_v3",
        "manifest_path": "annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv",
        "device": "cuda",
        "model_type": "ss",
        "spat_checkpoint": "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth",
        "spec_checkpoint": "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth",
        "train_split": "train",
        "val_split": "val",
        "save_dir": "experiments/runs_v3",
        "batch_size": 2,
        "grad_clip": 1.0,
        "loss_type": "pos_weight_bce",
        "pos_weight": "auto",
    }

    # best run 名に応じて初期 freeze_strategy を推定
    if "head_only" in best_run_name:
        freeze_strategy = "head_only"
    elif "spec_last" in best_run_name:
        freeze_strategy = "spec_last"
    elif "dual_last" in best_run_name:
        freeze_strategy = "dual_last"
    elif "progressive" in best_run_name:
        freeze_strategy = "progressive"
    else:
        freeze_strategy = "head_only"

    plans: list[dict[str, Any]] = []

    # N1: best run の lr 微調整
    plans.append({
        **base,
        "run_name": f"N1_from_{best_run_name}_lower_head_lr",
        "epochs": 20,
        "freeze_strategy": freeze_strategy,
        "head_lr": 5e-4,
        "encoder_lr": 1e-4 if freeze_strategy != "head_only" else 0.0,
        "unfreeze_spat_last_n": 2 if freeze_strategy in {"dual_last", "progressive"} else None,
        "unfreeze_spec_last_n": 2 if freeze_strategy in {"spec_last", "dual_last", "progressive"} else None,
    })

    # N2: encoder lr を弱める
    plans.append({
        **base,
        "run_name": f"N2_from_{best_run_name}_lower_encoder_lr",
        "epochs": 20,
        "freeze_strategy": freeze_strategy,
        "head_lr": 1e-3,
        "encoder_lr": 5e-5 if freeze_strategy != "head_only" else 0.0,
        "unfreeze_spat_last_n": 2 if freeze_strategy in {"dual_last", "progressive"} else None,
        "unfreeze_spec_last_n": 2 if freeze_strategy in {"spec_last", "dual_last", "progressive"} else None,
    })

    # N3: epoch を伸ばす
    plans.append({
        **base,
        "run_name": f"N3_from_{best_run_name}_longer_epochs",
        "epochs": 30,
        "freeze_strategy": freeze_strategy,
        "head_lr": 1e-3,
        "encoder_lr": 1e-4 if freeze_strategy != "head_only" else 0.0,
        "unfreeze_spat_last_n": 2 if freeze_strategy in {"dual_last", "progressive"} else None,
        "unfreeze_spec_last_n": 2 if freeze_strategy in {"spec_last", "dual_last", "progressive"} else None,
    })

    # N4: progressive を追加探索
    plans.append({
        **base,
        "run_name": f"N4_from_{best_run_name}_progressive",
        "epochs": 30,
        "freeze_strategy": "progressive",
        "head_lr": 1e-3,
        "encoder_lr": 5e-5,
        "unfreeze_spat_last_n": 2,
        "unfreeze_spec_last_n": 2,
    })

    return plans


def _train_command(plan: dict[str, Any]) -> str:
    parts = [
        "PYTHONPATH=src:. conda run -n HyperSIGMA python research/training/train_wetness_pretrain.py",
        f'  --input_dir {plan["input_dir"]}',
        f'  --manifest_path {plan["manifest_path"]}',
        f'  --epochs {plan["epochs"]}',
        f'  --batch_size {plan["batch_size"]}',
        f'  --device {plan["device"]}',
        f'  --model_type {plan["model_type"]}',
        f'  --spat_checkpoint {plan["spat_checkpoint"]}',
        f'  --spec_checkpoint {plan["spec_checkpoint"]}',
        f'  --run_name {plan["run_name"]}',
        f'  --train_split {plan["train_split"]}',
        f'  --val_split {plan["val_split"]}',
        f'  --freeze_strategy {plan["freeze_strategy"]}',
        f'  --loss_type {plan["loss_type"]}',
        f'  --pos_weight {plan["pos_weight"]}',
        f'  --head_lr {plan["head_lr"]}',
        f'  --encoder_lr {plan["encoder_lr"]}',
        f'  --grad_clip {plan["grad_clip"]}',
        f'  --save_dir {plan["save_dir"]}',
    ]

    if plan.get("unfreeze_spat_last_n") is not None:
        parts.append(f'  --unfreeze_spat_last_n {plan["unfreeze_spat_last_n"]}')
    if plan.get("unfreeze_spec_last_n") is not None:
        parts.append(f'  --unfreeze_spec_last_n {plan["unfreeze_spec_last_n"]}')

    return " \\\n".join(parts)


def _eval_block(plan: dict[str, Any]) -> str:
    run_dir = f'{plan["save_dir"]}/{plan["run_name"]}'
    return f"""RUN_DIR={run_dir}

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \\
  --input_dir {plan["input_dir"]} \\
  --manifest_path {plan["manifest_path"]} \\
  --device {plan["device"]} \\
  --model_type {plan["model_type"]} \\
  --model_checkpoint ${{RUN_DIR}}/model_best.pt \\
  --split val \\
  --threshold 0.5 \\
  --output_json ${{RUN_DIR}}/val_metrics.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/select_threshold.py \\
  --metrics_json ${{RUN_DIR}}/val_metrics.json \\
  --output_json ${{RUN_DIR}}/best_val_threshold.json

BEST_THR=$(python - <<'EOF'
import json
from pathlib import Path
obj = json.loads(Path("{run_dir}/best_val_threshold.json").read_text(encoding="utf-8"))
print(obj["best_threshold"])
EOF
)

echo "[BEST_THR] $BEST_THR"

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \\
  --input_dir {plan["input_dir"]} \\
  --manifest_path {plan["manifest_path"]} \\
  --device {plan["device"]} \\
  --model_type {plan["model_type"]} \\
  --model_checkpoint ${{RUN_DIR}}/model_best.pt \\
  --split test \\
  --threshold "$BEST_THR" \\
  --output_json ${{RUN_DIR}}/test_metrics_tuned.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/compare_models.py \\
  --baseline_json experiments/eval_v3/baseline_test_metrics_tuned.json \\
  --hypersigma_json ${{RUN_DIR}}/test_metrics_tuned.json \\
  --output_json experiments/eval_v3/model_comparison_{plan["run_name"]}_vs_baseline.json \\
  --output_csv experiments/eval_v3/model_comparison_{plan["run_name"]}_vs_baseline.csv
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick the best run from ranking JSON and generate next experiment commands")
    ap.add_argument(
        "--ranking_json",
        default="experiments/eval_v3/hypersigma_finetune_ranking.json",
        help="Input ranking json made by summarize_results.py",
    )
    ap.add_argument(
        "--select_by",
        default="hypersigma_pr_auc",
        choices=["hypersigma_f1", "hypersigma_pr_auc", "hypersigma_roc_auc"],
        help="Metric used to pick the best run",
    )
    ap.add_argument(
        "--output_md",
        default="experiments/eval_v3/next_experiments.md",
        help="Human-readable markdown with commands",
    )
    ap.add_argument(
        "--output_sh",
        default="experiments/eval_v3/run_next_experiments.sh",
        help="Shell script with commands",
    )
    args = ap.parse_args()

    ranking_path = Path(args.ranking_json)
    rows = _load_ranking_rows(ranking_path)
    best = _pick_best_run(rows, args.select_by)

    best_run_name = str(best["run_name"])
    plans = _build_next_plan(best_run_name)

    out_md = Path(args.output_md)
    out_sh = Path(args.output_sh)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_sh.parent.mkdir(parents=True, exist_ok=True)

    md_parts: list[str] = []
    md_parts.append(f"# Next experiments from best run: {best_run_name}")
    md_parts.append("")
    md_parts.append(f"- selected_by: {args.select_by}")
    md_parts.append(f"- hypersigma_f1: {best.get('hypersigma_f1')}")
    md_parts.append(f"- hypersigma_pr_auc: {best.get('hypersigma_pr_auc')}")
    md_parts.append(f"- hypersigma_roc_auc: {best.get('hypersigma_roc_auc')}")
    md_parts.append("")

    sh_parts: list[str] = []
    sh_parts.append("#!/usr/bin/env bash")
    sh_parts.append("set -euo pipefail")
    sh_parts.append("")
    sh_parts.append("cd ~/MasterResearch/hsi_water_detection_app")
    sh_parts.append("")

    for i, plan in enumerate(plans, start=1):
        md_parts.append(f"## {i}. {plan['run_name']}")
        md_parts.append("")
        md_parts.append("### train")
        md_parts.append("```bash")
        md_parts.append(_train_command(plan))
        md_parts.append("```")
        md_parts.append("")
        md_parts.append("### evaluate / tune threshold / compare")
        md_parts.append("```bash")
        md_parts.append(_eval_block(plan).rstrip())
        md_parts.append("```")
        md_parts.append("")

        sh_parts.append(f'echo "===== START {plan["run_name"]} ====="')
        sh_parts.append(_train_command(plan))
        sh_parts.append("")
        sh_parts.append(_eval_block(plan).rstrip())
        sh_parts.append("")
        sh_parts.append(f'echo "===== END {plan["run_name"]} ====="')
        sh_parts.append("")

    out_md.write_text("\n".join(md_parts), encoding="utf-8")
    out_sh.write_text("\n".join(sh_parts), encoding="utf-8")

    print(f"[INFO] ranking source: {ranking_path}")
    print(f"[INFO] selected best run: {best_run_name}")
    print(f"[INFO] selected_by: {args.select_by}")
    print(f"[INFO] markdown saved to {out_md}")
    print(f"[INFO] shell script saved to {out_sh}")

    print("[INFO] next run names:")
    for plan in plans:
        print(" ", plan["run_name"])


if __name__ == "__main__":
    main()
