#!/usr/bin/env bash
set -euo pipefail

cd /home/zennakamura/MasterResearch/hsi_water_detection_app

SPAT_CHECKPOINT="/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth"
SPEC_CHECKPOINT="/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth"
RUN_ROOT_BASE="experiments/runs_confmask_thr008_splitquality_stress_repeat"
EVAL_ROOT="experiments/eval_confmask_thr008_splitquality_stress_repeat"
PROTOCOL="splitquality_stress"

run_python() {
  PYTHONPATH=src:. conda run --no-capture-output -n HyperSIGMA python "$@"
}

run_one() {
  local seed="$1"
  local manifest="$2"
  local dataset_dir="$3"
  local tag="gb32_thr008_${PROTOCOL}_seed${seed}"
  local run_root="${RUN_ROOT_BASE}/${tag}"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local run_name="E03_head_only_confidence_bce_confmask_${PROTOCOL}_gb32_thr008_${PROTOCOL}_seed${seed}"

  mkdir -p "$run_root" "$eval_dir"

  echo "[INFO] seed=${seed}: train HyperSIGMA E03 confidence_bce"
  HSI_DEBUG=0 run_python research/training/train_wetness_pretrain.py \
    --input_dir "$dataset_dir" \
    --manifest_path "$manifest" \
    --run_name "$run_name" \
    --save_dir "$run_root" \
    --epochs 20 \
    --batch_size 2 \
    --device cuda \
    --model_type ss \
    --spat_checkpoint "$SPAT_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --patch_size 64 \
    --train_split train \
    --val_split val \
    --eval_split test \
    --freeze_strategy head_only \
    --loss_type confidence_bce \
    --pos_weight auto \
    --head_lr 1e-3 \
    --encoder_lr 0.0 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --seed "$seed"

  local ckpt="${run_root}/${run_name}/model_best.pt"
  echo "[INFO] seed=${seed}: evaluate E03 raw"
  HSI_DEBUG=0 run_python research/evaluation/eval_wetness_detector.py \
    --input_dir "$dataset_dir" \
    --manifest_path "$manifest" \
    --device cuda \
    --model_type ss \
    --model_checkpoint "$ckpt" \
    --split val \
    --threshold 0.5 \
    --output_json "${eval_dir}/${run_name}_val_metrics_raw.json"
  HSI_DEBUG=0 run_python research/evaluation/eval_wetness_detector.py \
    --input_dir "$dataset_dir" \
    --manifest_path "$manifest" \
    --device cuda \
    --model_type ss \
    --model_checkpoint "$ckpt" \
    --split test \
    --threshold 0.5 \
    --output_json "${eval_dir}/${run_name}_test_metrics_raw.json"

  echo "[INFO] seed=${seed}: temperature scaling and tuned threshold"
  run_python research/evaluation/fit_temperature_scaling.py \
    --metrics_json "${eval_dir}/${run_name}_val_metrics_raw.json" \
    --output_json "${run_root}/${run_name}/temperature_scaling_val.json" \
    --threshold 0.5
  run_python research/evaluation/rescore_metrics.py \
    --metrics_json "${eval_dir}/${run_name}_val_metrics_raw.json" \
    --output_json "${eval_dir}/${run_name}_val_metrics_calibrated.json" \
    --threshold 0.5 \
    --temperature_json "${run_root}/${run_name}/temperature_scaling_val.json" \
    --mode_label temperature_scaled
  run_python research/evaluation/rescore_metrics.py \
    --metrics_json "${eval_dir}/${run_name}_test_metrics_raw.json" \
    --output_json "${eval_dir}/${run_name}_test_metrics_calibrated.json" \
    --threshold 0.5 \
    --temperature_json "${run_root}/${run_name}/temperature_scaling_val.json" \
    --mode_label temperature_scaled
  run_python research/evaluation/select_threshold.py \
    --metrics_json "${eval_dir}/${run_name}_val_metrics_calibrated.json" \
    --output_json "${eval_dir}/${run_name}_best_val_threshold.json"
  local threshold
  threshold="$(run_python -c "import json; print(json.load(open('${eval_dir}/${run_name}_best_val_threshold.json'))['best_threshold'])")"
  run_python research/evaluation/rescore_metrics.py \
    --metrics_json "${eval_dir}/${run_name}_test_metrics_raw.json" \
    --output_json "${eval_dir}/${run_name}_test_metrics_tuned.json" \
    --threshold "$threshold" \
    --temperature_json "${run_root}/${run_name}/temperature_scaling_val.json" \
    --mode_label temperature_scaled
}

run_one \
  13 \
  "annotations/manifests/split_quality_stress/thr008_stress_c24_r1024_constrained_base13_cand0.csv" \
  "datasets/processed/wetness_pretrain_confmask_gb32_thr008_splitquality_stress_seed13"

run_one \
  99 \
  "annotations/manifests/split_quality_stress/thr008_stress_c24_r1024_constrained_base99_cand21.csv" \
  "datasets/processed/wetness_pretrain_confmask_gb32_thr008_splitquality_stress_seed99"

echo "[INFO] E03 confidence_bce seed13/seed99 complete"
