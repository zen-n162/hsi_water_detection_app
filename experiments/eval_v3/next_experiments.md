# Next experiments from best run: E02_head_only_posw_v3_block224_calibrated

- selected_by: hypersigma_pr_auc
- hypersigma_f1: 0.4
- hypersigma_pr_auc: 0.5916666666666666
- hypersigma_roc_auc: 0.7333333333333334

## 1. N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr

### train
```bash
PYTHONPATH=src:. conda run -n HyperSIGMA python research/training/train_wetness_pretrain.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --epochs 20 \
  --batch_size 2 \
  --device cuda \
  --model_type ss \
  --spat_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth \
  --spec_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth \
  --run_name N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr \
  --train_split train \
  --val_split val \
  --freeze_strategy head_only \
  --loss_type pos_weight_bce \
  --pos_weight auto \
  --head_lr 0.0005 \
  --encoder_lr 0.0 \
  --grad_clip 1.0 \
  --save_dir experiments/runs_v3
```

### evaluate / tune threshold / compare
```bash
RUN_DIR=experiments/runs_v3/N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split val \
  --threshold 0.5 \
  --output_json ${RUN_DIR}/val_metrics.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/select_threshold.py \
  --metrics_json ${RUN_DIR}/val_metrics.json \
  --output_json ${RUN_DIR}/best_val_threshold.json

BEST_THR=$(python - <<'EOF'
import json
from pathlib import Path
obj = json.loads(Path("experiments/runs_v3/N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr/best_val_threshold.json").read_text(encoding="utf-8"))
print(obj["best_threshold"])
EOF
)

echo "[BEST_THR] $BEST_THR"

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split test \
  --threshold "$BEST_THR" \
  --output_json ${RUN_DIR}/test_metrics_tuned.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/compare_models.py \
  --baseline_json experiments/eval_v3/baseline_test_metrics_tuned.json \
  --hypersigma_json ${RUN_DIR}/test_metrics_tuned.json \
  --output_json experiments/eval_v3/model_comparison_N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr_vs_baseline.json \
  --output_csv experiments/eval_v3/model_comparison_N1_from_E02_head_only_posw_v3_block224_calibrated_lower_head_lr_vs_baseline.csv
```

## 2. N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr

### train
```bash
PYTHONPATH=src:. conda run -n HyperSIGMA python research/training/train_wetness_pretrain.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --epochs 20 \
  --batch_size 2 \
  --device cuda \
  --model_type ss \
  --spat_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth \
  --spec_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth \
  --run_name N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr \
  --train_split train \
  --val_split val \
  --freeze_strategy head_only \
  --loss_type pos_weight_bce \
  --pos_weight auto \
  --head_lr 0.001 \
  --encoder_lr 0.0 \
  --grad_clip 1.0 \
  --save_dir experiments/runs_v3
```

### evaluate / tune threshold / compare
```bash
RUN_DIR=experiments/runs_v3/N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split val \
  --threshold 0.5 \
  --output_json ${RUN_DIR}/val_metrics.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/select_threshold.py \
  --metrics_json ${RUN_DIR}/val_metrics.json \
  --output_json ${RUN_DIR}/best_val_threshold.json

BEST_THR=$(python - <<'EOF'
import json
from pathlib import Path
obj = json.loads(Path("experiments/runs_v3/N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr/best_val_threshold.json").read_text(encoding="utf-8"))
print(obj["best_threshold"])
EOF
)

echo "[BEST_THR] $BEST_THR"

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split test \
  --threshold "$BEST_THR" \
  --output_json ${RUN_DIR}/test_metrics_tuned.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/compare_models.py \
  --baseline_json experiments/eval_v3/baseline_test_metrics_tuned.json \
  --hypersigma_json ${RUN_DIR}/test_metrics_tuned.json \
  --output_json experiments/eval_v3/model_comparison_N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr_vs_baseline.json \
  --output_csv experiments/eval_v3/model_comparison_N2_from_E02_head_only_posw_v3_block224_calibrated_lower_encoder_lr_vs_baseline.csv
```

## 3. N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs

### train
```bash
PYTHONPATH=src:. conda run -n HyperSIGMA python research/training/train_wetness_pretrain.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --epochs 30 \
  --batch_size 2 \
  --device cuda \
  --model_type ss \
  --spat_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth \
  --spec_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth \
  --run_name N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs \
  --train_split train \
  --val_split val \
  --freeze_strategy head_only \
  --loss_type pos_weight_bce \
  --pos_weight auto \
  --head_lr 0.001 \
  --encoder_lr 0.0 \
  --grad_clip 1.0 \
  --save_dir experiments/runs_v3
```

### evaluate / tune threshold / compare
```bash
RUN_DIR=experiments/runs_v3/N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split val \
  --threshold 0.5 \
  --output_json ${RUN_DIR}/val_metrics.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/select_threshold.py \
  --metrics_json ${RUN_DIR}/val_metrics.json \
  --output_json ${RUN_DIR}/best_val_threshold.json

BEST_THR=$(python - <<'EOF'
import json
from pathlib import Path
obj = json.loads(Path("experiments/runs_v3/N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs/best_val_threshold.json").read_text(encoding="utf-8"))
print(obj["best_threshold"])
EOF
)

echo "[BEST_THR] $BEST_THR"

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split test \
  --threshold "$BEST_THR" \
  --output_json ${RUN_DIR}/test_metrics_tuned.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/compare_models.py \
  --baseline_json experiments/eval_v3/baseline_test_metrics_tuned.json \
  --hypersigma_json ${RUN_DIR}/test_metrics_tuned.json \
  --output_json experiments/eval_v3/model_comparison_N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs_vs_baseline.json \
  --output_csv experiments/eval_v3/model_comparison_N3_from_E02_head_only_posw_v3_block224_calibrated_longer_epochs_vs_baseline.csv
```

## 4. N4_from_E02_head_only_posw_v3_block224_calibrated_progressive

### train
```bash
PYTHONPATH=src:. conda run -n HyperSIGMA python research/training/train_wetness_pretrain.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --epochs 30 \
  --batch_size 2 \
  --device cuda \
  --model_type ss \
  --spat_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth \
  --spec_checkpoint /home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth \
  --run_name N4_from_E02_head_only_posw_v3_block224_calibrated_progressive \
  --train_split train \
  --val_split val \
  --freeze_strategy progressive \
  --loss_type pos_weight_bce \
  --pos_weight auto \
  --head_lr 0.001 \
  --encoder_lr 5e-05 \
  --grad_clip 1.0 \
  --save_dir experiments/runs_v3 \
  --unfreeze_spat_last_n 2 \
  --unfreeze_spec_last_n 2
```

### evaluate / tune threshold / compare
```bash
RUN_DIR=experiments/runs_v3/N4_from_E02_head_only_posw_v3_block224_calibrated_progressive

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split val \
  --threshold 0.5 \
  --output_json ${RUN_DIR}/val_metrics.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/select_threshold.py \
  --metrics_json ${RUN_DIR}/val_metrics.json \
  --output_json ${RUN_DIR}/best_val_threshold.json

BEST_THR=$(python - <<'EOF'
import json
from pathlib import Path
obj = json.loads(Path("experiments/runs_v3/N4_from_E02_head_only_posw_v3_block224_calibrated_progressive/best_val_threshold.json").read_text(encoding="utf-8"))
print(obj["best_threshold"])
EOF
)

echo "[BEST_THR] $BEST_THR"

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/eval_wetness_detector.py \
  --input_dir datasets/processed/wetness_pretrain_v3 \
  --manifest_path annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv \
  --device cuda \
  --model_type ss \
  --model_checkpoint ${RUN_DIR}/model_best.pt \
  --split test \
  --threshold "$BEST_THR" \
  --output_json ${RUN_DIR}/test_metrics_tuned.json

PYTHONPATH=src:. conda run -n HyperSIGMA python research/evaluation/compare_models.py \
  --baseline_json experiments/eval_v3/baseline_test_metrics_tuned.json \
  --hypersigma_json ${RUN_DIR}/test_metrics_tuned.json \
  --output_json experiments/eval_v3/model_comparison_N4_from_E02_head_only_posw_v3_block224_calibrated_progressive_vs_baseline.json \
  --output_csv experiments/eval_v3/model_comparison_N4_from_E02_head_only_posw_v3_block224_calibrated_progressive_vs_baseline.csv
```
