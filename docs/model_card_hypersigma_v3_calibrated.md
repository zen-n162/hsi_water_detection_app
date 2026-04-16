# Model Card: HyperSIGMA v3 Calibrated

## Model Name

`hypersigma_v3_calibrated`

## Intended Use

Patch-based wetness inference for the GUI and backend production candidate path in this repository.

## Input Data

- Hyperspectral cube input
- Hyperion production path
- Patch size: `64x64`
- Band count: `170`

## Label Source

- Source of truth: `confidence_ali.tif`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`

## Split Policy

- `spatial_block(block_size=224)`

## Primary Metrics

From the v3 calibrated test evaluation:

- ROC-AUC: `0.7333`
- PR-AUC: `0.5917`
- F1: `0.4000`

## Calibration

- Method: temperature scaling
- Fit split: validation only
- File: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`

## Threshold

- Threshold source: validation-selected calibrated threshold
- Value: `0.327428693347738`
- File: `experiments/runs_v3/E02_head_only_posw_v3_block224/best_val_threshold_calibrated.json`

## Known Failure Modes

- Fine-tuned raw scores can collapse toward zero without calibration.
- Small dataset size makes split policy influential.
- Probability quality is still sensitive to calibration choices even when ranking is acceptable.

## Non-Goals

- This card does not certify cross-scene generalization outside the current data source.
- This model is not the baseline reference path.
- This model card does not define future research candidates; it only describes the frozen production candidate.
