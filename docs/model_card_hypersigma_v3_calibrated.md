# Model Card: HyperSIGMA v3 Calibrated

## Model Name

`hypersigma_v3_calibrated`

## Model Role

This model is the current **GUI / backend production-candidate inference path** for wetness-related ROI inference in this repository.

It is the frozen deploy default selected from the v3 research pipeline.

## Intended Use

- ROI-based wetness inference from hyperspectral image subsets
- Hyperion-focused application path in the current GUI/backend system
- Production-candidate default for the deploy-config-driven frontend/backend route

## Out of Scope

- This card does not claim robust generalization to all scenes or sensors
- This card does not certify scientific correctness outside the current label source and split policy
- This card does not describe the baseline reference model
- This card does not define future best models beyond the current frozen candidate

## Input Specification

- Sensor family currently targeted: Hyperion
- Patch size: `64 x 64`
- Stride in deploy path: `32`
- Band count: `170`
- Model type: `ss`

## Label Source

The current label source of truth is derived from:

- `confidence_ali.tif`

Derived artifacts used in the frozen pipeline:

- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`

## Split Policy

- `spatial_block(block_size=224)`

This split policy was chosen to reduce patch-level leakage relative to earlier dataset versions.

## Training Identity

Frozen deployment run:

- Run name: `E02_head_only_posw_v3_block224`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`

## Calibration

Calibration is enabled.

- Method: validation-only temperature scaling
- File: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`

This was added because raw HyperSIGMA scores showed strong collapse toward very small values, making uncalibrated probabilities unstable for deployment.

## Operating Threshold

- Threshold source: validation-selected calibrated threshold
- Threshold value: `0.327428693347738`

This threshold is intended to be used together with the calibrated score path, not with the raw uncalibrated score path.

## Primary Metrics

From the v3 calibrated test evaluation:

- ROC-AUC: `0.7333`
- PR-AUC: `0.5917`
- F1: `0.4000`

Interpretation:
- ranking quality is usable
- calibration is improved relative to the uncalibrated run
- the model is viable as the current GUI default because it preserves the deploy path and attention-capable outputs
- however, overall operating-point behavior is still sensitive

## Comparison Context

A baseline reference path remains important.

In the current project state:
- baseline is still competitive and in some views safer
- HyperSIGMA remains the preferred GUI production candidate because it fits the existing attention-based inference and deploy structure while remaining competitive under the v3 pipeline

## Known Failure Modes

- Raw scores can collapse strongly toward zero without calibration
- Small dataset size makes results sensitive to split design
- Threshold choice can materially change observed classification performance
- Probability calibration remains imperfect even after temperature scaling
- The model may preserve ranking quality while still giving poor default-threshold behavior

## Deployment Provenance

The GUI/backend deployment path should always resolve through:

- `configs/deploy/hypersigma_v3_calibrated.json`

Saved GUI/backend outputs are expected to preserve provenance through:
- API response fields
- `metadata.json`
- deploy consistency audit outputs
- release freeze / release verification records

## Recommended Monitoring

For future runs, monitor at least:

- ROC-AUC
- PR-AUC
- F1 at tuned threshold
- score distribution shape
- Brier score
- ECE
- evidence of score collapse
- split leakage indicators

## Recommended Next Work

1. Improve calibration beyond single-temperature scaling
2. Add explicit score-collapse diagnostics into training/evaluation
3. Strengthen spatial split policy further
4. Add a baseline GUI inference path for side-by-side deployment comparison
5. Revisit threshold policy for more stable real-world operation
