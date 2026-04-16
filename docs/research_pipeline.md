# Research Pipeline

This document explains the wetness / water-related HyperSIGMA research pipeline in this repository after the move to the `confidence_ali.tif`-based label source and the v3 block-split dataset.

## 1. Goal

The goal of the research pipeline is to make **HyperSIGMA-based wetness inference** reproducible, comparable, and deployable.

The pipeline supports:

- truth raster to patch-manifest generation
- patch export
- baseline and HyperSIGMA training
- validation-based threshold selection
- temperature scaling calibration
- tuned test evaluation
- comparison against a baseline
- experiment ranking and next-experiment generation
- GUI/backend deployment of a frozen best run

## 2. Label source of truth

The current source of truth is centered on:

- `confidence_ali.tif`
- `label_ali.tif`
- `gt_wetsoil_any.tif`
- `gt_wetsoil_highmid.tif`
- `gt_wetsoil_high.tif`
- `ignore_mask.tif`
- `eval_valid_mask.tif`

The important interpretation used in the rebuilt pipeline is:

- `255`: ignore / invalid
- `0`: valid dry-negative background
- `1, 2, 3`: wet-positive confidence tiers

This means the current pipeline no longer depends on the older ROI manifest as the label authority.

## 3. Dataset versions

### Legacy dataset
Earlier iterations used a smaller ROI-based manifest and generated patch datasets from those ROI rows. This was useful for initial prototyping, but the label authority and grid alignment were less clean.

### v2 dataset
The pipeline was rebuilt from `confidence_ali.tif`, producing:

- new manifest generation
- new patch export
- new train/val/test split assignment
- separated output locations to avoid mixing with legacy artifacts

### v3 dataset
The current production-candidate research pipeline uses:

- block-based spatial split
- `block_size = 224`
- reduced train/val/test touching or overlapping patch pairs relative to v2
- calibration-aware evaluation outputs

The v3 inputs are:

- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`

## 4. Patch and manifest generation rules

The rebuilt manifest pipeline uses patch-level labeling rules derived from `confidence_ali.tif`.

Core rules:

- Patch size: `64 x 64`
- Default sampling: non-overlapping patch extraction during manifest generation
- Only patches with `valid_ratio >= 0.90` are retained
- `confidence_ali in {1,2,3}` is treated as wet-positive
- `confidence_ali == 0` is treated as dry-negative
- `255` is ignored
- If `wet_ratio_valid >= 0.10`, patch label becomes `wet`
- Otherwise, patch label becomes `dry`

Manifest confidence is stored as a patch-level summary rather than copied from a prior ROI file.

## 5. Training pipelines

## 5.1 Baseline
The baseline path is a compact CNN used as a reference model.

Typical flow:
1. train on train split
2. evaluate on val split
3. select best threshold from val
4. evaluate tuned threshold on test
5. compare against HyperSIGMA

Baseline remains useful because it provides a stable non-attention reference path.

## 5.2 HyperSIGMA
HyperSIGMA training supports multiple fine-tuning strategies, including:

- head-only tuning
- spectral-last partial unfreezing
- dual-last partial unfreezing
- progressive unfreezing

Named experiment families in this work include:

- `E01_head_only_bce`
- `E02_head_only_posw`
- `E05_spec_last_posw`
- `E07_dual_last_posw`
- `E09_progressive_posw`

The best production-candidate HyperSIGMA run in the current frozen setup is:

- `E02_head_only_posw_v3_block224`

## 6. Evaluation pipeline

The evaluation scripts now support:

- split-specific evaluation
- fine-tuned checkpoint loading via `--model_checkpoint`
- sample-level output recording
- threshold-based classification metrics
- run comparison and ranking

Core metrics used through the pipeline:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Confusion Matrix

Additional v3 calibration-aware metrics may include:

- Brier score
- ECE

## 7. Threshold tuning

Thresholds are chosen from validation predictions only.

Flow:
1. evaluate on val
2. store per-sample scores in `samples[]`
3. run `select_threshold.py`
4. save best threshold JSON
5. re-run test evaluation with tuned threshold

This is important because HyperSIGMA scores can collapse toward very small values while still preserving some ranking structure.

## 8. Calibration

The pipeline now supports **temperature scaling** fit on validation outputs only.

Why it was added:
- HyperSIGMA raw scores were observed to collapse strongly toward zero
- ranking quality could remain usable while probability quality became poor
- thresholds chosen on raw collapsed scores were unstable and hard to interpret

Calibration outputs include:
- temperature JSON
- recalibrated evaluation
- Brier score / ECE summaries where enabled

The current production-candidate GUI path uses calibrated HyperSIGMA v3.

## 9. Leakage audit and split quality

Leakage auditing became a major step once the dataset was rebuilt.

The pipeline now explicitly compares:
- v2 split behavior
- v3 block-split behavior

The key reason is that patch-level datasets can accidentally allow spatially adjacent or overlapping regions to appear across train/val/test, inflating apparent generalization.

v3 block splitting is the current preferred policy because it reduced this issue materially versus v2.

## 10. Ranking and best-run selection

The following utilities support experiment management:

- `compare_models.py`
- `summarize_results.py`
- `make_next_experiments.py`

These are used to:
- compare baseline vs HyperSIGMA
- rank multiple HyperSIGMA runs
- select a current best run
- generate follow-up experiment commands automatically

In the current frozen setup:
- best HyperSIGMA deployment candidate: `E02_head_only_posw_v3_block224`
- baseline remains an important reference competitor

## 11. GUI deployment handoff

The research pipeline feeds directly into the GUI deployment path through a frozen deploy config.

That deploy config fixes:
- manifest
- dataset
- checkpoint
- temperature JSON
- threshold
- split policy
- run name

This means the GUI is no longer using an ambiguous or manually assembled inference configuration. It is tied to a reproducible research artifact.

## 12. Current limitations

- Dataset size is still modest
- HyperSIGMA score collapse still requires calibration handling
- Baseline may remain stronger on some operating points even when HyperSIGMA wins on deployability or PR-AUC
- Spatial split design can still be improved
- Threshold policy may need redesign for more stable test-time behavior

## 13. Recommended next research steps

1. Improve calibration beyond single-temperature scaling
2. Add score-collapse diagnostics directly to training logs
3. Design stronger spatially separated splits
4. Add a baseline deploy path into the GUI/backend stack
5. Revisit operating-point selection beyond single best-F1 thresholding
