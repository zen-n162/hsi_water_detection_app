# confmask Next Plan After Full-scene Policy Check

## Current State

The current research candidate is:

- boundary: `wet_ratio_valid >= 0.08`
- split: `guard32 + split-quality-selected constrained`
- selection stress budget: `24` candidates, `1024` assignment restarts
- repeat seeds: `42, 13, 7, 21, 99`
- patch-level threshold: `val_f1_tuned`
- full-scene conservative extent policy candidates: `area_cap_15` for localization enrichment, `area_cap_20` for higher sampled F1

GUI, backend, deploy config, and production freeze remain unchanged.

## What Changed

Patch-level repeat and full-scene operation are now separated:

- Patch-level repeat asks whether the model can rank/test patches under a deployable validation threshold.
- Full-scene extent sanity asks whether the threshold produces plausible spatial coverage on the full Hyperion scene.

This matters because `val_f1_tuned` can pass patch-level repeat while still predicting too much of the full scene as wet.

## Latest Finding

On sampled full-scene coverage for seed42, seed13, and seed99, the earlier extent check showed:

- `val_f1_tuned` calibrated positive rate range: `0.6677-0.9997`
- `prior_aware_train_prevalence` calibrated positive rate range: `0.5172-0.5438`
- `validation_prevalence_target` calibrated positive rate range: `0.4995-0.5000`
- `area_cap_10` calibrated positive rate range: `0.0970-0.1000`

The conservative area cap is the only tested policy that prevents whole-scene over-positive behavior while preserving a fixed spatial extent bound.

The follow-up random same-area check showed that calibrated area caps beat random selection at the same positive rate, but only modestly:

- `area_cap_05`: F1 `0.0648`, random F1 `0.0609`, delta `0.0039`
- `area_cap_10`: F1 `0.0977`, random F1 `0.0880`, delta `0.0097`
- `area_cap_15`: F1 `0.1236`, random F1 `0.1042`, delta `0.0194`
- `area_cap_20`: F1 `0.1329`, random F1 `0.1143`, delta `0.0186`

`area_cap_15` is currently the best localization-enrichment operating point. `area_cap_20` has the highest mean sampled F1 but selects a broader area.

## Decision

Do not run `E03_head_only_confidence_bce` yet.

The current failure is not yet proven to be a loss-specific HyperSIGMA collapse. A conservative threshold policy can control full-scene extent, so the next step should validate localization quality under that policy before changing the loss.

Do not promote to GUI / production yet.

## Next Work Plan

1. Run the same full-scene sampled area-cap sweep for the baseline model.
   The current HyperSIGMA area-cap sweep controls positive area, but localization precision remains low. We need a baseline under the exact same full-scene operating policy before calling this a HyperSIGMA-specific failure.

2. Compare baseline against the random same-area comparator.
   The HyperSIGMA comparator is now implemented. Reuse the same `--random_baseline_repeats 64` protocol for baseline so the comparison is fair.

3. Increase full-scene sample size.
   If runtime permits, rerun selected HyperSIGMA and baseline checks with `192` or `256` sampled patches to reduce localization-estimate noise, focusing first on `area_cap_15` and `area_cap_20`.

4. Then decide whether to run `E03_head_only_confidence_bce`.
   Run it only if baseline localizes better while HyperSIGMA remains weak or seed-unstable under the same area-cap policy.

## Suggested Commands

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app

HSI_DEBUG=0 PYTHONPATH=src:. conda run --no-capture-output -n HyperSIGMA \
  python research/evaluation/run_fullscene_threshold_sanity.py \
  --runs_json experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_sanity_runs.json \
  --split_repeat_summary_json experiments/eval_confmask_thr008_splitquality_stress_repeat/split_repeat_summary_5seed.json \
  --output_dir experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled \
  --device cuda \
  --patch_size 64 \
  --stride 64 \
  --max_patches 96 \
  --area_caps 0.05,0.10,0.15,0.20 \
  --random_baseline_repeats 64

PYTHONPATH=src:. python research/evaluation/evaluate_fullscene_policy_gate.py \
  --input_csv experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_threshold_sanity_summary.csv \
  --output_json experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_policy_gate.json \
  --output_md experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_policy_gate.md
```
