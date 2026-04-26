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

The baseline matched-seed check is now complete for the same `42`, `13`, `99` sampled full-scene protocol:

- `area_cap_05`: HyperSIGMA F1 `0.0648`, baseline F1 `0.0000`
- `area_cap_10`: HyperSIGMA F1 `0.0977`, baseline F1 `0.0000`
- `area_cap_15`: HyperSIGMA F1 `0.1236`, baseline F1 `0.1008`
- `area_cap_20`: HyperSIGMA F1 `0.1329`, baseline F1 `0.1008`

Baseline uses one scalar score per patch, so small area caps can collapse to zero positives when score ties are coarse. Even with this caveat, the current evidence does not show a baseline localization advantage over HyperSIGMA.

## Decision

Do not run `E03_head_only_confidence_bce` yet.

The current failure is not proven to be a loss-specific HyperSIGMA collapse, and the baseline does not beat HyperSIGMA under the same sampled full-scene area-cap policy. A conservative threshold policy can control full-scene extent, so the next step should reduce full-scene estimate noise and inspect localization before changing the loss.

Do not promote to GUI / production yet.

## Next Work Plan

1. Increase full-scene sample size for the selected policies.
   Rerun HyperSIGMA and baseline checks with `192` or `256` sampled patches, focusing on `area_cap_15` and `area_cap_20`. This reduces noise in the localization estimate before changing training.

2. Render and inspect selected-area overlays.
   Use the existing overlay renderer for `area_cap_15` and `area_cap_20`, then check whether selected pixels cluster on plausible wet structures or simply track broad scene artifacts.

3. Improve the baseline comparator if needed.
   The current baseline is patch-scalar and coarse. If it remains part of the argument, add a patch-score smoothing or stride-reduced baseline map so area caps do not collapse at small rates because of ties.

4. Then decide whether to run `E03_head_only_confidence_bce`.
   Run it only if higher-sample overlays show HyperSIGMA remains weak or seed-unstable even when baseline/coarse-score limitations are accounted for.

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

HSI_DEBUG=0 PYTHONPATH=src:. conda run --no-capture-output -n HyperSIGMA \
  python research/evaluation/run_baseline_fullscene_threshold_sanity.py \
  --runs_json experiments/eval_confmask_thr008_splitquality_stress_repeat/baseline_fullscene_sanity_runs.json \
  --split_repeat_summary_json experiments/eval_confmask_thr008_splitquality_stress_repeat/split_repeat_summary_5seed.json \
  --output_dir experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_comparator_baseline_sampled \
  --device cpu \
  --patch_size 64 \
  --stride 64 \
  --max_patches 96 \
  --area_caps 0.05,0.10,0.15,0.20 \
  --random_baseline_repeats 64

PYTHONPATH=src:. python research/evaluation/compare_fullscene_area_cap_models.py \
  --hypersigma_csv experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_threshold_sanity_summary.csv \
  --baseline_csv experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_comparator_baseline_sampled/fullscene_threshold_sanity_summary.csv \
  --output_csv experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_model_comparison.csv \
  --output_md docs/confmask_fullscene_area_cap_model_comparison.md
```
