# confmask Area-cap Random Baseline Summary

## Scope

This is a research-only diagnostic for the current HyperSIGMA confmask candidate.

Protocol:

- boundary: `wet_ratio_valid >= 0.08`
- split: `guard32 + split-quality-selected constrained`
- model: HyperSIGMA `E02_head_only_posw`
- seeds: `42`, `13`, `99`
- score mode: calibrated
- full-scene sample: `96` uniformly selected patches
- area caps: `0.05`, `0.10`, `0.15`, `0.20`
- random baseline: `64` same-area random selections per policy

## Outputs

- `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_threshold_sanity_summary.csv`
- `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_threshold_sanity_summary.json`
- `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_policy_gate.md`
- `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap_random_baseline_sampled/fullscene_policy_gate.json`

## Key Result

Calibrated area-cap policies remain spatially stable and beat same-area random selection, but the margin is still modest.

| policy | positive rate range | F1 mean | random F1 | F1-random |
|---|---:|---:|---:|---:|
| `area_cap_05` | `0.0479-0.0500` | `0.0648` | `0.0609` | `0.0039` |
| `area_cap_10` | `0.0970-0.1000` | `0.0977` | `0.0880` | `0.0097` |
| `area_cap_15` | `0.1494-0.1500` | `0.1236` | `0.1042` | `0.0194` |
| `area_cap_20` | `0.1997-0.2000` | `0.1329` | `0.1143` | `0.0186` |

Interpretation:

- `area_cap_15` gives the strongest localization enrichment over random.
- `area_cap_20` gives the highest mean sampled F1, but with broader selected area.
- `val_f1_tuned` is still too broad for full-scene operation: calibrated positive rate range is `0.6677-0.9997`.

## Decision

Do not promote this checkpoint/policy to production.

Reason:

- The model does beat random same-area selection, so there is a usable signal.
- The improvement over random is small, so full-scene localization remains weak.
- `area_cap_15` and `area_cap_20` are now the most useful HyperSIGMA operating points to compare against the baseline.

Do not run `E03_head_only_confidence_bce` yet.

Reason:

- The next highest-value question is whether this weak localization is HyperSIGMA-specific.
- Run the same random-baseline area-cap protocol for the baseline model first.

## Suggested Next Step

Run baseline full-scene sampled area-cap evaluation using the same caps and random comparator. If baseline clearly beats HyperSIGMA at `area_cap_15` or `area_cap_20`, then the next HyperSIGMA experiment should change the loss or fine-tuning strategy. If baseline is similarly weak, prioritize label policy, full-scene sampling, or post-processing before another HyperSIGMA training run.
