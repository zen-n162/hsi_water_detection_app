# Full-scene Threshold Policy Gate

This gate evaluates sampled full-scene predicted-positive-rate stability across selected seeds.

| mode | policy | pred_pos_rate min | max | range | mean | F1 mean | random F1 | F1-random | extreme? | pass stability | pass cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | area_cap_05 | 0.0000 | 0.0003 | 0.0003 | 0.0001 | 0.0000 | 0.0002 | -0.0002 | true | false | true |
| raw | area_cap_10 | 0.0000 | 0.0003 | 0.0003 | 0.0001 | 0.0000 | 0.0002 | -0.0002 | true | false | true |
| raw | area_cap_15 | 0.1342 | 0.1479 | 0.0137 | 0.1415 | 0.1008 | 0.1022 | -0.0014 | false | true | true |
| raw | area_cap_20 | 0.1342 | 0.1479 | 0.0137 | 0.1415 | 0.1008 | 0.1024 | -0.0016 | false | true | true |
| raw | fixed_0_5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true | false | true |
| raw | prior_aware_train_prevalence | 0.4988 | 0.5127 | 0.0139 | 0.5080 | 0.1575 | 0.1383 | 0.0192 | false | true | true |
| raw | val_f1_tuned | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.1481 | 0.1481 | 0.0000 | true | false | true |
| raw | validation_prevalence_target | 0.3894 | 0.4988 | 0.1094 | 0.4275 | 0.1543 | 0.1345 | 0.0197 | false | false | true |

Interpretation:
- `passes_positive_rate_stability` requires range <= 0.10 and no seed with positive rate <= 0.02 or >= 0.98.
- `passes_area_cap` additionally checks area-cap policies do not exceed the requested cap by more than 0.01.
- `random F1` is a same-area random baseline, so positive `F1-random` indicates localization better than selecting the same number of pixels randomly.
- Patch-level F1 is reported for context only; this gate is about full-scene operating stability.
