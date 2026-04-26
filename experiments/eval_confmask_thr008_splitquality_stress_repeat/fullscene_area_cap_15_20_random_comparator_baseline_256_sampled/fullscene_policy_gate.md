# Full-scene Threshold Policy Gate

This gate evaluates sampled full-scene predicted-positive-rate stability across selected seeds.

| mode | policy | pred_pos_rate min | max | range | mean | F1 mean | random F1 | F1-random | extreme? | pass stability | pass cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | area_cap_15 | 0.1086 | 0.1476 | 0.0390 | 0.1290 | 0.1106 | 0.0913 | 0.0193 | false | true | true |
| raw | area_cap_20 | 0.1763 | 0.1999 | 0.0236 | 0.1891 | 0.1165 | 0.1031 | 0.0134 | false | true | true |
| raw | fixed_0_5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true | false | true |
| raw | prior_aware_train_prevalence | 0.5026 | 0.5338 | 0.0312 | 0.5146 | 0.1507 | 0.1245 | 0.0261 | false | true | true |
| raw | val_f1_tuned | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.1321 | 0.1321 | -0.0000 | true | false | true |
| raw | validation_prevalence_target | 0.4843 | 0.4981 | 0.0137 | 0.4902 | 0.1492 | 0.1237 | 0.0256 | false | true | true |

Interpretation:
- `passes_positive_rate_stability` requires range <= 0.10 and no seed with positive rate <= 0.02 or >= 0.98.
- `passes_area_cap` additionally checks area-cap policies do not exceed the requested cap by more than 0.01.
- `random F1` is a same-area random baseline, so positive `F1-random` indicates localization better than selecting the same number of pixels randomly.
- Patch-level F1 is reported for context only; this gate is about full-scene operating stability.
