# Full-scene Threshold Policy Gate

This gate evaluates sampled full-scene predicted-positive-rate stability across selected seeds.

| mode | policy | pred_pos_rate min | max | range | mean | F1 mean | random F1 | F1-random | extreme? | pass stability | pass cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | area_cap_15 | 0.1500 | 0.1500 | 0.0000 | 0.1500 | 0.1285 | 0.0963 | 0.0323 | false | true | true |
| calibrated | area_cap_20 | 0.1989 | 0.2000 | 0.0011 | 0.1996 | 0.1425 | 0.1045 | 0.0380 | false | true | true |
| calibrated | fixed_0_5 | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0882 | 0.0881 | 0.0001 | true | false | true |
| calibrated | prior_aware_train_prevalence | 0.5166 | 0.5439 | 0.0273 | 0.5334 | 0.1420 | 0.1247 | 0.0173 | false | true | true |
| calibrated | val_f1_tuned | 0.6440 | 1.0000 | 0.3560 | 0.8621 | 0.1332 | 0.1304 | 0.0028 | true | false | true |
| calibrated | validation_prevalence_target | 0.4984 | 0.4999 | 0.0015 | 0.4994 | 0.1441 | 0.1239 | 0.0202 | false | true | true |
| raw | area_cap_15 | 0.0000 | 0.1500 | 0.1500 | 0.0995 | 0.0903 | 0.0640 | 0.0262 | true | false | true |
| raw | area_cap_20 | 0.0000 | 0.2000 | 0.2000 | 0.1190 | 0.0909 | 0.0673 | 0.0236 | true | false | true |
| raw | fixed_0_5 | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0882 | 0.0881 | 0.0001 | true | false | true |
| raw | prior_aware_train_prevalence | 0.4503 | 0.5397 | 0.0893 | 0.5069 | 0.1419 | 0.1241 | 0.0178 | false | true | true |
| raw | val_f1_tuned | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0882 | 0.0881 | 0.0001 | true | false | true |
| raw | validation_prevalence_target | 0.4503 | 0.4985 | 0.0482 | 0.4792 | 0.1417 | 0.1233 | 0.0184 | false | true | true |

Interpretation:
- `passes_positive_rate_stability` requires range <= 0.10 and no seed with positive rate <= 0.02 or >= 0.98.
- `passes_area_cap` additionally checks area-cap policies do not exceed the requested cap by more than 0.01.
- `random F1` is a same-area random baseline, so positive `F1-random` indicates localization better than selecting the same number of pixels randomly.
- Patch-level F1 is reported for context only; this gate is about full-scene operating stability.
