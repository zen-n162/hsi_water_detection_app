# Full-scene Threshold Policy Gate

This gate evaluates sampled full-scene predicted-positive-rate stability across selected HyperSIGMA seeds.

| mode | policy | pred_pos_rate min | max | range | mean | F1 mean | random F1 | F1-random | extreme? | pass stability | pass cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | area_cap_05 | 0.0479 | 0.0500 | 0.0021 | 0.0491 | 0.0648 | 0.0609 | 0.0039 | false | true | true |
| calibrated | area_cap_10 | 0.0970 | 0.1000 | 0.0030 | 0.0987 | 0.0977 | 0.0880 | 0.0097 | false | true | true |
| calibrated | area_cap_15 | 0.1494 | 0.1500 | 0.0006 | 0.1498 | 0.1236 | 0.1042 | 0.0194 | false | true | true |
| calibrated | area_cap_20 | 0.1997 | 0.2000 | 0.0003 | 0.1999 | 0.1329 | 0.1143 | 0.0186 | false | true | true |
| calibrated | fixed_0_5 | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0987 | 0.0987 | 0.0000 | true | false | true |
| calibrated | prior_aware_train_prevalence | 0.5172 | 0.5438 | 0.0266 | 0.5336 | 0.1476 | 0.1390 | 0.0086 | false | true | true |
| calibrated | val_f1_tuned | 0.6677 | 0.9997 | 0.3320 | 0.8699 | 0.1497 | 0.1461 | 0.0036 | true | false | true |
| calibrated | validation_prevalence_target | 0.4995 | 0.5000 | 0.0005 | 0.4998 | 0.1446 | 0.1381 | 0.0065 | false | true | true |
| raw | area_cap_05 | 0.0000 | 0.0499 | 0.0499 | 0.0329 | 0.0453 | 0.0408 | 0.0045 | true | false | true |
| raw | area_cap_10 | 0.0000 | 0.0990 | 0.0990 | 0.0542 | 0.0584 | 0.0536 | 0.0048 | true | false | true |
| raw | area_cap_15 | 0.0000 | 0.1500 | 0.1500 | 0.0712 | 0.0720 | 0.0585 | 0.0135 | true | false | true |
| raw | area_cap_20 | 0.0000 | 0.2000 | 0.2000 | 0.1307 | 0.0901 | 0.0755 | 0.0145 | true | false | true |
| raw | fixed_0_5 | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0987 | 0.0987 | 0.0000 | true | false | true |
| raw | prior_aware_train_prevalence | 0.5144 | 0.5397 | 0.0253 | 0.5283 | 0.1460 | 0.1392 | 0.0068 | false | true | true |
| raw | val_f1_tuned | 0.0000 | 1.0000 | 1.0000 | 0.6666 | 0.0987 | 0.0987 | 0.0000 | true | false | true |
| raw | validation_prevalence_target | 0.0000 | 0.5000 | 0.5000 | 0.2731 | 0.0947 | 0.0888 | 0.0059 | true | false | true |

Interpretation:
- `passes_positive_rate_stability` requires range <= 0.10 and no seed with positive rate <= 0.02 or >= 0.98.
- `passes_area_cap` additionally checks area-cap policies do not exceed the requested cap by more than 0.01.
- `random F1` is a same-area random baseline, so positive `F1-random` indicates localization better than selecting the same number of pixels randomly.
- Patch-level F1 is reported for context only; this gate is about full-scene operating stability.
