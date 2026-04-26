# confmask E03 Confidence BCE Full-scene Gate Summary

## Scope

This run executes the next gate after the area-cap post-processing check:

1. Reuse the existing E02 full-scene maps to compare `area_cap_10`, `area_cap_15`, and `area_cap_20` with `min_component_1024`.
2. Train `E03_head_only_confidence_bce` on the same seed13 and seed99 split-quality-stress datasets.
3. Evaluate E03 with the same full-scene gate: seed13/seed99, calibrated maps, `area_cap_10/15/20`, strict and weak-inclusive labels, `min_component_1024`.

The key point is that E03 is judged by full-scene behavior, not by patch-level validation alone.

## E02 Area-cap Recheck

Reusing the saved E02 full-scene maps showed that lowering the area cap does not substantially improve strict precision.

| seed | label_mode | best policy | precision | recall | F1 |
|---:|---|---|---:|---:|---:|
| 13 | `strict_tier1_2` | `area_cap_20` | `0.0516` | `0.1930` | `0.0814` |
| 13 | `weak_inclusive_tier1_2_3` | `area_cap_10` | `0.0935` | `0.1086` | `0.1005` |
| 99 | `strict_tier1_2` | `area_cap_20` | `0.0540` | `0.2106` | `0.0859` |
| 99 | `weak_inclusive_tier1_2_3` | `area_cap_20` | `0.1021` | `0.2574` | `0.1462` |

This supports the decision to test E03: the current limit is not just over-selecting too much area.

## E03 Patch-level Result

E03 improves patch-level ranking, especially after validation-threshold tuning.

| seed | split/score | ROC-AUC | PR-AUC | precision | recall | F1 | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 13 | test tuned | `0.8056` | `0.8218` | `0.7143` | `0.8333` | `0.7692` | `0.4329` |
| 99 | test tuned | `0.7500` | `0.7708` | `0.6667` | `0.5000` | `0.5714` | `0.7069` |

Patch-level metrics alone would make E03 look promising. The full-scene gate is therefore essential.

## E03 Full-scene Gate

Using calibrated E03 maps with `min_component_1024`:

| seed | label_mode | policy | precision | recall | F1 | selected_rate_search |
|---:|---|---|---:|---:|---:|---:|
| 13 | `strict_tier1_2` | `area_cap_10` | `0.0575` | `0.1115` | `0.0759` | `0.0941` |
| 13 | `strict_tier1_2` | `area_cap_15` | `0.0536` | `0.1605` | `0.0804` | `0.1438` |
| 13 | `strict_tier1_2` | `area_cap_20` | `0.0530` | `0.2117` | `0.0847` | `0.1906` |
| 13 | `weak_inclusive_tier1_2_3` | `area_cap_10` | `0.1063` | `0.1333` | `0.1183` | `0.0941` |
| 13 | `weak_inclusive_tier1_2_3` | `area_cap_15` | `0.0922` | `0.1765` | `0.1211` | `0.1438` |
| 13 | `weak_inclusive_tier1_2_3` | `area_cap_20` | `0.0849` | `0.2155` | `0.1218` | `0.1906` |
| 99 | `strict_tier1_2` | `area_cap_10` | `0.0453` | `0.0898` | `0.0602` | `0.0950` |
| 99 | `strict_tier1_2` | `area_cap_15` | `0.0552` | `0.1637` | `0.0825` | `0.1440` |
| 99 | `strict_tier1_2` | `area_cap_20` | `0.0525` | `0.2084` | `0.0838` | `0.1924` |
| 99 | `weak_inclusive_tier1_2_3` | `area_cap_10` | `0.0822` | `0.1039` | `0.0918` | `0.0950` |
| 99 | `weak_inclusive_tier1_2_3` | `area_cap_15` | `0.1034` | `0.1982` | `0.1359` | `0.1440` |
| 99 | `weak_inclusive_tier1_2_3` | `area_cap_20` | `0.0992` | `0.2540` | `0.1426` | `0.1924` |

## Decision

E03 does not pass the full-scene gate.

Reasons:

- The best strict precision is only `0.0575` on seed13 and `0.0552` on seed99.
- The best weak-inclusive precision is only `0.1063` on seed13 and `0.1034` on seed99.
- E03 improves some lower-cap precision points, but it does not clearly beat E02 at the practical full-scene operating point. E02 seed99 `area_cap_20` remains slightly better in weak-inclusive F1: `0.1462` vs E03 `0.1426`.
- Patch-level improvement does not transfer cleanly to full-scene localization.

Do not promote E03 as the new GUI/runtime model.

## Next Work Plan

1. Diagnose full-scene score-map transfer before another training run.
   Compare E02 and E03 maps by selected-pixel overlap, wet/dry score distributions, and false-positive regions. The key question is whether E03 shifts scores globally or actually changes ranking near wet labels.

2. Run one partial-backbone experiment only after that diagnostic.
   The likely next candidate is `E04_dual_last1_confidence_bce` or `E04_spat_last1_confidence_bce` on seed99 first, with a low encoder LR such as `1e-5` to `3e-5`. Keep seed13 as a confirmatory run, not the first expensive run.

3. Tighten the acceptance gate.
   A candidate should beat E02 seed99 on the same full-scene gate, not merely improve patch metrics. A reasonable next gate is weak-inclusive precision `>= 0.12`, strict precision `>= 0.06`, and no worse weak-inclusive F1 than E02 at the chosen area cap.

4. If partial-backbone still fails, revisit label policy.
   The full-scene ceiling may be limited by patch labels that are too coarse for pixel-level water detection. At that point, build tier-aware or pixel-ratio-aware supervision rather than only changing model capacity.
