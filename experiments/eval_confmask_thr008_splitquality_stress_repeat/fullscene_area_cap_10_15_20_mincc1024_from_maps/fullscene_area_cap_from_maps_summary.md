# Full-scene area-cap comparison from saved maps

## Scope

- Reuses saved full-scene probability maps.
- Compares `area_cap_10`, `area_cap_15`, and `area_cap_20` without rerunning HyperSIGMA inference.
- Post-processing variant of interest: `min_component_1024`.
- Label modes: `strict_tier1_2` and `weak_inclusive_tier1_2_3`.

## `min_component` comparison

| seed | label_mode | policy | precision | recall | F1 | selected_rate_search | tier3 selected |
|---:|---|---|---:|---:|---:|---:|---:|
| 13 | strict_tier1_2 | `area_cap_10` | 0.0484 | 0.0874 | 0.0623 | 0.0873 | 2528 |
| 13 | strict_tier1_2 | `area_cap_15` | 0.0500 | 0.1398 | 0.0737 | 0.1348 | 3662 |
| 13 | strict_tier1_2 | `area_cap_20` | 0.0516 | 0.1930 | 0.0814 | 0.1803 | 4850 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.0935 | 0.1086 | 0.1005 | 0.0873 | 2528 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_15` | 0.0922 | 0.1654 | 0.1184 | 0.1348 | 3662 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_20` | 0.0932 | 0.2238 | 0.1316 | 0.1803 | 4850 |
| 99 | strict_tier1_2 | `area_cap_10` | 0.0488 | 0.0959 | 0.0647 | 0.0951 | 2782 |
| 99 | strict_tier1_2 | `area_cap_15` | 0.0534 | 0.1585 | 0.0799 | 0.1437 | 4190 |
| 99 | strict_tier1_2 | `area_cap_20` | 0.0540 | 0.2106 | 0.0859 | 0.1894 | 5894 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.0943 | 0.1193 | 0.1053 | 0.0951 | 2782 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_15` | 0.0985 | 0.1883 | 0.1293 | 0.1437 | 4190 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_20` | 0.1021 | 0.2574 | 0.1462 | 0.1894 | 5894 |

## Best precision by seed and label mode

| seed | label_mode | best policy | precision | recall | F1 | selected_rate_search |
|---:|---|---|---:|---:|---:|---:|
| 13 | strict_tier1_2 | `area_cap_20` | 0.0516 | 0.1930 | 0.0814 | 0.1803 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.0935 | 0.1086 | 0.1005 | 0.0873 |
| 99 | strict_tier1_2 | `area_cap_20` | 0.0540 | 0.2106 | 0.0859 | 0.1894 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_20` | 0.1021 | 0.2574 | 0.1462 | 0.1894 |

## Interpretation rule

- If lower area caps lift strict precision substantially, keep the conservative cap as the current operating baseline.
- If strict precision remains low even at `area_cap_10`, the bottleneck is not just over-selection; move to confidence-aware learning.
