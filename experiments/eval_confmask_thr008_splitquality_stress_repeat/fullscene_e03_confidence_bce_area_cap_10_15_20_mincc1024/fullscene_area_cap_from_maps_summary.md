# Full-scene area-cap comparison from saved maps

## Scope

- Reuses saved full-scene probability maps.
- Compares `area_cap_10`, `area_cap_15`, and `area_cap_20` without rerunning HyperSIGMA inference.
- Post-processing variant of interest: `min_component_1024`.
- Label modes: `strict_tier1_2` and `weak_inclusive_tier1_2_3`.

## `min_component` comparison

| seed | label_mode | policy | precision | recall | F1 | selected_rate_search | tier3 selected |
|---:|---|---|---:|---:|---:|---:|---:|
| 13 | strict_tier1_2 | `area_cap_10` | 0.0575 | 0.1115 | 0.0759 | 0.0941 | 2985 |
| 13 | strict_tier1_2 | `area_cap_15` | 0.0536 | 0.1605 | 0.0804 | 0.1438 | 3591 |
| 13 | strict_tier1_2 | `area_cap_20` | 0.0530 | 0.2117 | 0.0847 | 0.1906 | 3940 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.1063 | 0.1333 | 0.1183 | 0.0941 | 2985 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_15` | 0.0922 | 0.1765 | 0.1211 | 0.1438 | 3591 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_20` | 0.0849 | 0.2155 | 0.1218 | 0.1906 | 3940 |
| 99 | strict_tier1_2 | `area_cap_10` | 0.0453 | 0.0898 | 0.0602 | 0.0950 | 2248 |
| 99 | strict_tier1_2 | `area_cap_15` | 0.0552 | 0.1637 | 0.0825 | 0.1440 | 4497 |
| 99 | strict_tier1_2 | `area_cap_20` | 0.0525 | 0.2084 | 0.0838 | 0.1924 | 5802 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.0822 | 0.1039 | 0.0918 | 0.0950 | 2248 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_15` | 0.1034 | 0.1982 | 0.1359 | 0.1440 | 4497 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_20` | 0.0992 | 0.2540 | 0.1426 | 0.1924 | 5802 |

## Best precision by seed and label mode

| seed | label_mode | best policy | precision | recall | F1 | selected_rate_search |
|---:|---|---|---:|---:|---:|---:|
| 13 | strict_tier1_2 | `area_cap_10` | 0.0575 | 0.1115 | 0.0759 | 0.0941 |
| 13 | weak_inclusive_tier1_2_3 | `area_cap_10` | 0.1063 | 0.1333 | 0.1183 | 0.0941 |
| 99 | strict_tier1_2 | `area_cap_15` | 0.0552 | 0.1637 | 0.0825 | 0.1440 |
| 99 | weak_inclusive_tier1_2_3 | `area_cap_15` | 0.1034 | 0.1982 | 0.1359 | 0.1440 |

## Interpretation rule

- If lower area caps lift strict precision substantially, keep the conservative cap as the current operating baseline.
- If strict precision remains low even at `area_cap_10`, the bottleneck is not just over-selection; move to confidence-aware learning.
