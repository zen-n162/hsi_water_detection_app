# area_cap_20 full-scene post-processing diagnostics

## Label modes

- `strict_tier1_2`: tier1+tier2をwet、tier3は評価から除外。
- `weak_inclusive_tier1_2_3`: tier1+tier2+tier3をwetとして評価。

## Best precision by seed and label mode

| seed | label_mode | raw precision | raw recall | raw F1 | best variant | precision | recall | F1 | selected_rate_search | tier3 selected |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 13 | strict_tier1_2 | 0.0482 | 0.2005 | 0.0777 | `min_component_1024` | 0.0516 | 0.1930 | 0.0814 | 0.1803 | 4850 |
| 13 | weak_inclusive_tier1_2_3 | 0.0886 | 0.2360 | 0.1289 | `min_component_1024` | 0.0932 | 0.2238 | 0.1316 | 0.1803 | 4850 |
| 99 | strict_tier1_2 | 0.0518 | 0.2140 | 0.0835 | `min_component_1024` | 0.0540 | 0.2106 | 0.0859 | 0.1894 | 5894 |
| 99 | weak_inclusive_tier1_2_3 | 0.0993 | 0.2648 | 0.1445 | `min_component_1024` | 0.1021 | 0.2574 | 0.1462 | 0.1894 | 5894 |

## Notes

- ここでのpost-processingは、area_cap_20で選ばれた探索候補からピクセルを削る診断です。面積を再充填していないため、precision改善とrecall低下のトレードオフを見ます。
- `selected_rate_search` は元の探索valid-mask上の選択率です。area_cap_20からどれだけ候補が削られたかを見る指標です。
- overlayはconfidence labelを背景に、選択候補をmagentaで重ねています。探索範囲外はblackです。
