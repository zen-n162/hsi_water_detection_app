# Full-scene Area-cap Model Comparison

This compares the current HyperSIGMA confmask candidate against the baseline model on the same sampled full-scene area-cap protocol.

| policy | HyperSIGMA F1 | baseline F1 | F1 delta | HyperSIGMA F1-random | baseline F1-random | HyperSIGMA pos rate | baseline pos rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| area_cap_15 | 0.1285 | 0.1106 | 0.0179 | 0.0323 | 0.0193 | 0.1500 | 0.1290 |
| area_cap_20 | 0.1425 | 0.1165 | 0.0260 | 0.0380 | 0.0134 | 0.1996 | 0.1891 |

Notes:
- HyperSIGMA uses calibrated pixel-level scores from the sampled full-scene evaluator.
- Baseline uses one scalar score per patch, repeated over valid pixels, so small area caps can collapse to zero positives when score ties are coarse.
- Positive `F1-random` means the model-selected area beats same-area random valid-pixel selection.
