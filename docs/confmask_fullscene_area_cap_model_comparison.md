# Full-scene Area-cap Model Comparison

This compares the current HyperSIGMA confmask candidate against the baseline model on the same sampled full-scene area-cap protocol.

| policy | HyperSIGMA F1 | baseline F1 | F1 delta | HyperSIGMA F1-random | baseline F1-random | HyperSIGMA pos rate | baseline pos rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| area_cap_05 | 0.0648 | 0.0000 | 0.0648 | 0.0039 | -0.0002 | 0.0491 | 0.0001 |
| area_cap_10 | 0.0977 | 0.0000 | 0.0977 | 0.0097 | -0.0002 | 0.0987 | 0.0001 |
| area_cap_15 | 0.1236 | 0.1008 | 0.0228 | 0.0194 | -0.0014 | 0.1498 | 0.1415 |
| area_cap_20 | 0.1329 | 0.1008 | 0.0321 | 0.0186 | -0.0016 | 0.1999 | 0.1415 |

Notes:
- HyperSIGMA uses calibrated pixel-level scores from the sampled full-scene evaluator.
- Baseline uses one scalar score per patch, repeated over valid pixels, so small area caps can collapse to zero positives when score ties are coarse.
- Positive `F1-random` means the model-selected area beats same-area random valid-pixel selection.
