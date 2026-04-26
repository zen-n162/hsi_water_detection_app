# confmask Area-cap 20 Full-scene Post-processing Summary

## Scope

This run tests whether the current HyperSIGMA confmask candidate can be made precise enough by full-scene operating policy and light spatial post-processing, before starting a new training experiment.

Protocol:

- model family: `E02_head_only_posw_confmask_splitquality_stress`
- selected seeds: `13` and `99`
- map mode: calibrated full-scene probability
- full-scene inference: `patch_size=64`, `stride=64`, exhaustive reconstruction
- operating policy: `area_cap_20`
- label modes:
  - `strict_tier1_2`: tier1+tier2 are wet, tier3 is excluded from evaluation
  - `weak_inclusive_tier1_2_3`: tier1+tier2+tier3 are wet

Generated outputs:

- full-scene maps and threshold summaries: `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap20_exhaustive_seed13_seed99/`
- post-processing metrics and overlays: `experiments/eval_confmask_thr008_splitquality_stress_repeat/fullscene_area_cap20_postprocessing_seed13_seed99/`

The large `.npy` probability maps are local experiment artifacts. The committed summaries and overlays are the intended lightweight record.

## Full-scene Area-cap 20 Baseline

Using the calibrated full-scene maps with the original `area_cap_20` threshold:

| seed | label mode | precision | recall | F1 | selected rate |
|---:|---|---:|---:|---:|---:|
| 13 | `strict_tier1_2` | `0.0482` | `0.2005` | `0.0777` | `0.1999` |
| 13 | `weak_inclusive_tier1_2_3` | `0.0886` | `0.2360` | `0.1289` | `0.1999` |
| 99 | `strict_tier1_2` | `0.0518` | `0.2140` | `0.0835` | `0.1998` |
| 99 | `weak_inclusive_tier1_2_3` | `0.0993` | `0.2648` | `0.1445` | `0.1998` |

Seed99 remains the stronger operating candidate, but the strict precision is still only about `0.052`.

## Post-processing Result

The best tested variant was `min_component_1024`, which removes connected components smaller than 1024 pixels.

| seed | label mode | raw precision | raw recall | raw F1 | best variant | precision | recall | F1 | selected rate |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 13 | `strict_tier1_2` | `0.0482` | `0.2005` | `0.0777` | `min_component_1024` | `0.0516` | `0.1930` | `0.0814` | `0.1803` |
| 13 | `weak_inclusive_tier1_2_3` | `0.0886` | `0.2360` | `0.1289` | `min_component_1024` | `0.0932` | `0.2238` | `0.1316` | `0.1803` |
| 99 | `strict_tier1_2` | `0.0518` | `0.2140` | `0.0835` | `min_component_1024` | `0.0540` | `0.2106` | `0.0859` | `0.1894` |
| 99 | `weak_inclusive_tier1_2_3` | `0.0993` | `0.2648` | `0.1445` | `min_component_1024` | `0.1021` | `0.2574` | `0.1462` | `0.1894` |

Other tested variants:

- valid-mask erosion improved precision slightly in some cases, but less than `min_component_1024`.
- local score percentile filters did not beat connected-component filtering.
- thin isolated component removal had almost no effect after component filtering, which means many false positives are broad connected regions rather than narrow isolated artifacts.

## Interpretation

Post-processing helps, but only weakly. The best weak-inclusive precision reaches about `0.102` on seed99, and strict precision reaches about `0.054`. This is not enough to treat the current model as a solved full-scene detector.

The overlays show coherent selected blocks rather than random speckle. That is useful because the model is learning a spatial signal, but it also means simple denoising cannot remove the main false-positive mass. The remaining errors look like large high-score regions, so the next improvement likely needs better supervision or calibration, not just cleanup after prediction.

## Decision

The condition for moving beyond post-processing is now met: precision remains weak after connected-component filtering, valid-mask erosion, local percentile filtering, and thin-isolated removal.

Do not promote this model to GUI/production as the final detector. Keep it as the current calibrated HyperSIGMA baseline and start a controlled learning experiment.

## Next Work Plan

1. Re-evaluate existing full-scene maps at `area_cap_10`, `area_cap_15`, and `area_cap_20` without rerunning inference.
   This is a cheap threshold-only check. Use the same strict/weak label modes and `min_component_1024`. If `area_cap_10` or `area_cap_15` materially improves strict precision, keep that as the conservative operating baseline for later comparisons.

2. Start `E03_head_only_confidence_bce`.
   Use confidence-tier-aware supervision rather than treating all weak labels equally. The acceptance gate should be full-scene, not only patch validation: seed99 post-processed weak precision should clearly exceed `0.12`, strict precision should approach or exceed `0.06`, and the selected area should remain bounded.

3. If E03 does not improve the full-scene gate, run partial backbone fine-tuning.
   Keep the backbone mostly frozen and unfreeze only the last spatial/spectral blocks with a lower learning rate. The goal is to adapt localization features without destabilizing the pretrained HyperSIGMA representation.

4. Keep overlay review in the loop.
   For every candidate, render seed99 and seed13 overlays under the same policy. A model that improves patch metrics but moves the selected full-scene blocks away from labeled wet neighborhoods should be rejected.
