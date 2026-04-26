# confmask Full-scene Area-cap 15/20 256-patch Summary

## Scope

This is a research-only localization check for the current confmask protocol.

Protocol:

- boundary: `wet_ratio_valid >= 0.08`
- split: `guard32 + split-quality-selected constrained`
- seeds: `42`, `13`, `99`
- full-scene sample: `256` uniformly selected patches
- policies: `area_cap_15`, `area_cap_20`
- random comparator: `64` same-area repeats
- HyperSIGMA score mode: calibrated pixel-level scores
- baseline score mode: raw patch-scalar scores repeated over sampled valid pixels

## Main Result

The 256-patch rerun preserves the 96-patch trend: HyperSIGMA beats baseline and same-area random, but absolute localization remains weak.

| policy | HyperSIGMA F1 | baseline F1 | F1 delta | HyperSIGMA F1-random | baseline F1-random |
|---|---:|---:|---:|---:|---:|
| `area_cap_15` | `0.1285` | `0.1106` | `+0.0179` | `+0.0323` | `+0.0193` |
| `area_cap_20` | `0.1425` | `0.1165` | `+0.0260` | `+0.0380` | `+0.0134` |

Interpretation:

- `area_cap_20` is the stronger sampled-F1 operating point.
- `area_cap_15` is slightly more conservative, but its mean F1 is lower.
- The gap over baseline is present but not large enough to call localization solved.

## Baseline Coarseness Correction

Baseline scores are patch-scalar. A normal threshold area-cap can select whole repeated-score patch regions and can under- or over-shoot caps because of ties. To correct for that, exact area-cap ranking was evaluated by selecting exactly the requested number of sampled valid pixels and breaking score ties deterministically.

| policy | HyperSIGMA exact F1 | baseline exact F1 | HyperSIGMA exact F1-random | baseline exact F1-random |
|---|---:|---:|---:|---:|
| `area_cap_15` | `0.1285` | `0.1178` | `+0.0323` | `+0.0214` |
| `area_cap_20` | `0.1424` | `0.1199` | `+0.0379` | `+0.0152` |

The exact-rank correction narrows the gap at `area_cap_15`, but HyperSIGMA still stays ahead. This argues against the current weak full-scene localization being a HyperSIGMA-only failure relative to the baseline.

## Overlay Check

Generated overlays:

- `outputs/diagnostics_confmask_thr008_fullscene_area_cap_15_20_256_hypersigma/`
- `outputs/diagnostics_confmask_thr008_fullscene_area_cap_15_20_256_baseline_threshold/`
- `outputs/diagnostics_confmask_thr008_fullscene_area_cap_15_20_256_baseline_exact_rank/`

Overlay summary means across seeds:

| model / selection | policy | precision | recall | mean selected tier3 weak pixels |
|---|---:|---:|---:|---:|
| HyperSIGMA threshold | `area_cap_15` | `0.0946` | `0.2005` | `516.0` |
| HyperSIGMA threshold | `area_cap_20` | `0.0965` | `0.2723` | `688.3` |
| baseline threshold | `area_cap_15` | `0.0848` | `0.1606` | `463.7` |
| baseline threshold | `area_cap_20` | `0.0801` | `0.2141` | `572.0` |
| baseline exact-rank | `area_cap_15` | `0.0871` | `0.1847` | `516.3` |
| baseline exact-rank | `area_cap_20` | `0.0811` | `0.2293` | `600.3` |

Visual inspection notes:

- HyperSIGMA selections are not random speckle; they form coherent selected blocks in several wet-label neighborhoods.
- The same overlays still include many dry regions, so precision remains low.
- `area_cap_20` recovers more weak tier-3 wet pixels and more total wet pixels, at the cost of broader dry contamination.
- Baseline exact-rank fills the requested cap more fairly than threshold selection, but it still does not surpass HyperSIGMA.

## Decision

Do not run a new training experiment yet.

Reason:

- The baseline does not beat HyperSIGMA after the larger sample or exact-rank correction.
- The likely next bottleneck is full-scene operating policy / spatial regularization / label-noise handling, not simply the E02 loss.
- A new loss experiment should be delayed until the current model's full-scene post-processing ceiling is clearer.

Recommended operating point for the next diagnostic:

- Use `area_cap_20` when optimizing recall and sampled F1.
- Keep `area_cap_15` as the conservative comparison point for precision/localization enrichment.
