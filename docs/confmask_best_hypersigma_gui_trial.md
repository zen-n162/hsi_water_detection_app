# confmask Best HyperSIGMA GUI Trial Profile

## Scope

This is a research-only GUI trial profile. It does not replace the frozen
`hypersigma_v3_calibrated` production-candidate deploy profile.

## Selected Checkpoint

The default GUI trial profile uses the strongest balanced current HyperSIGMA
candidate from the stress-selected confmask repeat:

- Deploy config: `configs/deploy/hypersigma_confmask_thr008_splitquality_stress_seed13.json`
- Run: `E02_head_only_posw_confmask_splitquality_stress_gb32_thr008_splitquality_stress_seed13`
- Checkpoint: `experiments/runs_confmask_thr008_splitquality_stress_repeat/gb32_thr008_splitquality_stress_seed13/E02_head_only_posw_confmask_splitquality_stress_gb32_thr008_splitquality_stress_seed13/model_best.pt`
- Calibration: `temperature_scaling_val.json`
- Threshold policy: validation F1 tuned
- Threshold: `0.4483769231023357`

Seed13 was selected as the best balanced GUI-inspection candidate. Seed7 has a
slightly higher tiny-test F1 (`0.800` on 8 test patches), but seed13 has stronger
ranking metrics and oracle headroom:

- ROC-AUC: `0.8333333333333335`
- PR-AUC: `0.7999999999999999`
- F1: `0.7692307692307693`
- Oracle F1: `0.8571428571428571`

For the strict highest-F1 checkpoint, use:

- Deploy config: `configs/deploy/hypersigma_confmask_thr008_splitquality_stress_seed7_best_f1.json`
- Run: `E02_head_only_posw_confmask_splitquality_stress_gb32_thr008_splitquality_stress_seed7`
- F1: `0.800`
- ROC-AUC: `0.6875`
- PR-AUC: `0.6458333333333333`

## Known Caveat

This profile is intentionally not promoted to production. On sampled full-scene
coverage, the validation-F1 threshold can still produce a very broad wet extent.
Use it to inspect the current best checkpoint in the GUI, not as a locked
deployment recommendation.

## Backend Startup

Use the helper script to start the backend with this profile as the default:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
./scripts/start_backend_confmask_best_hypersigma.sh
```

Then start the existing frontend dev server:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

The GUI can also use the profile explicitly by setting the deploy config field
to:

```text
configs/deploy/hypersigma_confmask_thr008_splitquality_stress_seed13.json
```
