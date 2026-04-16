# Final HyperSIGMA GUI Handover

## Fixed Source Of Truth

The only ground-truth source for the wetness pipeline is `confidence_ali.tif`.

- Label raster: `../HyperSIGMA/HyperspectralDetection/Hyperion_WaterLabel_20111222/data/processed/labels/confidence_ali.tif`
- Manifest used by training/evaluation/GUI provenance: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset used by training/evaluation/GUI provenance: `datasets/processed/wetness_pretrain_v3`

No legacy ROI coordinates from `wetness_manifest.csv` are reused as labels.

## Split Policy

The production-candidate dataset uses spatial block splitting.

- Patch size: `64x64`
- Split policy: `spatial_block(block_size=224)`
- Leakage audit comparison: `experiments/eval_v3/leakage_comparison.json`

This v3 split reduces touching cross-split patch pairs relative to the earlier random split and is the fixed provenance for GUI deployment.

## Best HyperSIGMA Run

The adopted fine-tuned run is:

- Run directory: `experiments/runs_v3/E02_head_only_posw_v3_block224`
- Checkpoint: `model_best.pt`
- Calibration: `temperature_scaling_val.json`
- Validation-tuned threshold: `best_val_threshold_calibrated.json`

The deploy config that fixes these choices is:

- `configs/deploy/hypersigma_v3_calibrated.json`

## Production Default

The production default is the calibrated HyperSIGMA v3 deploy profile and not just the raw fine-tuned checkpoint.

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- GUI default: calibrated HyperSIGMA v3
- Backend default: deploy-config-driven `GET /inference/deploy-config` plus `POST /inference/run`
- Baseline: reference only

This means the operational default is the tuple of checkpoint, calibration file, threshold, manifest, dataset, and split policy.

## Research Best Model Vs GUI Default

The research best fine-tuned HyperSIGMA candidate and the GUI default currently align, but they are not conceptually the same thing.

- Research best model: `E02_head_only_posw_v3_block224`
- GUI default: calibrated deployment of `E02_head_only_posw_v3_block224`

If a future fine-tuned run ranks first in research but has weaker calibration or integration stability, it should remain a research candidate until it passes the same freeze and acceptance steps.

## What Calibration Changed

The raw HyperSIGMA scores collapsed toward zero on v3, which preserved ranking signal but made fixed thresholds unstable.

Temperature scaling was fit on the validation split only and then applied to test and GUI inference.

- Calibration preserves ROC-AUC and PR-AUC.
- Calibration materially improves probability spread and ECE/Brier versus uncalibrated HyperSIGMA.
- The validation-tuned threshold is stored separately and reused by the GUI deploy config.

## Calibration And Threshold Meaning

- `temperature_scaling_val.json` stores a calibration transform fit on the validation split only.
- `best_val_threshold_calibrated.json` stores the validation-selected operating threshold after calibration.
- The GUI and backend do not fit either of these on test or during inference.
- The production deploy config points to both so inference remains reproducible.

## Why The GUI Default Is HyperSIGMA Calibrated v3

The GUI default is fixed to calibrated HyperSIGMA v3 because it balances research quality and product fit.

- It ties the baseline on tuned test F1 in the current v3 evaluation.
- It preserves higher PR-AUC than the baseline.
- It already matches the existing HyperSIGMA attention-based inference path used by the GUI.
- Calibration reduces the worst score-collapse behavior seen in the raw fine-tuned outputs.

The baseline remains available as a backend reference path because its tuned test Brier score and ECE are still slightly better.

## Baseline Reference Path

The baseline remains part of the release as a comparison and fallback planning artifact.

- Checkpoint: `experiments/runs_v3/baseline_v3_block224/model_best.pt`
- Threshold JSON: `experiments/eval_v3/baseline_best_val_threshold.json`
- Role: reference only

It is intentionally not the GUI default, but it stays in deploy provenance so future work can compare against a stable non-attention path.

## Deploy Artifacts

The artifact index is:

- `experiments/eval_v3/artifacts_manifest.json`
- `experiments/eval_v3/release_freeze.json`

The GUI default provenance is:

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Temperature JSON: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`

## GUI / Backend Contract

- Backend deploy profile endpoint: `GET /inference/deploy-config`
- Backend inference endpoint: `POST /inference/run`
- Contract doc: `docs/gui_inference_contract.md`
- Acceptance smoke test: `research/evaluation/smoke_test_gui_pipeline.py`

Every saved GUI inference run writes `metadata.json` with the resolved deploy config, manifest, dataset, checkpoint, calibration file, threshold, and split policy.

## Freeze Production Candidate

The production candidate is frozen as the calibrated HyperSIGMA v3 deploy profile.

- Release freeze manifest: `experiments/eval_v3/release_freeze.json`
- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Backend entrypoint: `backend.app.main:app`
- GUI entrypoint: `frontend/src/App.tsx`
- Smoke test: `research/evaluation/smoke_test_gui_pipeline.py`

This freeze keeps the GUI default fixed on calibrated HyperSIGMA v3 and keeps the baseline as a reference-only artifact.

## GUI Acceptance Criteria

The GUI path is considered accepted when all of the following are true.

- The backend can load `configs/deploy/hypersigma_v3_calibrated.json` by default.
- The frontend renders run name, checkpoint, threshold, and calibration state from the deploy profile.
- ROI inference completes and saves output artifacts.
- `probability_overlay.png` is produced.
- `metadata.json` is produced.
- `metadata.json` includes resolved checkpoint, threshold, calibration file, run name, and executed device.
- The acceptance record is saved to `experiments/eval_v3/gui_acceptance_result.json`.

## Known Constraints

- The frontend depends on the backend to provide the deploy profile.
- The deployed model is still sensitive to split design because the total patch count is small.
- Baseline and HyperSIGMA remain close enough that future spatial split revisions could change the default recommendation.
- Temperature scaling improves calibration, but it does not remove all ranking instability.

## Reproduction

The shortest reproduction path after the freeze is:

1. Create the runtime environment from `configs/env/hypersigma_runtime_conda.yaml`.
2. Install frontend dependencies with the existing `frontend/package-lock.json`.
3. Run `research/evaluation/check_cuda_runtime.py` to confirm CUDA visibility.
4. Start the backend with `PYTHONPATH=src:. MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`.
5. Start the frontend with `npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173`.
6. Run `research/evaluation/smoke_test_gui_pipeline.py` for the acceptance smoke test.

The concrete commands are tracked in `experiments/eval_v3/commands_run.txt`.

For a single entry-point helper, use:

- `scripts/reproduce_release_candidate.sh`

For release packaging:

- Release notes: `docs/release_notes_hypersigma_v3_calibrated.md`
- Release assets manifest: `experiments/eval_v3/release_assets_manifest.json`

## Next Improvements

- Add a baseline inference path to the backend if GUI-side fallback selection becomes necessary.
- Re-run v3 with a larger block size or scene-level partition if more conservative leakage control is required.
- Add richer calibration diagnostics to the GUI result metadata if probability quality becomes a primary UI concern.
