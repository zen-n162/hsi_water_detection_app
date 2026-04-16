# Release Notes: HyperSIGMA v3 Calibrated Production Candidate

## Release Title

`hypersigma-v3-calibrated-prod-candidate`

## Production Default

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run name: `E02_head_only_posw_v3_block224`
- GUI default: calibrated HyperSIGMA v3

## Frozen Provenance

- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Temperature JSON: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Dataset: `datasets/processed/wetness_pretrain_v3`

## Verification Status

- GPU acceptance passed
- Deploy consistency audit passed
- GUI acceptance passed
- Release verification passed on the freeze tag commit

## Known Limitations

- The deployed model remains sensitive to split design because the dataset is small.
- Calibration improves probability quality, but does not eliminate all score-shape instability.
- Baseline remains a reference-only path and is not yet exposed in the GUI.

## Reproduction Steps

1. Create the runtime from `configs/env/hypersigma_runtime_conda.yaml`.
2. Install frontend dependencies with `npm --prefix frontend ci`.
3. Confirm CUDA with `research/evaluation/check_cuda_runtime.py`.
4. Start the backend with `PYTHONPATH=src:. MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`.
5. Start the frontend with `npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173`.
6. Run `research/evaluation/smoke_test_gui_pipeline.py` with the deploy config.

## Suggested Release Assets

- `configs/deploy/hypersigma_v3_calibrated.json`
- `experiments/eval_v3/release_freeze.json`
- `experiments/eval_v3/runtime_versions.json`
- `experiments/eval_v3/release_verification.json`
- `docs/final_hypersigma_gui_handover.md`
- `docs/model_card_hypersigma_v3_calibrated.md`
