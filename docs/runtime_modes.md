# Runtime Modes

This repository now keeps production and research inference paths separate.

## Production Mode

Production mode is the GUI default and must remain stable.

- Model family: calibrated HyperSIGMA v3
- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Calibration: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`

Use this mode for the backend, GUI, smoke tests, and any reproducible demo or handover.

## Reference Mode

Reference mode is kept for comparison and fallback planning.

- Model family: baseline v3
- Checkpoint: `experiments/runs_v3/baseline_v3_block224/model_best.pt`
- Threshold: `experiments/eval_v3/baseline_best_val_threshold.json`
- Role: reference only, not the GUI default

Use this mode when comparing probability quality or validating that HyperSIGMA regressions are real.

## Research Mode

Research mode covers any future fine-tuning candidate that is not yet frozen.

- Candidate runs: future `experiments/runs_v3/*` or later versioned runs
- Calibration variants beyond temperature scaling
- Alternative split policies
- Alternative threshold policies

Research mode must not overwrite the production deploy config until it passes the same acceptance and freeze steps.
