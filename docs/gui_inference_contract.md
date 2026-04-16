# GUI Inference Contract

This document fixes the deploy-ready inference contract for the calibrated HyperSIGMA v3 GUI path.

## Deploy Default

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Temperature scaling: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`
- Split policy: `spatial_block(block_size=224)`

## GET `/inference/deploy-config`

Returns the current deploy profile that the frontend should treat as the default state.

Query params:

- `deploy_config_path` optional. When omitted, the backend uses `configs/deploy/hypersigma_v3_calibrated.json`.

Response fields:

- `deploy_config_path`
- `deploy_config_relative`
- `deploy_config`
- `resolved`
- `reference_models`

The `resolved` object contains the absolute paths and concrete values the backend will use if the request provides no override.

## POST `/inference/run`

The frontend uploads an HSI file plus ROI coordinates and may optionally override deploy defaults.

Required multipart form fields:

- `hsi_file`
- `sensor`
- `device`
- `row_start`
- `row_stop`
- `col_start`
- `col_stop`

Optional override fields:

- `deploy_config_path`
- `wavelength_file`
- `model_type`
- `model_checkpoint`
- `temperature`
- `temperature_json`
- `decision_threshold`
- `manifest_path`
- `patch_dataset_path`
- `split_policy`
- `spat_checkpoint`
- `spec_checkpoint`
- `patch_size`
- `stride`
- `xmin`
- `ymin`
- `xmax`
- `ymax`

Resolution rules:

1. The backend loads the deploy config first.
2. Any non-empty request override replaces the deploy default.
3. `model_checkpoint` is required after resolution.
4. If `temperature_json` is present, calibration is applied.
5. If `decision_threshold` is omitted, the deploy-config threshold is used.

## Response JSON

The backend returns a JSON payload with these stable provenance fields:

- `deploy_config_path`
- `resolved_model_checkpoint`
- `resolved_temperature_json`
- `resolved_temperature`
- `resolved_threshold`
- `resolved_manifest`
- `resolved_dataset`
- `resolved_split_policy`
- `resolved_run_name`
- `resolved_band_count`
- `resolved_model_type`
- `resolved_patch_size`
- `resolved_stride`
- `requested_device`
- `executed_device`
- `model_provenance`
- `files`
- `urls`

`model_provenance` mirrors the deployed research artifact identity:

- `deploy_config_path`
- `deploy_name`
- `run_name`
- `manifest_path`
- `patch_dataset_path`
- `model_checkpoint_path`
- `calibration_file_path`
- `temperature`
- `threshold`
- `split_policy`
- `band_count`
- `reference_models`

## Output Directory

Each run writes to `outputs/web_ui/<timestamp>/`.

Files that must exist for an accepted GUI inference run:

- `probability_map.png`
- `probability_overlay.png`
- `spatial_attention_overlay.png`
- `spectral_attention.png`
- `metadata.json`
- `run_config.json`
- `api_result.json`

`metadata.json` is the stable artifact-level provenance file for saved inference outputs.
