# GUI Inference Contract

This document defines the deploy-ready inference contract for the calibrated HyperSIGMA v3 GUI path and clarifies how the frontend, backend, CLI, and frozen research artifacts connect.

## 1. Deploy default

The current GUI production-candidate default is resolved from:

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run name: `E02_head_only_posw_v3_block224`
- Model type: `ss`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Temperature scaling: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`
- Split policy: `spatial_block(block_size=224)`

Frontend defaults must be derived from this deploy config rather than hardcoded per-file assumptions.

## 2. System flow

The deployed inference path is:

1. **Frontend** requests deploy defaults
2. **Backend** resolves deploy config into concrete artifact paths and values
3. **Frontend** submits ROI inference request
4. **Backend** launches inference path and records resolved provenance
5. **CLI / inference code** runs the selected model on the requested ROI
6. **Output artifacts** are written under `outputs/web_ui/<timestamp>/`
7. **API response + metadata.json** both expose resolved checkpoint, threshold, calibration, and run identity

This contract is designed so that the saved output directory is sufficient to reconstruct what model and configuration were used.

## 3. GET `/inference/deploy-config`

Purpose:
- provide the frontend with the current deploy-ready default model configuration

Optional query param:
- `deploy_config_path`

When omitted, backend must use:
- `configs/deploy/hypersigma_v3_calibrated.json`

Response should include at least:
- `deploy_config_path`
- `deploy_config_relative`
- `deploy_config`
- `resolved`
- `reference_models`

`resolved` is the concrete backend interpretation of the deploy config.

## 4. POST `/inference/run`

Required request fields:

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

1. Backend loads deploy config first
2. Request overrides replace non-empty deploy defaults
3. `model_checkpoint` must be available after resolution
4. If `temperature_json` is resolved, calibration is applied
5. If `decision_threshold` is absent, deploy-config threshold is used
6. Backend must record both requested and executed device

## 5. Stable provenance fields in API response

The backend response should preserve these stable keys:

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

`model_provenance` should include:

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

## 6. Output directory contract

Each GUI/backend inference run writes into:

- `outputs/web_ui/<timestamp>/`

Expected artifacts:

- `probability_map.png`
- `probability_overlay.png`
- `spatial_attention_overlay.png`
- `spectral_attention.png`
- `metadata.json`
- `run_config.json`
- `api_result.json`

`metadata.json` is the canonical saved provenance artifact.

At minimum it must expose:
- resolved checkpoint
- resolved threshold
- resolved temperature / calibration file
- run name
- manifest path
- dataset path
- split policy
- requested device
- executed device

## 7. Frontend expectations

The frontend should visibly expose:

- run name
- checkpoint path or compact checkpoint identity
- threshold
- calibration / temperature state

This matters because the GUI is not just a demo viewer. It is the user-facing interface to a frozen research artifact.

## 8. Baseline path status

The current production default is **not** the baseline model.

Baseline remains:
- reference-only
- useful for research comparison
- not yet the default deploy path in the current GUI contract

If a baseline GUI path is later added, it should be introduced as a separate deploy profile or service path rather than mixed into the HyperSIGMA default path.

## 9. Acceptance conditions for “GUI connected”

The GUI path can be considered connected and deploy-ready when all of the following are true:

- deploy config loads successfully
- frontend build succeeds
- backend resolves deploy config consistently
- ROI inference returns success
- output artifacts are saved
- metadata.json includes resolved provenance fields
- API response provenance matches metadata provenance
- executed device is recorded
- smoke test / acceptance audit passes

That condition has already been achieved for the current calibrated HyperSIGMA v3 production-candidate path.
