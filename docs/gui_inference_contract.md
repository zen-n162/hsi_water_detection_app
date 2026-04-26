# GUI Inference Contract

This document defines the deployment-aware contract between the frontend, backend, CLI, and frozen model artifacts.

## 1. Deploy profiles

Local default profile:

- `configs/deploy/hypersigma_v3_calibrated.json`

Public web profile:

- `configs/deploy/hypersigma_v3_calibrated_web.json`

Local GPU web profile:

- `configs/deploy/hypersigma_v3_local_gpu_web.json`

The backend selects the default profile from `HSI_DEPLOY_CONFIG` / `HSI_DEFAULT_DEPLOY_CONFIG`.

## 2. Runtime modes that affect the API

### Local mode

- `HSI_APP_MODE=local`
- `input_path` may be accepted
- `deploy_config_path` override may be accepted
- advanced path overrides may be accepted

### Public mode

- `HSI_APP_MODE=public`
- `input_path` is rejected
- `deploy_config_path` override is rejected
- path-like override fields such as `model_checkpoint` are rejected
- upload-based requests are the supported public path

### Local GPU web mode

- `HSI_APP_MODE=local_gpu_web`
- backend runs on the owner-operated research PC
- upload-based requests are the supported public path
- `input_path` is rejected
- `deploy_config_path` override is rejected
- path-like override fields are rejected
- public responses and `metadata.json` must stay path-safe

## 3. `GET /inference/deploy-config`

Purpose:

- tell the frontend which frozen profile the backend is serving

Optional query parameter:

- `deploy_config_path`

Public mode rule:

- if `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=false`, non-empty `deploy_config_path` must be rejected

External web rule:

- in `public` and `local_gpu_web`, sensitive local path fields must be omitted or masked

Response fields:

- `deploy_config_path`
- `deploy_config_relative`
- `deploy_config`
- `resolved`
- `reference_models`
- `runtime`

`runtime` should expose:

- `app_mode`
- `allow_server_file_paths`
- `allow_deploy_config_override`
- `output_root`
- `output_url_prefix`
- `default_device`
- `public_base_url`
- `provenance_visibility`

## 4. `POST /preview/grayscale`

Supported public request shape:

- `hsi_file`
- optional `wavelength_file`
- `sensor`
- optional preview band controls

Local-only request shape:

- `input_path`

Public mode rule:

- `input_path` must be rejected

## 5. `POST /inference/run`

Required public request fields:

- `hsi_file`
- `sensor`
- `device`
- `row_start`
- `row_stop`
- `col_start`
- `col_stop`

Local-only fields:

- `input_path`
- `deploy_config_path`
- `model_checkpoint`
- `temperature_json`
- `manifest_path`
- `patch_dataset_path`
- `spat_checkpoint`
- `spec_checkpoint`

Resolution rules:

1. backend loads the default deploy config
2. local-only overrides are applied only when the corresponding runtime flags allow them
3. resolved output URLs must point to backend-served `/outputs/...` paths
4. requested and executed device must both be preserved in the response

## 6. Output directory contract

The backend serves runtime artifacts from:

- `HSI_OUTPUT_URL_PREFIX` mounted to `HSI_OUTPUT_ROOT`

Inference outputs are written under:

- `<HSI_OUTPUT_ROOT>/web_ui/<timestamp>/`

Preview outputs are written under:

- `<HSI_OUTPUT_ROOT>/preview_ui/<timestamp>/`

Expected inference artifacts:

- `probability_map.png`
- `probability_overlay.png`
- `spatial_attention_overlay.png`
- `spectral_attention.png`
- `metadata.json`

Local-only artifacts:

- `run_config.json`
- `api_result.json`

External web rule:

- `public` and `local_gpu_web` must not leave path-sensitive sibling artifacts in the statically served output directory

## 7. Health contract

`GET /health` must remain cheap and deployment-safe.

The endpoint should report at least:

- `ok`
- `app_mode`
- `default_deploy_config`
- `default_deploy_config_exists`
- `output_root`
- `output_url_prefix`
- `allow_server_file_paths`
- `allow_deploy_config_override`

## 8. Provenance expectations

Stable response / metadata fields should include:

- `resolved_threshold`
- `resolved_run_name`
- `resolved_output_root`
- `requested_device`
- `executed_device`
- `model_provenance`

Local-only provenance may also include:

- `resolved_model_checkpoint`
- `resolved_temperature_json`
- `resolved_manifest`
- `resolved_dataset`

External web rule:

- `public` and `local_gpu_web` should expose deploy name, run name, threshold, model type, requested device, and executed device
- `public` and `local_gpu_web` should not expose local absolute checkpoint, dataset, manifest, or calibration paths
