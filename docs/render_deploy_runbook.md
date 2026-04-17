# Render Deploy Runbook

This runbook defines the exact Render setup for the production backend.

## Preferred method

Use the repository root [render.yaml](/home/zennakamura/MasterResearch/hsi_water_detection_app/render.yaml) as the Blueprint source of truth.

## Blueprint usage

1. In Render, choose `New +`
2. Choose `Blueprint`
3. Connect the GitHub repository
4. Point Render to the repository root so it finds `render.yaml`
5. Review the generated service configuration before applying

## Service values

Expected service configuration from `render.yaml`:

- type: `web`
- runtime: `python`
- build command: `pip install -r backend/requirements.txt`
- start command: `PYTHONPATH=src:. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- health check path: `/health`

## Root directory

Use repository root.

- Root directory: leave unset or set to repo root

Do not set root directory to `backend`, because:

- `src/` must remain available on `PYTHONPATH`
- deploy config and docs live outside `backend/`

## Persistent disk

Create one persistent disk:

- mount path: `/opt/render/project/src/runtime`
- initial size: `5 GB` minimum for the current public demo profile

This must exist before the service can retain:

- mounted runtime assets
- generated preview/inference outputs

## Environment variables

Required variables:

```text
HSI_APP_MODE=public
HSI_RUNTIME_ROOT=runtime
HSI_DEPLOY_CONFIG=configs/deploy/hypersigma_v3_calibrated_web.json
HSI_OUTPUT_ROOT=runtime/outputs
HSI_OUTPUT_URL_PREFIX=/outputs
HSI_ALLOW_SERVER_FILE_PATHS=false
HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=false
HSI_INFERENCE_RUNTIME=current
HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app
HYPERSIGMA_ROOT=runtime/assets/upstream/HyperSIGMA
HYPERSIGMA_SPAT_CHECKPOINT=runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth
HYPERSIGMA_SPEC_CHECKPOINT=runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth
```

Optional emergency overrides that should normally stay unset:

```text
HSI_MODEL_CHECKPOINT
HSI_TEMPERATURE_JSON
HSI_THRESHOLD
HSI_MANIFEST_PATH
HSI_DATASET_PATH
HSI_INFERENCE_CONDA_ENV
HSI_INFERENCE_PYTHON_BIN
```

## Runtime asset loading

After first boot:

1. open the Render shell
2. create the directory structure under `/opt/render/project/src/runtime`
3. transfer the files described in [render_runtime_asset_layout.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/render_runtime_asset_layout.md)
4. restart the service

## First boot checks

Check these endpoints in order:

1. `GET /health`
2. `GET /inference/deploy-config`
3. `POST /preview/grayscale` with a sample HSI upload
4. `POST /inference/run` with a sample HSI upload and ROI

The backend is not release-ready until all four succeed.

## Logs and diagnostics

Check these places in the Render dashboard:

- `Logs` for runtime import errors, missing files, and CORS-related failures
- `Events` for deploy failures and restart history
- `Shell` for on-disk asset verification
- `Metrics` if the service restarts under memory pressure

## Common first-boot failure patterns

| Symptom | Likely cause | First check |
| --- | --- | --- |
| `/health` fails | import/runtime problem | service logs |
| `/health` ok but `/inference/deploy-config` wrong | wrong `HSI_DEPLOY_CONFIG` or env override | environment variables and resolved config |
| preview works but inference fails | missing model assets or upstream HyperSIGMA tree | `runtime/assets` layout |
| assets generate but frontend cannot read them | wrong `HSI_OUTPUT_ROOT`, mount, or CORS | `/health`, output URLs, CORS value |

## Manual fallback if not using Blueprint

If the Blueprint flow is unavailable, create a Web Service manually with:

- Environment: Python
- Root directory: repo root
- Build command: `pip install -r backend/requirements.txt`
- Start command: `PYTHONPATH=src:. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Persistent disk mount: `/opt/render/project/src/runtime`

Then copy the same environment variables from this runbook.
