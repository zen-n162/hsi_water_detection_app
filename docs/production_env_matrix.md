# Production Environment Matrix

Date: 2026-04-17

Scope:

- Netlify production static frontend
- Render production backend
- current public production candidate: `configs/deploy/hypersigma_v3_calibrated_web.json`

## Notes

- `VITE_DEFAULT_DEPLOY_NAME` is **not** consumed by the current frontend code, so it is intentionally not part of the active production matrix.
- The backend now accepts both `HSI_DEPLOY_CONFIG` and the older `HSI_DEFAULT_DEPLOY_CONFIG`, but `HSI_DEPLOY_CONFIG` is the canonical production name.
- The backend also supports emergency runtime overrides such as `HSI_MODEL_CHECKPOINT`, but these should normally stay unset so the frozen public deploy config remains the source of truth.

## Frontend

| Name | Required | Example value | Where used | Production value source | Secret |
| --- | --- | --- | --- | --- | --- |
| `VITE_API_BASE_URL` | required | `https://hsi-water-detection-api.onrender.com` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:17), [frontend/src/lib/api.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/api.ts:3) | Netlify Site configuration > Environment variables | no |
| `VITE_APP_MODE` | required | `public` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:16) | Netlify Site configuration > Environment variables | no |
| `VITE_DEFAULT_DEVICE` | optional | `cpu` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:19), [frontend/src/App.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/App.tsx:50) | Netlify Site configuration > Environment variables | no |
| `VITE_ALLOWED_DEVICES` | optional | `cpu` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:20), [frontend/src/components/SidebarControls.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/components/SidebarControls.tsx:8) | Netlify Site configuration > Environment variables | no |
| `VITE_DEFAULT_MODEL_TYPE` | optional | `ss` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:21), [frontend/src/App.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/App.tsx:51) | Netlify Site configuration > Environment variables | no |
| `VITE_DEFAULT_SENSOR` | optional | `hyperion` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:22), [frontend/src/App.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/App.tsx:49) | Netlify Site configuration > Environment variables | no |
| `VITE_SHOW_SERVER_PATH_INPUTS` | optional | `false` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:31), [frontend/src/components/SidebarControls.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/components/SidebarControls.tsx:54) | Netlify Site configuration > Environment variables | no |
| `VITE_SHOW_DEPLOY_CONFIG_INPUT` | optional | `false` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:32), [frontend/src/App.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/App.tsx:128) | Netlify Site configuration > Environment variables | no |
| `VITE_SHOW_ADVANCED_OVERRIDES` | optional | `false` | [frontend/src/lib/runtime.ts](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/lib/runtime.ts:33), [frontend/src/App.tsx](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/src/App.tsx:173) | Netlify Site configuration > Environment variables | no |

## Backend

| Name | Required | Example value | Where used | Production value source | Secret |
| --- | --- | --- | --- | --- | --- |
| `HSI_APP_MODE` | required | `public` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:110), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:82) | `render.yaml` plus Render dashboard review | no |
| `HSI_DEPLOY_CONFIG` | required | `configs/deploy/hypersigma_v3_calibrated_web.json` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:114), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:46) | `render.yaml` | no |
| `HSI_RUNTIME_ROOT` | required | `runtime` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:111), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:83), [src/hsi_water_detection_app/models/hyper_sigma.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/src/hsi_water_detection_app/models/hyper_sigma.py:37) | `render.yaml` | no |
| `HSI_OUTPUT_ROOT` | required | `runtime/outputs` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:117), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:27), [backend/app/services/inference_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/inference_service.py:89) | `render.yaml` | no |
| `HSI_OUTPUT_URL_PREFIX` | required | `/outputs` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:132), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:30), [backend/app/services/preview_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/preview_service.py:26) | `render.yaml` | no |
| `HSI_CORS_ALLOW_ORIGINS` | required | `https://hsi-water-detection.netlify.app,https://deploy-preview-123--hsi-water-detection.netlify.app` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:121), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:18) | Render dashboard environment variable | no |
| `HSI_CORS_ALLOW_CREDENTIALS` | optional | `false` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:148), [backend/app/main.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/main.py:19) | Render dashboard or default | no |
| `HSI_ALLOW_SERVER_FILE_PATHS` | required | `false` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:127), [backend/app/api/routes_preview.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/api/routes_preview.py:45), [backend/app/api/routes_inference.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/api/routes_inference.py:18) | `render.yaml` | no |
| `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE` | required | `false` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:131), [backend/app/api/routes_inference.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/api/routes_inference.py:22) | `render.yaml` | no |
| `HSI_MODEL_CHECKPOINT` | optional | `runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:160), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:67) | Render dashboard only for emergency override | no |
| `HSI_TEMPERATURE_JSON` | optional | `runtime/assets/models/E02_head_only_posw_v3_block224/temperature_scaling_val.json` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:161), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:70) | Render dashboard only for emergency override | no |
| `HSI_THRESHOLD` | optional | `0.327428693347738` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:139), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:73) | Render dashboard only for emergency override | no |
| `HSI_MANIFEST_PATH` | optional | `runtime/assets/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:163), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:76) | Render dashboard only for emergency override | no |
| `HSI_DATASET_PATH` | optional | `runtime/assets/datasets/wetness_pretrain_v3_minimal` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:164), [backend/app/services/deploy_config_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/deploy_config_service.py:79) | Render dashboard only for emergency override | no |
| `HSI_INFERENCE_RUNTIME` | optional | `current` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:136), [backend/app/services/inference_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/inference_service.py:45) | `render.yaml` or default | no |
| `HSI_INFERENCE_CONDA_ENV` | optional | `HyperSIGMA` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:157), [backend/app/services/inference_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/inference_service.py:48) | Render dashboard only if `HSI_INFERENCE_RUNTIME=conda` | no |
| `HSI_INFERENCE_PYTHON_BIN` | optional | `/opt/render/project/src/.venv/bin/python` | [backend/app/settings.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/settings.py:158), [backend/app/services/inference_service.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/app/services/inference_service.py:46) | Render dashboard only if interpreter override is needed | no |
| `HYPERSIGMA_ROOT` | optional | `runtime/assets/upstream/HyperSIGMA` | [src/hsi_water_detection_app/models/hyper_sigma.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/src/hsi_water_detection_app/models/hyper_sigma.py:33) | `render.yaml` or inferred from `HSI_RUNTIME_ROOT` | no |
| `HYPERSIGMA_SPAT_CHECKPOINT` | optional | `runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth` | [src/hsi_water_detection_app/models/hyper_sigma.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/src/hsi_water_detection_app/models/hyper_sigma.py:107) | `render.yaml` or inferred from `HYPERSIGMA_ROOT` layout | no |
| `HYPERSIGMA_SPEC_CHECKPOINT` | optional | `runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth` | [src/hsi_water_detection_app/models/hyper_sigma.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/src/hsi_water_detection_app/models/hyper_sigma.py:115) | `render.yaml` or inferred from `HYPERSIGMA_ROOT` layout | no |

## Shared

There are no first-party application environment variables consumed by both frontend and backend at runtime.

Operationally, record these non-env release values together:

- Netlify production URL
- Render production URL
- Git tag / commit SHA used for the release
- Render disk snapshot or asset manifest reference
