# Rollback Plan For Web Release

This document defines the first-response rollback plan for the Netlify + Render release.

## 1. Frontend-only failure

Symptoms:

- Netlify site fails to load
- UI is broken but backend `/health` still works

Action:

1. identify the last known-good Netlify deploy
2. restore or redeploy that previous Netlify production deploy
3. keep the Render backend untouched
4. rerun:
   - frontend root check
   - `/health`
   - preview smoke test

Check:

- current `VITE_API_BASE_URL`
- current Netlify production deploy ID
- whether frontend env vars changed

## 2. Backend-only failure

Symptoms:

- Netlify loads but preview/inference fail
- Render `/health` fails or inference endpoints fail

Action:

1. inspect Render logs and events
2. compare current env vars against the approved matrix
3. if the bad deploy introduced code/config regressions, roll back to the previous Render deploy
4. verify `/health`
5. rerun preview and inference acceptance

Check:

- `HSI_DEPLOY_CONFIG`
- `HSI_RUNTIME_ROOT`
- `HSI_OUTPUT_ROOT`
- `HSI_CORS_ALLOW_ORIGINS`
- mounted disk contents

## 3. Model asset path failure

Symptoms:

- `/health` works
- `/inference/deploy-config` works
- inference fails with missing file or import errors

Action:

1. open Render shell
2. verify:
   - `runtime/assets/models/.../model_best.pt`
   - `runtime/assets/models/.../temperature_scaling_val.json`
   - `runtime/assets/manifests/...`
   - `runtime/assets/upstream/HyperSIGMA/...`
3. if a file was replaced incorrectly, restore the previous asset copy
4. restart the backend service
5. rerun live acceptance

Check:

- [configs/deploy/hypersigma_v3_calibrated_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_calibrated_web.json)
- Render shell file layout
- `/health` override fields

## 4. CORS failure

Symptoms:

- frontend loads
- browser shows cross-origin fetch failures
- direct backend endpoint access works

Action:

1. compare Netlify site domain with `HSI_CORS_ALLOW_ORIGINS`
2. add the exact production and allowed preview domains
3. redeploy or restart the backend if needed
4. refresh the browser and retry preview

Check:

- frontend origin URL
- `HSI_CORS_ALLOW_ORIGINS`
- browser network panel

## 5. Previous deploy rollback sequence

### Netlify

1. locate the last known-good production deploy
2. restore that deploy
3. confirm the site root loads

### Render

1. locate the last known-good deploy in `Events`
2. roll back the service to that deploy
3. confirm `/health`
4. confirm disk assets are still intact

## 6. What to verify after any rollback

- frontend production URL
- backend production URL
- `/health`
- `/inference/deploy-config`
- one preview request
- one inference request
- `metadata.json`

## 7. Config and file locations to inspect first

- [render.yaml](/home/zennakamura/MasterResearch/hsi_water_detection_app/render.yaml)
- [frontend/netlify.toml](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/netlify.toml)
- [configs/deploy/hypersigma_v3_calibrated_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_calibrated_web.json)
- [docs/production_env_matrix.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/production_env_matrix.md)
- [docs/render_runtime_asset_layout.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/render_runtime_asset_layout.md)
