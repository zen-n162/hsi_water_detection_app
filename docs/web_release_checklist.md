# Web Release Checklist

This checklist is intentionally ordered by execution sequence for the Netlify + Render release.

## 1. Git clean / tag / freeze confirmation

- [ ] `git status` has been reviewed and the intended release scope is understood
- [ ] the public production candidate is [configs/deploy/hypersigma_v3_calibrated_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_calibrated_web.json)
- [ ] freeze and provenance documents are present and unchanged
- [ ] the release commit/tag to publish has been recorded
- [ ] no secrets or private keys are committed

## 2. Frontend build confirmation

- [ ] `npm --prefix frontend run build` succeeds
- [ ] the built frontend title is `HSI Water Detection UI`
- [ ] public mode env defaults are confirmed in [frontend/.env.production.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/.env.production.example)

## 3. Backend import / local boot confirmation

- [ ] `python -m compileall backend src` succeeds
- [ ] importing `backend.app.main:app` succeeds
- [ ] local `/health` returns `ok: true`
- [ ] the backend reports the expected public deploy config when configured for public mode

## 4. Environment variable confirmation

- [ ] frontend production env vars are finalized from [docs/production_env_matrix.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/production_env_matrix.md)
- [ ] backend production env vars are finalized from [docs/production_env_matrix.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/production_env_matrix.md)
- [ ] `HSI_CORS_ALLOW_ORIGINS` contains the exact Netlify production domain
- [ ] optional emergency override envs remain unset unless intentionally used

## 5. Render disk preparation

- [ ] Render persistent disk is attached at `/opt/render/project/src/runtime`
- [ ] runtime directory tree matches [docs/render_runtime_asset_layout.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/render_runtime_asset_layout.md)
- [ ] required runtime assets are copied to the disk
- [ ] output directories exist under `runtime/outputs`

## 6. Render deploy

- [ ] Render Blueprint or manual Web Service creation uses [render.yaml](/home/zennakamura/MasterResearch/hsi_water_detection_app/render.yaml)
- [ ] build command is `pip install -r backend/requirements.txt`
- [ ] start command is `PYTHONPATH=src:. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- [ ] health check path is `/health`
- [ ] first boot succeeds

## 7. Netlify deploy

- [ ] Netlify project is connected to the GitHub repository
- [ ] Base directory is `frontend`
- [ ] Build command is `npm run build`
- [ ] Publish directory is `dist`
- [ ] production frontend env vars are entered
- [ ] the production site is deployed

## 8. CORS confirmation

- [ ] browser requests from the Netlify origin can reach the Render backend
- [ ] no cross-origin failures are visible in the browser network panel
- [ ] preview and inference requests are not blocked by CORS

## 9. Live preview test

- [ ] frontend root loads
- [ ] deploy info panel appears
- [ ] `/preview/grayscale` succeeds with live upload input
- [ ] grayscale preview image loads in the UI
- [ ] ROI selection UI is visible

## 10. Live inference test

- [ ] `/inference/run` succeeds with live upload input
- [ ] result assets are generated
- [ ] Pseudo Color renders
- [ ] H2O Detection Overlay renders
- [ ] Spatial Attention renders
- [ ] Spectral Attention renders

## 11. Metadata / provenance confirmation

- [ ] `metadata.json` is reachable
- [ ] `resolved_model_checkpoint` is present
- [ ] `resolved_temperature_json` is present
- [ ] `resolved_threshold` is present
- [ ] `resolved_run_name` is present
- [ ] `executed_device` is present
- [ ] saved output URLs resolve from `/outputs/...`

## 12. Post-release rollback record

- [ ] last known-good Netlify deploy ID is recorded
- [ ] last known-good Render deploy ID is recorded
- [ ] runtime asset version/checksum record is stored
- [ ] rollback procedure from [docs/rollback_plan_web_release.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/rollback_plan_web_release.md) is attached to the release record
